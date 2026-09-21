"""Real, randomly initialized HF text backbones; no model/network downloads."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", minversion="5.3.0")
ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


semantic = _load_module("semantic_qwen35_test", "verl/workers/semantic_value.py")
adapter = _load_module("semantic_qwen35_adapter_test", "verl/models/transformers/qwen3_5.py")


class TinyTokenizer:
    bos_token_id = 1
    special_tokens_map = {"bos_token": "<bos>"}

    def encode(self, text, **kwargs):
        return [3 + ord(char) % 61 for char in text]

    def get_vocab(self):
        return {str(index): index for index in range(64)}


@pytest.fixture(autouse=True)
def isolated_adapter(monkeypatch):
    # This test doesn't import verl's optional Ray/TensorDict package runtime.
    monkeypatch.setitem(sys.modules, "verl.models.transformers.qwen3_5", adapter)
    torch.set_num_threads(1)


def tiny_model(moe=False, attention="eager"):
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=1024, head_dim=16, linear_conv_kernel_dim=3,
                linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                 "partial_rotary_factor": 1., "mrope_section": [2, 2, 4]})
    if moe:
        text.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16,
                    shared_expert_intermediate_size=16, output_router_logits=False)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                  patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
                  out_hidden_size=32, num_position_embeddings=16)
    config_class = transformers.Qwen3_5MoeConfig if moe else transformers.Qwen3_5Config
    model_class = transformers.Qwen3_5MoeForConditionalGeneration if moe else transformers.Qwen3_5ForConditionalGeneration
    config = config_class(text_config=text, vision_config=vision,
                          image_token_id=60, video_token_id=61,
                          vision_start_token_id=62, vision_end_token_id=63)
    config._attn_implementation = attention
    config.text_config._attn_implementation = attention
    config.vision_config._attn_implementation = attention
    with torch.random.fork_rng():
        torch.manual_seed(43)
        return model_class(config).eval()


def scorer_from_model(model):
    class TinyScorer(semantic.PrefixValueScorer):
        def _load_backbone(self):
            adapter.install_qwen3_5_padding_fix(model.config)
            return semantic._extract_qwen35_language_model(model, model.config), TinyTokenizer()
    return TinyScorer({"model_path": "tiny-official-model", "max_length": 1024,
                       "head_hidden_dim": 8, "dropout": 0., "train_epochs": 1})


def record(question="What is 1+1?"):
    return {"traj_uid": "one", "question": question, "label": 1., "max_solver_turns": 2,
            "actions": [{"role": "solver", "text": "3"},
                        {"role": "verifier", "text": "<verify>reject</verify>"},
                        {"role": "solver", "text": "2"}]}


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_real_hybrid_text_prefixes_are_causal_and_stateless(moe, attention):
    scorer = scorer_from_model(tiny_model(moe, attention))
    source = record()
    calls = []
    handle = scorer.encoder.register_forward_pre_hook(lambda *args: calls.append(1))
    full, _ = scorer._encode(source)
    handle.remove()
    assert len(calls) == 1  # Every observed prefix comes from one causal pass.
    for count in range(len(source["actions"])):
        prefix = {**source, "actions": source["actions"][:count]}
        short, _ = scorer._encode(prefix)
        torch.testing.assert_close(full["features"][:count + 1], short["features"], rtol=2e-5, atol=2e-5)
    changed = copy.deepcopy(source)
    changed["label"] = 0.
    changed["actions"][-1]["text"] = "An entirely different future response, with extra tokens."
    future, _ = scorer._encode(changed)
    torch.testing.assert_close(full["features"][:-1], future["features"][:-1], rtol=2e-5, atol=2e-5)
    assert not torch.allclose(full["features"][-1], future["features"][-1])
    scorer._encode(record("A different intervening trajectory with another question."))
    repeated, _ = scorer._encode(source)
    torch.testing.assert_close(full["features"], repeated["features"], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in scorer.encoder.parameters())
    assert scorer.feature_dim == 40


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_complete_official_checkpoint_retains_exact_text_weights(tmp_path, monkeypatch, moe):
    from safetensors.torch import load_file, save_file
    original = tiny_model(moe)
    original.save_pretrained(tmp_path)
    # Official checkpoints may contain MTP tensors. The HF model's explicit
    # ignore rule must consume them before our strict loading-info check.
    weights_path = tmp_path / "model.safetensors"
    # Detach from safetensors' mmap before replacing the same file on Windows.
    checkpoint_weights = {name: value.clone() for name, value in load_file(str(weights_path)).items()}
    checkpoint_weights["mtp.test.weight"] = torch.ones(2, 2)
    save_file(checkpoint_weights, str(weights_path), metadata={"format": "pt"})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    scorer = semantic.PrefixValueScorer({"model_path": str(tmp_path), "head_hidden_dim": 8,
                                        "dropout": 0., "attn_implementation": "eager"})
    expected = original.model.language_model.state_dict()
    actual = scorer.encoder.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert not any("visual" in key or "lm_head" in key for key in actual)
    assert scorer.encoder_identity["source_model_type"] == ("qwen3_5_moe" if moe else "qwen3_5")
    assert scorer.encoder_identity["model_type"] == ("qwen3_5_moe_text" if moe else "qwen3_5_text")
    assert len(scorer.encoder_identity["text_config_sha256"]) == 64
    assert len(scorer.encoder_identity["tokenizer_sha256"]) == 64
    assert len(scorer.encoder_identity["text_weights_sha256"]) == 64
    assert scorer.prepare([record()])["metrics"]["encoded_trajectories"] == 1
    checkpoint = tmp_path / "semantic.pt"
    scorer.save(checkpoint)
    restored = semantic.PrefixValueScorer({"model_path": str(tmp_path), "head_hidden_dim": 8,
                                          "dropout": 0., "attn_implementation": "eager"})
    restored.load(checkpoint, resume=True)
    assert scorer.encoder_identity == restored.encoder_identity


def test_partial_backbone_checkpoint_is_rejected(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    original = tiny_model()
    original.save_pretrained(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    weights = {name: value.clone() for name, value in load_file(str(weights_path)).items()}
    del weights["model.language_model.embed_tokens.weight"]
    save_file(weights, str(weights_path), metadata={"format": "pt"})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    with pytest.raises(ValueError, match="did not load completely.*missing_keys"):
        semantic.PrefixValueScorer({"model_path": str(tmp_path), "attn_implementation": "eager"})


@pytest.mark.parametrize("same_architecture", [False, True])
def test_semantic_checkpoint_rejects_different_backbone(tmp_path, same_architecture):
    dense = scorer_from_model(tiny_model())
    other_model = tiny_model(not same_architecture)
    if same_architecture:
        with torch.no_grad():
            other_model.model.language_model.embed_tokens.weight[0, 0].add_(1.)
    other = scorer_from_model(other_model)
    checkpoint = tmp_path / "head.pt"
    dense.save(checkpoint)
    with pytest.raises(ValueError, match="encoder identity"):
        other.load(checkpoint, warm_start=True)


@pytest.mark.parametrize("field", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("nested", [False, True])
def test_new_and_nested_length_dependent_rope_are_rejected(field, nested):
    rope = {"rope_type": "dynamic", "factor": 2.}
    if nested:
        rope = {"full_attention": rope}
    text = {"model_type": "qwen3_5_text", "hidden_size": 32, field: rope}
    with pytest.raises(ValueError, match="sequence-length-dependent"):
        semantic._validate_backbone_config({"model_type": "qwen3_5", "text_config": text})


def test_mismatched_composite_text_config_is_rejected():
    with pytest.raises(ValueError, match="matching"):
        semantic._validate_backbone_config({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_moe_text"}})


def test_old_qwen3_encoder_identity_is_unchanged():
    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=8, model_type="qwen3", vocab_size=64)
    class OldScorer(semantic.PrefixValueScorer):
        def _load_backbone(self):
            return Encoder(), TinyTokenizer()
    scorer = OldScorer({"model_path": "original-qwen3"})
    assert scorer.encoder_identity == {
        "model_path": "original-qwen3", "revision": None, "model_type": "qwen3",
        "commit_hash": None, "hidden_size": 8, "vocab_size": 64,
        "serialization": "math-prefix-segments-v2", "structural_dim": 8,
    }
