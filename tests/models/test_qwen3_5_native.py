"""Real tiny HF dense/MoE models: native forward, masking, gradient and reload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen35_adapter_test", ROOT / "verl/models/transformers/qwen3_5.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def tiny_config(moe=False):
    from transformers import Qwen3_5Config, Qwen3_5MoeConfig

    text = dict(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128,
                linear_conv_kernel_dim=3, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
                pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2]})
    if moe:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=2,
                  spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=32, num_position_embeddings=16)
    config = (Qwen3_5MoeConfig if moe else Qwen3_5Config)(
        text_config=text, vision_config=vision, image_token_id=45, video_token_id=46, vision_start_token_id=47)
    adapter.configure_qwen3_5_native(config)
    return config


def tiny_model(moe=False):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = tiny_config(moe)
    model = adapter.select_auto_model_class(config).from_config(config)
    adapter.prepare_qwen3_5_for_text_training(model)
    return model


def forward(model, ids, mask=None):
    ids = torch.as_tensor(ids, dtype=torch.long)
    mask = torch.ones_like(ids) if mask is None else torch.as_tensor(mask, dtype=torch.long)
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)


@pytest.mark.parametrize("moe", [False, True])
def test_complete_checkpoint_native_text_forward_backward_and_frozen_vision(moe):
    model = tiny_model(moe).train()
    output = forward(model, [[3, 4, 5, 6], [6, 5, 4, 3]])
    assert output.logits.shape == (2, 4, 48)
    assert output.past_key_values is None
    loss = F.cross_entropy(output.logits[:, :-1].reshape(-1, 48), torch.tensor([4, 5, 6, 5, 4, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(not p.requires_grad and p.grad is None for p in model.model.visual.parameters())
    recurrent = [p for name, p in model.named_parameters() if name.endswith(("A_log", "dt_bias"))]
    assert recurrent and all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in recurrent)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.language_model.parameters())
    assert {m.__class__.__name__ for m in model.modules()}.issuperset(model._no_split_modules)


@pytest.mark.parametrize("moe", [False, True])
def test_single_row_left_padding_has_same_logits_and_gradients_as_unpadded(moe):
    model = tiny_model(moe).eval()
    # Deliberately use a nonzero embedding under the mask: padding correctness
    # must not depend on whether a tokenizer's pad token has a zero embedding.
    padded = forward(model, [[7, 7, 3, 4, 5, 6]], [[0, 0, 1, 1, 1, 1]]).logits[:, 2:]
    loss = padded.square().mean()
    loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    unpadded = forward(model, [[3, 4, 5, 6]]).logits
    unpadded.square().mean().backward()
    torch.testing.assert_close(padded, unpadded, rtol=2e-5, atol=2e-6)
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(gradients[name], p.grad, rtol=5e-4, atol=2e-6)


@pytest.mark.parametrize("moe", [False, True])
def test_batch_rows_and_future_tokens_do_not_change_another_prefix(moe):
    model = tiny_model(moe).eval()
    with torch.no_grad():
        batch = forward(model, [[7, 7, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13]],
                        [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]).logits
        alone = forward(model, [[3, 4, 5, 6]]).logits
        changed = forward(model, [[3, 4, 20, 21]]).logits
    torch.testing.assert_close(batch[0, 2:], alone[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(alone[:, :2], changed[:, :2], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("moe", [False, True])
def test_complete_conditional_checkpoint_save_load_keeps_official_names(moe, tmp_path):
    model = tiny_model(moe).eval()
    model.save_pretrained(tmp_path)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(tmp_path)
    adapter.configure_qwen3_5_native(config)
    loaded, loading_info = adapter.select_auto_model_class(config).from_pretrained(
        tmp_path, config=config, dtype=torch.float32, output_loading_info=True)
    adapter.validate_qwen3_5_loading_info(loading_info)
    adapter.prepare_qwen3_5_for_text_training(loaded)
    loaded.eval()
    assert any(name.startswith("model.visual.") for name in loaded.state_dict())
    assert any(name.startswith("model.language_model.") for name in loaded.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(forward(model, [[3, 4, 5]]).logits, forward(loaded, [[3, 4, 5]]).logits)


@pytest.mark.parametrize("options", [{"use_remove_padding": True}, {"ulysses_sp_size": 2},
                                     {"use_fused_kernels": True}, {"use_liger": True}, {"lora_rank": 8}])
def test_unsafe_attention_only_optimizations_fail_before_allocation(options):
    config = SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace())
    with pytest.raises(ValueError, match="Qwen3.5"):
        adapter.validate_qwen3_5_execution(config, **options)


@pytest.mark.parametrize("nested", [False, True])
def test_quantized_actor_checkpoints_fail_before_allocation(nested):
    config = SimpleNamespace(model_type="qwen3_5_moe", text_config=SimpleNamespace())
    setattr(config.text_config if nested else config, "quantization_config", {"quant_method": "fp8"})
    with pytest.raises(ValueError, match="non-quantized"):
        adapter.validate_qwen3_5_execution(config)


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_or_mismatched_weights_are_not_silently_initialized(key):
    with pytest.raises(ValueError, match="did not load completely"):
        adapter.validate_qwen3_5_loading_info({key: ["model.language_model.layers.0.linear_attn.A_log"]})


def test_qwen3_still_uses_causal_lm_and_adapter_does_not_change_it():
    from transformers import AutoModelForCausalLM, Qwen3Config
    config = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    assert not adapter.is_qwen3_5(config)
    assert adapter.select_auto_model_class(config) is AutoModelForCausalLM
    adapter.validate_qwen3_5_execution(config, use_remove_padding=True, ulysses_sp_size=2)
