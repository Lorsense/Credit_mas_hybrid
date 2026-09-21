"""Weight transport/validation without launching a CUDA SGLang server."""

import ast
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.utils.sglang_weight_sync import (
    STRICT_QWEN35_LOADER,
    flush_weight_caches,
    iter_weight_buckets,
    qwen35_sglang_target_name,
    require_engine_success,
    strict_load_weights,
    update_weight_bucket,
    validate_local_tp_topology,
)

ROOT = Path(__file__).resolve().parents[2]


def test_same_node_tp_and_multinode_fsdp_are_allowed_but_cross_node_ipc_is_rejected():
    validate_local_tp_topology([("node-a", [0, 1]), ("node-a", [0, 1]),
                                ("node-b", [2, 3]), ("node-b", [2, 3])])
    with pytest.raises(ValueError, match="TP group to fit on one node"):
        validate_local_tp_topology([("node-a", [0, 1]), ("node-b", [0, 1])])


@pytest.mark.parametrize("source,target", [
    ("model.language_model.layers.3.self_attn.q_proj.weight", "model.layers.3.qkv_proj.weight"),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.in_proj_qkvz.weight"),
    ("model.language_model.layers.0.linear_attn.in_proj_z.weight", "model.layers.0.linear_attn.in_proj_qkvz.weight"),
    ("model.language_model.layers.0.linear_attn.in_proj_a.weight", "model.layers.0.linear_attn.in_proj_ba.weight"),
    ("model.language_model.layers.0.linear_attn.conv1d.weight", "model.layers.0.linear_attn.conv1d.weight"),
    ("model.language_model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.A_log"),
    ("model.language_model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.w13_weight"),
    ("model.language_model.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.experts.w2_weight"),
    ("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", "model.layers.0.mlp.shared_expert.gate_up_proj.weight"),
    ("model.visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.qkv_proj.weight"),
])
def test_official_dense_and_moe_destinations(source, target):
    assert qwen35_sglang_target_name(source) == target


def test_buckets_are_lazy_bounded_and_do_not_split_stacked_experts():
    parameters = [(str(i), torch.ones(size)) for i, size in enumerate([3, 2, 11, 1])]
    materialized = []
    def gather(tensor):
        materialized.append(tensor.numel())
        return tensor
    buckets = iter_weight_buckets(parameters, max_bytes=20, materialize=gather)
    first = next(buckets)
    assert [name for name, _ in first] == ["0", "1"]
    assert materialized == [3, 2]
    second = next(buckets)
    assert [name for name, _ in second] == ["2"]
    assert materialized == [3, 2, 11]
    assert [name for name, _ in next(buckets)] == ["3"]
    with pytest.raises(StopIteration):
        next(buckets)


@pytest.mark.parametrize("reply", [False, {"success": False, "message": "partial update"}, (False, "failed"), None, {}, "Success"])
def test_failure_or_missing_acknowledgment_is_fatal(reply):
    with pytest.raises(RuntimeError, match="failed or was not acknowledged"):
        require_engine_success(reply, "update")


@pytest.mark.parametrize("reply", [True, {"success": True}, (True, "Success"), SimpleNamespace(success=True)])
def test_documented_acknowledgments(reply):
    require_engine_success(reply, "update")


def test_public_api_gets_full_hf_tensors_and_separate_checked_cache_flush():
    calls = []
    class Engine:
        def update_weights_from_tensor(self, **kwargs):
            calls.append(kwargs)
            return True, "Success"
        def flush_cache(self):
            calls.append("flush_all_cache")
            return True
    tensor = torch.arange(8).reshape(2, 4)
    bucket = [("model.language_model.layers.0.linear_attn.A_log", tensor)]
    engine = Engine()
    update_weight_bucket(engine, bucket, load_format=STRICT_QWEN35_LOADER)
    flush_weight_caches(engine)
    assert calls[0]["named_tensors"][0][1] is tensor
    assert calls[0]["load_format"] == STRICT_QWEN35_LOADER
    assert calls[0]["flush_cache"] is False
    assert calls[1] == "flush_all_cache"


def test_busy_cache_aborts_before_generation():
    with pytest.raises(RuntimeError, match="cache flush"):
        flush_weight_caches(SimpleNamespace(flush_cache=lambda: False))


def _tiny_model(moe):
    spec = importlib.util.spec_from_file_location("weight_sync_tiny_fixture", ROOT / "tests/models/test_qwen3_5_native.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture.tiny_model(moe)


def _upstream_loader(moe):
    # Prefer the pinned installed runtime. CPU development can point to the
    # unmodified v0.5.10 source without importing its CUDA-only dependencies.
    source = os.environ.get("SGLANG_QWEN35_SOURCE")
    if source is None:
        spec = importlib.util.find_spec("sglang")
        if spec is not None:
            source = Path(spec.origin).parent / "srt/models/qwen3_5.py"
    if source is None or not Path(source).is_file():
        pytest.skip("SGLang 0.5.10 source not installed; set SGLANG_QWEN35_SOURCE to test its actual loader")
    name = "Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"
    module = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    namespace = {
        "torch": torch, "logger": logging.getLogger(__name__),
        "get_layer_id": lambda name: None,  # TP1, all decoder layers are local.
        "default_weight_loader": lambda param, tensor: param.weight_loader(param, tensor),
        "FusedMoE": SimpleNamespace(make_expert_params_mapping=lambda **kwargs: []),
        "LazyValue": lambda factory: SimpleNamespace(factory=factory),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
    return name, namespace["load_weights"]


def _backend(model, moe):
    name, loader = _upstream_loader(moe)
    records, parameters = {}, {}
    for source in model.state_dict():
        destination = qwen35_sglang_target_name(source)
        def record(param, tensor, *args, _target=destination, **kwargs):
            records.setdefault(_target, []).append((tensor.detach().clone(), args, kwargs))
        parameters[destination] = SimpleNamespace(weight_loader=record)
    config = model.config.text_config
    config.encoder_only = False
    backend = type(name, (), {"load_weights": loader, "named_parameters": lambda self, **kwargs: parameters.items()})()
    backend.config = config
    backend.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    return backend, records, parameters


@pytest.mark.parametrize("moe", [False, True])
def test_real_transformers_state_dict_through_actual_sglang_0510_loader(moe):
    """Exercise upstream destination selection and expert splitting, not mocks of update success."""
    model = _tiny_model(moe)
    backend, records, parameters = _backend(model, moe)
    state = model.state_dict()
    # Small buckets also verify that native load_weights works across calls.
    for bucket in iter_weight_buckets(state.items(), max_bytes=16 * 1024):
        strict_load_weights(backend, bucket)
    assert set(records) == set(parameters)
    for source, tensor in state.items():
        target = qwen35_sglang_target_name(source)
        actual = records[target]
        if source.endswith(".experts.gate_up_proj"):
            assert len(actual) == 2 * tensor.shape[0]
            for expert in range(tensor.shape[0]):
                for shard, offset in [("w1", 0), ("w3", tensor.shape[0])]:
                    value, args, _ = actual[offset + expert]
                    assert args[-2:] == (shard, expert)
                    torch.testing.assert_close(value, tensor[expert].chunk(2, dim=0)[0 if shard == "w1" else 1])
        elif source.endswith(".experts.down_proj"):
            assert len(actual) == tensor.shape[0]
            for expert, (value, args, _) in enumerate(actual):
                assert args[-2:] == ("w2", expert)
                torch.testing.assert_close(value, tensor[expert])
        else:
            assert any(torch.equal(value, tensor) for value, _, _ in actual), source


@pytest.mark.parametrize("moe", [False, True])
def test_unknown_destination_rejected_before_native_loader_mutates_any_weight(moe):
    model = _tiny_model(moe)
    backend, records, _ = _backend(model, moe)
    weights = list(model.state_dict().items())
    weights.append(("model.language_model.layers.0.typo.weight", torch.ones(1)))
    with pytest.raises(KeyError, match="no destination"):
        strict_load_weights(backend, weights)
    assert not records


def test_wrong_moe_stacked_layout_rejected():
    model = _tiny_model(True)
    backend, records, _ = _backend(model, True)
    source = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with pytest.raises(ValueError, match="Invalid stacked"):
        strict_load_weights(backend, [(source, model.state_dict()[source][..., :-1])])
    assert not records


def test_native_loader_skipping_existing_weight_is_fatal():
    model = _tiny_model(False)
    backend, records, _ = _backend(model, False)
    backend.load_weights = lambda tensors: set()
    with pytest.raises(RuntimeError, match="skipped required"):
        strict_load_weights(backend, [("lm_head.weight", model.lm_head.weight)])


def _manager_methods(dist):
    """Execute the actual manager without importing a CUDA-only SGLang package."""
    from verl.utils.sglang_weight_sync import DEFAULT_WEIGHT_BUCKET_BYTES
    path = ROOT / "verl/workers/sharding_manager/fsdp_sglang.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPSGLangShardingManager")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"update_weights", "_run_engine_operation"}]
    namespace = dict(dist=dist, STRICT_QWEN35_LOADER=STRICT_QWEN35_LOADER,
                     iter_weight_buckets=iter_weight_buckets, update_weight_bucket=update_weight_bucket,
                     flush_weight_caches=flush_weight_caches, _preprocess_tensor_for_update_weights=lambda tensor: tensor)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    manager = namespace[cls.name]()
    manager.qwen35 = True
    manager.tp_rank = 0
    manager.weight_update_bucket_bytes = DEFAULT_WEIGHT_BUCKET_BYTES
    return manager


def test_manager_stops_all_peers_on_engine_failure_without_flushing_partial_weights():
    dist = SimpleNamespace(get_world_size=lambda: 2,
                           all_gather_object=lambda output, error: output.__setitem__(slice(None), [error, None]))
    manager = _manager_methods(dist)
    calls = []
    class Engine:
        def resume_memory_occupation(self):
            calls.append("resume")
        def update_weights_from_tensor(self, **kwargs):
            calls.append("update")
            return False, "injected backend failure"
        def flush_cache(self):
            calls.append("flush")
            return True
    manager.inference_engine = Engine()
    manager.weight_update_bucket_bytes = 4
    with pytest.raises(RuntimeError, match="injected backend failure"):
        manager.update_weights({"one": torch.ones(1), "two": torch.ones(1)})
    assert calls == ["resume", "update"]


def test_nonleader_observes_failure_from_another_engine():
    dist = SimpleNamespace(get_world_size=lambda: 2,
                           all_gather_object=lambda output, error: output.__setitem__(slice(None), ["remote failure", error]))
    manager = _manager_methods(dist)
    manager.tp_rank = 1
    with pytest.raises(RuntimeError, match="remote failure"):
        manager._run_engine_operation(lambda: pytest.fail("a TP follower must not call Engine"))
