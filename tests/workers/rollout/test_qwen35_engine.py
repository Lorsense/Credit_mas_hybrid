"""Check actual rollout constructor arguments without launching GPU servers."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from verl.models.transformers.qwen3_5 import get_text_config, is_qwen3_5


SOURCE = Path(__file__).resolve().parents[3] / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"


def production_functions(**extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_engine_runtime_kwargs", "_engine_port"}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SGLangRollout")
    selected.extend(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_verify_config", "_init_inference_engine"})
    namespace = {"__name__": __name__, "OmegaConf": OmegaConf, "get_text_config": get_text_config,
                 "is_qwen3_5": is_qwen3_5, "asyncio": asyncio, **extra}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace


def config(**updates):
    cfg = OmegaConf.create({
        "max_model_len": None, "prompt_length": 1024, "response_length": 512,
        "multi_turn": {"max_turns": None}, "enforce_eager": True,
        "engine_kwargs": {"sglang": {"attention_backend": None}},
        "load_format": "dummy_dtensor", "dtype": "bfloat16", "gpu_memory_utilization": .4,
        "agent_port": 0,
    })
    return OmegaConf.merge(cfg, updates)


def model_config(qwen35=True):
    text = SimpleNamespace(max_position_embeddings=262144)
    return SimpleNamespace(model_type="qwen3_5", text_config=text) if qwen35 else SimpleNamespace(
        model_type="qwen3", max_position_embeddings=32768)


def test_real_engine_constructor_receives_qwen35_limits_and_strict_loader():
    captured = {}
    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def release_memory_occupation(self):
            captured["released"] = True

    ns = production_functions(Engine=Engine, dist=SimpleNamespace(get_rank=lambda: 0),
                              os=SimpleNamespace(environ={}))
    worker = SimpleNamespace(config=config(engine_kwargs={"sglang": {"attention_backend": "triton"}}),
                             _tp_size=1, _tp_rank=0, visible_devices_set={"0"})
    ns["_verify_config"](worker, model_config())
    ns["_init_inference_engine"](worker, False, "Qwen/Qwen3.5-4B", None)
    assert captured["context_length"] == 1536
    assert captured["disable_cuda_graph"] is True
    assert captured["attention_backend"] == "triton"
    assert captured["load_format"] == "dummy"
    assert captured["custom_weight_loader"] == ["verl.utils.sglang_weight_sync.strict_load_weights"]
    assert captured["released"] and worker.is_sleep


def test_qwen3_legacy_config_needs_no_new_loader():
    ns = production_functions()
    worker = SimpleNamespace(config=config())
    ns["_verify_config"](worker, model_config(False))
    result = ns["_engine_runtime_kwargs"](worker.config, worker.model_hf_config)
    assert result == {"context_length": 1536, "disable_cuda_graph": True}


@pytest.mark.parametrize("override", [{"tp_size": 2}, {"context_length": 999999},
                                     {"disable_cuda_graph": False}, {"quantization": "fp8"},
                                     {"enable_lora": True}])
def test_engine_overrides_cannot_break_qwen35_sync_or_execution(override):
    cfg = config(max_model_len=1536, engine_kwargs={"sglang": override})
    with pytest.raises(ValueError):
        production_functions()["_engine_runtime_kwargs"](cfg, model_config())


def test_nested_model_context_limit_is_checked():
    worker = SimpleNamespace(config=config(max_model_len=300000))
    with pytest.raises(AssertionError, match="model context length"):
        production_functions()["_verify_config"](worker, model_config())


@pytest.mark.parametrize("groups", [2, 3])
def test_sixteen_gpu_engines_get_disjoint_valid_port_ranges(groups):
    env = {}
    port_for = production_functions(os=SimpleNamespace(environ=env))["_engine_port"]
    ports = []
    for local_rank in range(16):
        env["RAY_LOCAL_RANK"] = str(local_rank)
        for agent_index in range(groups):
            ports.append(port_for(config(agent_port=agent_index, agent_count=groups), local_rank + 16))
    assert len(ports) == len(set(ports))
    assert all(right - left > 1000 for left, right in zip(sorted(ports), sorted(ports)[1:]))
    assert max(ports) + 1000 <= 65535


def test_port_range_overflow_fails_before_engine_start():
    port_for = production_functions(os=SimpleNamespace(environ={"RAY_LOCAL_RANK": "99"}))["_engine_port"]
    with pytest.raises(ValueError, match="Too many colocated"):
        port_for(config(), 99)
