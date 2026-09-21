"""Exercise the real logging adapter without contacting the W&B service."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def test_tracking_sends_every_metric_with_global_step_to_wandb(monkeypatch):
    calls = []
    wandb = ModuleType("wandb")
    wandb.init = lambda **kwargs: calls.append(("init", kwargs))
    wandb.log = lambda **kwargs: calls.append(("log", kwargs))
    wandb.finish = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    spec = importlib.util.spec_from_file_location("hybrid_tracking_test", ROOT / "verl/utils/tracking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracking = module.Tracking("hybrid", "observation-test", ["wandb"], {"trainer": {"save_freq": 10}})
    metrics = {"episode/pass@8": .75, "entropy_credit/global/final_multiplier_mean": 1.01,
               "semantic_value/train/candidate_solver_auc": .73, "entropy_control/active_fraction": .25,
               "hybrid/solver/advantage/base_zero_fraction": .2, "trajectory_dump/train/trajectories": 240.}
    tracking.log(metrics, step=10)
    assert calls[0][1]["project"] == "hybrid"
    assert calls[1] == ("log", {"data": metrics, "step": 10})


def test_explicit_wandb_environment_is_carried_without_unrelated_secrets(monkeypatch):
    source = ast.parse((ROOT / "verl/trainer/main_ppo.py").read_text(encoding="utf-8"))
    fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_tracking_env_vars")
    namespace = {"os": os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "main_ppo.py", "exec"), namespace)
    monkeypatch.setattr(os, "environ", {"WANDB_API_KEY": "test-key", "WANDB_ENTITY": "test-team",
                                       "WANDB_MODE": "offline", "UNRELATED_SECRET": "not-forwarded"})
    assert namespace["_tracking_env_vars"]() == {"WANDB_API_KEY": "test-key", "WANDB_ENTITY": "test-team",
                                                "WANDB_MODE": "offline"}
