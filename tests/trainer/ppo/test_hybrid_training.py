"""Driver lifecycle tests use real semantic modules with a synchronous Ray shim."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    credit = load("training_test_metadata", "verl/utils/semantic_credit.py")
    control = load("training_test_control", "verl/utils/semantic_entropy_control.py")
    observability = load("training_test_observability", "verl/utils/hybrid_observability.py")
    worker_tests = load("training_test_worker", "tests/workers/test_semantic_value.py")
    ns = {"np": np, "torch": torch, "OmegaConf": OmegaConf, "json": json, "os": os, "Path": Path,
          "build_trajectory_records": credit.build_trajectory_records,
          "attach_value_predictions": credit.attach_value_predictions,
          "SemanticEntropyController": control.SemanticEntropyController,
          "compute_hybrid_observability": observability.compute_hybrid_observability}
    for path, kinds in (("verl/trainer/ppo/hybrid_credit.py", (ast.FunctionDef,)),
                        ("verl/trainer/ppo/hybrid_training.py", (ast.FunctionDef, ast.ClassDef))):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, kinds)]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), ns)
    ray = ModuleType("ray")
    ray.get = lambda result, **kwargs: result
    monkeypatch.setitem(sys.modules, "ray", ray)
    return ns, worker_tests


class Proxy:
    def __init__(self, scorer):
        self.scorer, self.calls = scorer, []

    def __getattr__(self, name):
        def remote(*args, **kwargs):
            self.calls.append((name, kwargs))
            return getattr(self.scorer, name)(*args, **kwargs)
        return SimpleNamespace(remote=remote)


def driver(runtime, *, semantic=True, control=True):
    ns, worker_tests = runtime
    cfg = OmegaConf.create({"algorithm": {"adv_estimator": "team_event_gae",
        "team_event_gae": {"math_value": {"mode": "math_answer_loto"}},
        "entropy_credit": {"enable": False},
        "semantic_value": {"enable": semantic, "model_path": "tiny-fixed", "require_pretrained": False},
        "semantic_entropy_control": {"enabled": control, "min_calibration_actions": 1,
                                     "min_group_actions": 1, "ramp_steps": 1}},
        "agent": {"orchestra_type": "math", "orchestra": {"math": {"max_loop_num": 2}}},
        "trainer": {"default_hdfs_dir": None, "val_only": False}})
    obj = ns["HybridTrainingMixin"]()
    obj.config = cfg
    obj.wg_to_agents_mapping = {"solver": [{"config_actor_rollout_ref": OmegaConf.create({
        "actor": {"strategy": "fsdp", "entropy_coeff": 0., "entropy_control": {"enabled": control, "loss_coef": .0025}},
        "rollout": {"name": "sglang", "mode": "sync", "top_logprobs_num": 16}})}]}
    obj.global_steps, obj._actor_update_count = 1, 4
    obj._configure_hybrid()
    if semantic:
        obj.semantic_scorer = Proxy(worker_tests.scorer())
    return obj


class Batch:
    def __init__(self):
        self.batch = {"responses": torch.zeros((3, 4), dtype=torch.long),
                      "attention_mask": torch.tensor([[1, 1, 1, 0]] * 3),
                      "advantages": torch.tensor([[1., 1., 1., 0.]] * 3),
                      "returns": torch.tensor([[2., 2., 2., 0.]] * 3)}
        self.non_tensor_batch = {"uid": np.array(["q"] * 3), "traj_uid": np.array(["t"] * 3),
            "event_uid": np.array(["e0", "e1", "e2"]), "event_index": np.array([0, 1, 2]),
            "role_event_index": np.array([0, 0, 1]), "role_turn_index": np.array([0, 0, 1]),
            "agent_id": np.array(["Solver Agent", "Verifier Agent", "Solver Agent"]),
            "value_question": np.array(["complete question"] * 3), "value_max_solver_turns": np.array([2] * 3),
            "value_action_index": np.array([0, 1, 2]),
            "value_action_text": np.array(["first", "<verify>reject</verify>", "last"]),
            "is_action_valid": np.ones(3, bool), "pass": np.ones(3),
            "action_full_entropy": np.ones(3)}

    def __len__(self):
        return 3


def qualify(scorer):
    scorer.ready = scorer.pretrained = True
    scorer.reliability = {"solver": 1., "verifier": 1.}
    scorer.version = 7


def test_prepare_freezes_predictions_then_controller_keeps_advantages_and_returns(runtime):
    obj, batch = driver(runtime), Batch()
    scorer = obj.semantic_scorer.scorer
    # Choose an actual training-split question; a held-out-only batch must not
    # initialize optimizer state. Keep the production split and qualification.
    question = next(f"training question {i}" for i in range(100)
                    if not scorer._is_validation(hashlib.sha256(f"training question {i}".encode()).hexdigest()))
    batch.non_tensor_batch["value_question"] = np.full(len(batch), question, dtype=object)
    qualify(scorer)
    original = copy.deepcopy(scorer.deployed_head.state_dict())
    tensors = {key: value.clone() for key, value in batch.batch.items()}
    obj._prepare_hybrid_rollout(batch)
    assert obj.semantic_ready and batch.non_tensor_batch["value_credit_scorer_version"].tolist() == [7] * 3
    assert len(scorer._pending) == 1 and not scorer.optimizer.state
    assert batch.non_tensor_batch["value_sem_after"][-1] > 0
    obj._apply_hybrid_after_advantage(batch)
    assert batch.batch["entropy_control_valid"].tolist() == [True, True, False]
    assert not set(batch.batch).intersection(batch.non_tensor_batch)
    for key in ("advantages", "returns"):
        torch.testing.assert_close(batch.batch[key], tensors[key])
    # Actor updates occur externally; only this explicit lifecycle call learns.
    obj._update_semantic_after_actors()
    assert scorer.optimizer.state and not scorer._pending
    for key, value in original.items():
        torch.testing.assert_close(value, scorer.deployed_head.state_dict()[key], atol=0, rtol=0)
    assert [name for name, _ in obj.semantic_scorer.calls] == ["prepare", "update"]


def test_missing_required_outcome_metadata_is_not_silently_ignored(runtime):
    obj, batch = driver(runtime), Batch()
    del batch.non_tensor_batch["pass"]
    with pytest.raises(ValueError, match="Missing semantic rollout metadata"):
        obj._prepare_hybrid_rollout(batch)


def test_cold_training_requires_explicit_opt_in_and_candidate_requires_file(runtime):
    obj = driver(runtime)
    obj._initialize_fresh_hybrid()
    assert obj.semantic_scorer.calls == []
    obj.semantic_value_config.require_pretrained = True
    with pytest.raises(ValueError, match="initial_checkpoint"):
        obj._initialize_fresh_hybrid()
    obj.semantic_value_config.require_pretrained = False
    obj.semantic_value_config.initialization_mode = "candidate"
    with pytest.raises(ValueError, match="initial_checkpoint"):
        obj._initialize_fresh_hybrid()


def test_fresh_qualified_and_candidate_load_use_correct_flags(runtime, tmp_path):
    obj = driver(runtime)
    qualify(obj.semantic_scorer.scorer)
    path = tmp_path / "qualified.pt"
    obj.semantic_scorer.scorer.save(path)
    obj.semantic_value_config.initial_checkpoint = str(path)
    obj._initialize_fresh_hybrid()
    assert obj.semantic_scorer.calls[-1] == ("load", {"resume": False, "warm_start": False})
    obj.semantic_value_config.initialization_mode = "candidate"
    obj._initialize_fresh_hybrid()
    assert obj.semantic_scorer.calls[-1] == ("load", {"resume": False, "warm_start": True})
    assert not obj.semantic_scorer.scorer.ready


def test_resume_restores_scorer_pending_controller_and_actor_counter(runtime, tmp_path):
    obj = driver(runtime)
    qualify(obj.semantic_scorer.scorer)
    batch = Batch()
    obj._prepare_hybrid_rollout(batch)
    obj._apply_hybrid_after_advantage(batch)
    (tmp_path / "data.pt").write_bytes(b"transport-state")
    obj._save_hybrid_checkpoint(tmp_path)
    other = driver(runtime)
    other._actor_update_count = 0
    other._load_hybrid_checkpoint(tmp_path)
    assert other._actor_update_count == 4
    assert other.entropy_controller.state_dict() == obj.entropy_controller.state_dict()
    assert other.semantic_scorer.scorer.ready and other.semantic_scorer.scorer.version == 7
    assert len(other.semantic_scorer.scorer._pending) == 1
    assert other.semantic_scorer.calls[-1] == ("load", {"resume": True})


def test_resume_rejects_missing_state_and_implicit_method_change(runtime, tmp_path):
    obj = driver(runtime)
    obj._save_hybrid_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="dataloader"):
        driver(runtime)._load_hybrid_checkpoint(tmp_path)
    (tmp_path / "data.pt").write_bytes(b"state")
    other = driver(runtime)
    other.semantic_control_config.ramp_steps = 2
    with pytest.raises(ValueError, match="method/config"):
        other._load_hybrid_checkpoint(tmp_path)
    disabled = driver(runtime, semantic=False, control=False)
    with pytest.raises(ValueError, match="all hybrid modules disabled"):
        disabled._load_hybrid_checkpoint(tmp_path)
    disabled.config.trainer.val_only = True
    disabled._load_hybrid_checkpoint(tmp_path)


def test_ray_trainer_calls_semantic_update_after_entire_actor_loop():
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    fit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fit")
    calls = {name: [node for node in ast.walk(fit) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == name]
             for name in ("_prepare_hybrid_rollout", "_apply_hybrid_after_advantage", "update_actor", "_update_semantic_after_actors")}
    assert all(len(value) == 1 for value in calls.values())
    lines = [calls[name][0].lineno for name in calls]
    assert lines == sorted(lines)
    actor_loop = next(node for node in ast.walk(fit) if isinstance(node, ast.For)
                      and any(call is calls["update_actor"][0] for call in ast.walk(node))
                      and not any(call is calls["_update_semantic_after_actors"][0] for call in ast.walk(node)))
    assert actor_loop.end_lineno < calls["_update_semantic_after_actors"][0].lineno
