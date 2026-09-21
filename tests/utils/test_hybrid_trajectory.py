"""Complete trajectory export, canonical pretraining round-trip and no sampling."""
import ast
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


EXPORT = module("test_hybrid_trajectory_export", "verl/utils/hybrid_trajectory.py")
PRETRAIN = module("test_hybrid_trajectory_pretrain", "examples/drmas_trainer/pretrain_semantic_value.py")


def opaque(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def batch():
    rows = []
    for trajectory, count in (("a", 2), ("b", 5)):
        for index in range(count):
            role = "Solver Agent" if index % 2 == 0 else "Verifier Agent"
            terminal = index == count - 1
            rows.append({
                "schema_version": "drmas.math_event.v1", "task_type": "math", "uid": "task-group",
                "task_uid": "task-hash", "traj_uid": trajectory, "event_uid": f"{trajectory}:{index}",
                "event_index": index, "event_count": count, "role_event_index": index // 2,
                "role_event_count": (count + 1) // 2 if index % 2 == 0 else count // 2,
                "agent_id": role, "wg_id": role, "model_id": "model", "env_step_index": 0,
                "sampling_try": 0, "hcapo_state_chat": [{"role": "user", "content": "完整 actor 观察 " * 300}],
                "raw_action_text": "完整 raw response " * 600,
                "executed_action_text": ("same executed solver" if index % 2 == 0 else
                                         "<verify>approve</verify>" if trajectory == "a" else "<verify>reject</verify>"),
                "is_action_valid": True, "prompt_was_truncated": True,
                "untruncated_prompt_token_count": 3000, "ended_with_eos": True,
                "terminal_action_text": "same executed solver", "episode_rewards": float(trajectory == "a"),
                "pass": trajectory == "a", "loop_index": index // 2,
                "verifier_decision": ("approve" if trajectory == "a" else "reject") if index % 2 else "not_applicable",
                "approved_before": False, "approved_after": terminal and trajectory == "a",
                "orchestration_stop_reason": "verifier_approved" if trajectory == "a" else "max_loop_exhausted",
                "event_type": "math_solution" if index % 2 == 0 else "math_verifier",
                "env_done": terminal, "env_reward": float(trajectory == "a") if terminal else 0.,
                "is_env_action": terminal, "env_action_owner": role if terminal else None,
                "original_question": "完整原题，与截断 actor prompt 不同", "max_solver_turns": 3,
                "generation_finish_reason": "stop", "value_action_truncated": False,
                "value_credit_D": .03, "entropy_credit_final_multiplier": 1.1,
                "entropy_control_brake": .6, "top16_entropy_mean": float("nan"),
            })
    size = len(rows)
    metadata = {key: opaque([row[key] for row in rows]) for key in rows[0]}
    scalar = torch.arange(1, size + 1, dtype=torch.float32)
    base_advantages = torch.stack((scalar, scalar, torch.zeros_like(scalar)), -1)
    tensors = {
        "prompts": torch.tensor([[0, 10, 20]] * size),
        "responses": torch.tensor([[30, 2, 0]] * size),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0]] * size),
        "response_mask": torch.tensor([[1, 1, 0]] * size),
        "rollout_log_probs": torch.tensor([[-.1, -.2, 151643.]] * size),
        "hybrid_base_advantages": base_advantages, "advantages": base_advantages * 1.1,
        "returns": base_advantages * .2, "event_values": scalar / 10,
        "token_level_scores": base_advantages * .3, "sample_weight": torch.ones(size),
        "entropy_control_weight": torch.full((size,), .6),
        "entropy_control_cap": torch.ones(size), "entropy_control_valid": torch.ones(size, dtype=torch.bool),
    }
    return SimpleNamespace(batch=tensors, non_tensor_batch=metadata)


def take(data, indices):
    return SimpleNamespace(batch={key: value[indices].clone() for key, value in data.batch.items()},
                           non_tensor_batch={key: value[indices].copy() for key, value in data.non_tensor_batch.items()})


def records(data, split="train"):
    return list(EXPORT.iter_hybrid_trajectories(data, split=split, global_step=10, run_id="run", actor_update_count=9))


def test_all_real_actions_saved_in_global_order_with_full_text_and_exact_token_ids():
    original = batch()
    duplicated = take(original, [6, 0, 4, 1, 2, 5, 3, 2, 0])
    result = {item["traj_uid"]: item for item in records(duplicated)}
    assert set(result) == {"a", "b"}
    assert [item["event_index"] for item in result["b"]["actions"]] == [0, 1, 2, 3, 4]
    assert result["a"]["optimizer_padding_copies_removed"] == 1
    assert result["b"]["optimizer_padding_copies_removed"] == 1
    assert len(result["b"]["actions"]) == 5
    solver_actions = [item for item in result["b"]["actions"] if item["role"] == "Solver Agent"]
    assert len(solver_actions) == 3
    assert len({item["event_uid"] for item in solver_actions}) == 3
    assert {item["text"] for item in solver_actions} == {"same executed solver"}
    action = result["a"]["actions"][0]
    assert action["raw_action_text"] == original.non_tensor_batch["raw_action_text"][0]
    assert action["hcapo_state_chat"] == original.non_tensor_batch["hcapo_state_chat"][0]
    assert action["prompt_token_ids"] == [10, 20]
    assert action["response_token_ids"] == [30, 2]
    assert action["rollout_log_probs"] == pytest.approx([-.1, -.2])
    assert action["base_advantage"] == 1.
    assert action["advantage"] == pytest.approx(1.1)
    assert action["advantages"] == pytest.approx([1.1, 1.1])
    assert action["entropy_credit_final_multiplier"] == 1.1
    assert action["entropy_control_weight"] == pytest.approx(.6)
    assert action["value_credit_D"] == .03
    assert action["top16_entropy_mean"] is None
    assert result["a"]["question"] == original.non_tensor_batch["original_question"][0]
    assert result["a"]["max_solver_turns"] == 3
    assert result["a"]["label"] and result["a"]["terminated"]
    assert result["a"]["actions"][-1]["env_reward"] == 1.
    assert not result["b"]["label"] and result["b"]["terminated"]


def test_validation_exports_every_action_without_training_or_semantic_requirements():
    data = batch()
    for key in list(data.batch):
        if key not in {"prompts", "responses", "attention_mask", "rollout_log_probs"}:
            data.batch.pop(key)
    for key in list(data.non_tensor_batch):
        if key.startswith(("value_", "entropy_", "top16_")):
            data.non_tensor_batch.pop(key)
    result = records(data, "val")
    assert len(result) == 2
    assert sum(item["event_count"] for item in result) == 7
    assert all(item["question"] for item in result)
    assert all("advantages" not in action for item in result for action in item["actions"])


def test_multiple_validation_batches_same_step_do_not_overwrite_and_pretrain_round_trip(tmp_path):
    data = batch()
    kwargs = dict(output_dir=str(tmp_path), split="val", global_step=0, run_id="run", actor_update_count=0)
    first, trajectories, events, part = EXPORT.dump_hybrid_trajectories(data, **kwargs)
    second, _, _, next_part = EXPORT.dump_hybrid_trajectories(data, **kwargs)
    assert first != second and part == 0 and next_part == 1
    assert trajectories == 2 and events == 7
    exported = [json.loads(line) for line in Path(first).read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 2
    assert all(item["schema_version"] == EXPORT.SCHEMA_VERSION for item in exported)
    loaded, provenance = PRETRAIN.load_records([first])
    assert len(loaded) == 2
    assert sorted(len(item["actions"]) for item in loaded) == [2, 5]
    assert provenance["trajectories"] == 2
    assert all(item["question"] == data.non_tensor_batch["original_question"][0] for item in loaded)


def test_driver_saves_every_train_step_and_every_validation_batch(tmp_path):
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    hook = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                and node.name == "_maybe_dump_hybrid_trajectories")
    namespace = {"DataProto": object, "dump_hybrid_trajectories": EXPORT.dump_hybrid_trajectories}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[hook], type_ignores=[])), "ray_trainer.py", "exec"), namespace)
    driver = SimpleNamespace(hybrid_enabled=True, global_steps=9, _actor_update_count=8, _event_trace_run_id="run",
                             _hybrid_trajectory_part_counters=defaultdict(int),
                             config=SimpleNamespace(trainer={"rollout_data_dir": str(tmp_path / "train"),
                                                             "validation_data_dir": str(tmp_path / "val"),
                                                             "save_freq": 10, "log_val_generations": 0}))
    save = namespace["_maybe_dump_hybrid_trajectories"]
    for step in (9, 10):
        driver.global_steps = step
        assert save(driver, batch(), "train") == {"trajectory_dump/train/trajectories": 2., "trajectory_dump/train/events": 7.}
    for _ in range(2):
        assert save(driver, batch(), "val") == {"trajectory_dump/val/trajectories": 2., "trajectory_dump/val/events": 7.}
    assert len(list((tmp_path / "train").glob("*.jsonl"))) == 2
    assert len(list((tmp_path / "val").glob("*.jsonl"))) == 2


@pytest.mark.parametrize("indices", [[0, 1, 2, 3, 5, 6], [0, 1, 2, 3, 4, 5]])
def test_incomplete_real_trajectory_is_rejected_instead_of_silently_exported(indices):
    with pytest.raises(ValueError, match="missing|incomplete"):
        records(take(batch(), indices))


def test_conflicting_padding_copy_is_rejected():
    data = take(batch(), [0, 1, 2, 3, 4, 5, 6, 0])
    data.batch["advantages"][-1, 0] = 999.
    with pytest.raises(ValueError, match="conflicting advantages"):
        records(data)


def test_zero_advantage_copy_still_requires_matching_frozen_credit_metadata():
    data = take(batch(), [0, 1, 2, 3, 4, 5, 6, 0])
    data.batch["advantages"].zero_()
    data.non_tensor_batch["entropy_credit_final_multiplier"][-1] = .8
    with pytest.raises(ValueError, match="conflicting entropy_credit_final_multiplier"):
        records(data)


def test_missing_original_question_is_not_replaced_by_truncated_prompt():
    data = batch()
    data.non_tensor_batch.pop("original_question")
    with pytest.raises(ValueError, match="complete original question"):
        records(data)
