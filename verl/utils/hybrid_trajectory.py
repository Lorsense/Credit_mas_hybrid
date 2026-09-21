"""Complete, unsampled trajectory exports built on the zxj raw event schema.

One JSONL line is one real trajectory. Only optimizer padding copies with the
same event_uid are removed; identical text from different actions is retained.
Raw generation text and exact generated token IDs are never re-tokenized.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import uuid

import torch

from agent_system.event_trace import iter_event_trace_records, _to_jsonable


SCHEMA_VERSION = "hybrid.trajectory.v1"
_TENSOR_ACTION_FIELDS = {
    "advantages", "hybrid_base_advantages", "returns", "token_level_rewards",
    "token_level_scores", "old_log_probs", "ref_log_prob", "entropy_control_weight",
    "entropy_control_cap", "entropy_control_valid", "sample_weight", "optimizer_mini_batch_id",
}
_METADATA_PREFIXES = ("value_", "entropy_credit_", "entropy_control_", "pure_", "top16_")
_IDENTITY_FIELDS = ("traj_uid", "agent_id", "event_index", "role_event_index", "event_count")


def _json_value(value):
    value = _to_jsonable(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _BatchView:
    def __init__(self, batch, indices):
        self.batch = {key: values[indices] for key, values in batch.batch.items()}
        self.non_tensor_batch = {key: values[indices] for key, values in batch.non_tensor_batch.items()}
        self.size = len(indices)

    def __len__(self):
        return self.size


def _unique_event_view(batch):
    meta = batch.non_tensor_batch
    if "event_uid" not in meta:
        raise ValueError("Complete trajectory export requires event_uid metadata")
    seen, counts, indices = {}, {}, []
    compare_tensors = {"prompts", "responses", "attention_mask", "advantages", "hybrid_base_advantages",
                       "returns", "entropy_control_weight", "entropy_control_cap", "entropy_control_valid"}
    for row, uid in enumerate(meta["event_uid"]):
        uid = str(uid)
        if not uid:
            raise ValueError("Complete trajectory export requires nonempty event_uid")
        counts[uid] = counts.get(uid, 0) + 1
        if uid not in seen:
            seen[uid] = row
            indices.append(row)
            continue
        first = seen[uid]
        for key in _IDENTITY_FIELDS:
            if key not in meta or _json_value(meta[key][row]) != _json_value(meta[key][first]):
                raise ValueError(f"Copied event {uid} has conflicting {key}")
        for key, values in meta.items():
            if key.startswith(_METADATA_PREFIXES) or key in {
                "raw_action_text", "executed_action_text", "hcapo_state_chat", "original_question", "pass",
            }:
                if _json_value(values[row]) != _json_value(values[first]):
                    raise ValueError(f"Copied event {uid} has conflicting {key}")
        for key in compare_tensors.intersection(batch.batch):
            if not torch.equal(batch.batch[key][row], batch.batch[key][first]):
                raise ValueError(f"Copied event {uid} has conflicting {key}")
    return _BatchView(batch, indices), counts


def iter_hybrid_trajectories(batch, *, split, global_step, run_id, actor_update_count):
    """Yield every complete trajectory with all real actions and training facts."""
    unique, counts = _unique_event_view(batch)
    groups = OrderedDict()
    records = iter_event_trace_records(
        unique, split=split, global_step=global_step, run_id=run_id,
        actor_update_count=actor_update_count, include_token_ids=True,
        include_rollout_log_probs=True, require_rollout_log_probs=False,
        require_generation_finish_reason=False,
    )
    for row, record in enumerate(records):
        meta = unique.non_tensor_batch
        for key, values in meta.items():
            if key.startswith(_METADATA_PREFIXES) or key in {
                "original_question", "max_solver_turns", "role_turn_index", "action_full_entropy",
                "submitted_solution_event_uid", "transition_owner_event_uid", "math_event_adapter_version",
            }:
                record[key] = _json_value(values[row])
        width = unique.batch["responses"].shape[-1]
        response_mask = unique.batch["attention_mask"][row, -width:].bool()
        actor_mask = unique.batch.get("response_mask", unique.batch["attention_mask"][:, -width:])[row].bool()
        record["actor_response_mask"] = actor_mask[response_mask].detach().cpu().tolist()
        for key, values in unique.batch.items():
            if key not in _TENSOR_ACTION_FIELDS and not key.startswith("event_"):
                continue
            value = values[row]
            if value.ndim == 1 and value.shape[0] == width:
                value = value[response_mask]
            record[key] = _json_value(value)
        for source, target in (("advantages", "advantage"), ("hybrid_base_advantages", "base_advantage")):
            if source in unique.batch:
                values = unique.batch[source][row]
                record[target] = _json_value(values[actor_mask].mean()) if actor_mask.any() else None
        record["optimizer_copy_count"] = counts[str(record["event_uid"])]
        # Canonical semantic pretraining consumes these short action aliases.
        record["text"] = record["executed_action_text"]
        record["valid"] = bool(record["is_action_valid"])
        record["truncated"] = bool(record.get("value_action_truncated", False)) or str(
            record.get("generation_finish_reason", "")
        ).lower() in {"length", "max_tokens", "max_new_tokens"}
        record["role_turn_index"] = int(record.get("role_turn_index", record["role_event_index"]))
        groups.setdefault(str(record["traj_uid"]), []).append(record)

    for trajectory, actions in groups.items():
        actions.sort(key=lambda action: int(action["event_index"]))
        if [int(action["event_index"]) for action in actions] != list(range(len(actions))):
            raise ValueError(f"Trajectory {trajectory} has missing or conflicting global event indices")
        if any(int(action["event_count"]) != len(actions) for action in actions):
            raise ValueError(f"Trajectory {trajectory} is incomplete; refusing a partial export")
        first, last = actions[0], actions[-1]
        for field in ("uid", "task_uid", "terminal_reward", "terminal_success"):
            if any(action[field] != first[field] for action in actions):
                raise ValueError(f"Trajectory {trajectory} has inconsistent {field}")
        questions = [action.get("original_question") or action.get("value_question") for action in actions]
        if any(not isinstance(question, str) or not question.strip() for question in questions):
            raise ValueError(f"Trajectory {trajectory} is missing the complete original question")
        if any(question != questions[0] for question in questions):
            raise ValueError(f"Trajectory {trajectory} has conflicting original questions")
        budget = first.get("max_solver_turns", first.get("value_max_solver_turns"))
        if first["task_type"] == "math" and (not isinstance(budget, int) or budget < 1):
            raise ValueError(f"Trajectory {trajectory} is missing the real Solver budget")
        for action in actions:
            action.setdefault("value_question", questions[0])
            action.setdefault("value_max_solver_turns", budget)
            action.setdefault("value_action_index", action["event_index"])
            action.setdefault("value_action_text", action["executed_action_text"])
        yield {
            "schema_version": SCHEMA_VERSION, "run_id": str(run_id), "split": str(split),
            "global_step": int(global_step), "actor_update_count": int(actor_update_count),
            "uid": first["uid"], "task_uid": first["task_uid"], "traj_uid": trajectory,
            "question": questions[0], "max_solver_turns": budget,
            "value_question": questions[0], "value_max_solver_turns": budget,
            "label": bool(first["terminal_success"]), "terminal_reward": first["terminal_reward"],
            "terminated": bool(last.get("env_done", False)),
            "terminal_action_text": last["terminal_action_text"],
            "final_answer_text": last["final_answer_text"],
            "orchestration_stop_reason": last["orchestration_stop_reason"],
            "event_count": len(actions),
            "optimizer_padding_copies_removed": sum(action["optimizer_copy_count"] - 1 for action in actions),
            "actions": actions,
        }


def dump_hybrid_trajectories(batch, *, output_dir, split, global_step, part_index=0, run_id, actor_update_count):
    """Atomically write all trajectories; never sample and never replace a part.

    Returns ``(path, trajectory_count, event_count, used_part_index)``. Call once
    for each training rollout and each validation batch, independently of logger
    table sizes. Repeated validation at the same step creates new part files.
    """
    if split not in {"train", "val", "validation", "eval"}:
        raise ValueError("Unsupported trajectory export split")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    used_part = int(part_index)
    if used_part < 0 or int(global_step) < 0:
        raise ValueError("Trajectory part and global step must be nonnegative")
    while True:
        path = os.path.join(output_dir, f"{split}_step_{int(global_step):08d}_part_{used_part:05d}.jsonl")
        if not os.path.exists(path):
            break
        used_part += 1
    temporary = f"{path}.tmp-{uuid.uuid4().hex}"
    trajectories = events = 0
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            for trajectory in iter_hybrid_trajectories(
                batch, split=split, global_step=global_step, run_id=run_id,
                actor_update_count=actor_update_count,
            ):
                stream.write(json.dumps(_json_value(trajectory), ensure_ascii=False, allow_nan=False) + "\n")
                trajectories += 1
                events += trajectory["event_count"]
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return path, trajectories, events, used_part
