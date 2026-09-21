"""Semantic prefix records and prediction deltas; no entropy feature dependency.

The learned probabilities only gate an independent entropy loss. They never
replace zxj LOTO values, returns, advantages or two-stage credit coefficients.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np


_RECORD_FIELDS = (
    "value_question", "value_action_text", "value_action_index", "uid",
    "traj_uid", "agent_id", "role_turn_index", "is_action_valid", "pass",
)

def _integer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("action and role-turn indices must be nonnegative integers")
    value = int(value)
    if value < 0:
        raise ValueError("action and role-turn indices must be nonnegative integers")
    return value


def _binary(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
        return bool(value)
    raise ValueError("validity and terminal labels must be bool or numeric 0/1")


def _role(value: Any) -> str:
    aliases = {"solver": "solver", "solver agent": "solver",
               "verifier": "verifier", "verifier agent": "verifier"}
    role = aliases.get(str(value).strip().lower())
    if role is None:
        raise ValueError("prefix-value credit only supports Solver and Verifier roles")
    return role


def _row_count(batch: Mapping[str, Sequence[Any]]) -> int:
    for key in ("traj_uid", "uid", "agent_id"):
        if key in batch:
            return len(batch[key])
    return len(next(iter(batch.values()))) if batch else 0


def _collect_trajectories(
    batch: Mapping[str, Sequence[Any]], max_solver_turns: int | None
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Validate whole trajectories before exposing any of their prefixes."""
    size = _row_count(batch)
    metrics: dict[str, float | int] = {
        "value_credit/record_rows": size,
        "value_credit/record_trajectories": 0,
        "value_credit/skipped_trajectories": 0,
        "value_credit/invalid_trajectories": 0,
        "value_credit/padding_rows": 0,
        "value_credit/record_unique_actions": 0,
        "value_credit/missing_metadata": 0,
        "value_credit/budget_mismatch_trajectories": 0,
        "value_credit/incomplete_trajectories": 0,
        "value_credit/budget_checked_trajectories": 0,
    }
    missing = [field for field in _RECORD_FIELDS if field not in batch]
    if missing:
        metrics["value_credit/missing_metadata"] = len(missing)
        if "traj_uid" in batch:
            metrics["value_credit/skipped_trajectories"] = len({str(x) for x in batch["traj_uid"]})
        return [], metrics
    checked_fields = _RECORD_FIELDS + (("value_max_solver_turns",) if "value_max_solver_turns" in batch else ())
    if any(len(batch[field]) != size for field in checked_fields):
        raise ValueError("value trajectory metadata fields have inconsistent lengths")

    groups: dict[str, list[int]] = defaultdict(list)
    for row, trajectory in enumerate(batch["traj_uid"]):
        groups[str(trajectory)].append(row)
    records: list[dict[str, Any]] = []
    for trajectory, rows in groups.items():
        try:
            if batch["traj_uid"][rows[0]] is None or not trajectory:
                raise ValueError("trajectory id is missing")
            budget = max_solver_turns
            if "value_max_solver_turns" in batch:
                logged_budgets = {_integer(batch["value_max_solver_turns"][row]) for row in rows}
                if (len(logged_budgets) != 1 or 0 in logged_budgets
                        or (max_solver_turns is not None and logged_budgets != {max_solver_turns})):
                    metrics["value_credit/budget_mismatch_trajectories"] += 1
                    raise ValueError("logged solver budget is inconsistent with the trajectory or configuration")
                budget = logged_budgets.pop()
            question = batch["value_question"][rows[0]]
            prompt = batch["uid"][rows[0]]
            label = _binary(batch["pass"][rows[0]])
            if not isinstance(question, str) or not question.strip() or prompt is None:
                raise ValueError("question or prompt identity is missing")
            actions: dict[int, dict[str, Any]] = {}
            for row in rows:
                if (batch["value_question"][row] != question
                        or str(batch["uid"][row]) != str(prompt)
                        or _binary(batch["pass"][row]) != label):
                    raise ValueError("question, prompt identity, or label differs inside trajectory")
                index = _integer(batch["value_action_index"][row])
                action = {
                    "role": _role(batch["agent_id"][row]),
                    "text": batch["value_action_text"][row],
                    "valid": _binary(batch["is_action_valid"][row]),
                    "role_turn_index": _integer(batch["role_turn_index"][row]),
                    "truncated": _binary(batch.get("value_action_truncated", [False] * size)[row]),
                }
                if not isinstance(action["text"], str):
                    raise ValueError("action text is missing")
                if index in actions and actions[index] != action:
                    raise ValueError("conflicting copies of one logical action")
                actions[index] = action
            indices = sorted(actions)
            if indices != list(range(len(actions))):
                raise ValueError("chronological action indices must start at zero without gaps")
            ordered = [actions[index] for index in indices]
            for index, action in enumerate(ordered):
                expected_role = "solver" if index % 2 == 0 else "verifier"
                if action["role"] != expected_role or action["role_turn_index"] != index // 2:
                    raise ValueError("roles or role-turn indices do not follow S,V,S,V chronology")
            if budget is not None and sum(a["role"] == "solver" for a in ordered) > budget:
                metrics["value_credit/budget_mismatch_trajectories"] += 1
                raise ValueError("trajectory exceeds configured solver-turn budget")
            if "value_max_solver_turns" in batch:
                # Replay the actual Math controller stop rule. Format validity
                # does not drive stopping: exact approve wins, exact reject
                # continues, and any unrecognized Verifier output stops.
                metrics["value_credit/budget_checked_trajectories"] += 1
                for index, action in enumerate(ordered):
                    if action["role"] == "solver":
                        terminal = action["role_turn_index"] + 1 == budget
                    else:
                        text = action["text"]
                        terminal = ("<verify>approve</verify>" in text
                                    or "<verify>reject</verify>" not in text)
                    if terminal != (index == len(ordered) - 1):
                        metrics["value_credit/incomplete_trajectories"] += 1
                        raise ValueError("history is truncated or contains actions after controller termination")
            records.append({
                "traj_uid": trajectory,
                "question": question,
                "question_key": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "max_solver_turns": budget,
                "actions": ordered,
                "label": float(label),
            })
            metrics["value_credit/padding_rows"] += len(rows) - len(ordered)
            metrics["value_credit/record_unique_actions"] += len(ordered)
            metrics["value_credit/invalid_trajectories"] += int(any(not a["valid"] for a in ordered))
        except (TypeError, ValueError, OverflowError):
            # Partial, ambiguous, or malformed histories must never masquerade
            # as a valid prefix chain. Other trajectories remain usable.
            metrics["value_credit/skipped_trajectories"] += 1
    metrics["value_credit/record_trajectories"] = len(records)
    return records, metrics


def build_trajectory_records(
    non_tensor_batch: Mapping[str, Sequence[Any]], max_solver_turns: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Reconstruct chronological scorer records; never deduplicate by text.

    Padding is removed by (traj_uid, global action index), with identical
    metadata required for copies. Validity-false actions remain in records so
    the scorer can learn from them; only that action is ineligible for control.
    Structurally incomplete or conflicting
    trajectories are skipped and counted rather than partially reconstructed.
    New logs carrying value_max_solver_turns must match the explicit budget
    and must end exactly where the Math controller would stop. Legacy logs
    without that field retain structural-only validation for compatibility.
    """
    turns = None if max_solver_turns is None else _integer(max_solver_turns)
    if turns == 0:
        raise ValueError("max_solver_turns must be positive")
    return _collect_trajectories(non_tensor_batch, turns)


def attach_value_predictions(non_tensor_batch, values, ready):
    """Record all available semantic predictions, control nonterminal actions only.

    Terminal probability remains a prediction, never zxj's terminal zero value.
    Eligibility depends on executed action identity, not entropy observations.
    """
    size = _row_count(non_tensor_batch)
    before = np.full(size, np.nan)
    after = np.full(size, np.nan)
    available, eligible = np.zeros(size, bool), np.zeros(size, bool)
    records, metrics = _collect_trajectories(non_tensor_batch, None)
    record_map = {record["traj_uid"]: record for record in records}
    for row, trajectory in enumerate(non_tensor_batch.get("traj_uid", [])):
        record = record_map.get(str(trajectory))
        if record is None:
            continue
        index = int(non_tensor_batch["value_action_index"][row])
        action = record["actions"][index]
        action_valid = action["valid"] and not action.get("truncated", False)
        eligible[row] = action_valid and index < len(record["actions"]) - 1
        prediction = values.get(str(trajectory))
        if prediction is None or index + 1 >= len(prediction):
            continue
        try:
            pair = [float(prediction[i]["sem"]) for i in (index, index + 1)]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in pair):
                continue
            before[row], after[row] = pair
            available[row] = bool(ready) and action_valid
        except (KeyError, TypeError, ValueError):
            continue
    delta = np.where(np.isfinite(before) & np.isfinite(after), after - before, 0.0)
    non_tensor_batch.update(value_sem_before=before, value_sem_after=after,
                            value_credit_before=before.copy(), value_credit_after=after.copy(),
                            value_credit_delta=delta, value_credit_available=available,
                            value_control_eligible=eligible)
    metrics.update({"value_credit/scorer_ready": int(bool(ready)),
                    "value_credit/prediction_available_rows": int(available.sum()),
                    "value_credit/prediction_coverage": float(available.mean()) if size else 0.0,
                    "value_credit/nonterminal_control_rows": int((available & eligible).sum())})
    return metrics


