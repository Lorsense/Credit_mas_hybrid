"""Offline integrity and coverage audit for Dr.MAS Search event traces."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from agent_system.event_trace import SEARCH_EVENT_TRACE_SCHEMA_VERSION


_SEARCH_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "task_type",
        "run_id",
        "split",
        "global_step",
        "actor_update_count",
        "uid",
        "traj_uid",
        "event_uid",
        "event_index",
        "event_count",
        "role",
        "agent_id",
        "role_event_index",
        "role_event_count",
        "wg_id",
        "model_id",
        "policy_snapshot_id",
        "env_step_index",
        "sampling_try",
        "task_uid",
        "data_source",
        "hcapo_state_chat",
        "raw_action_text",
        "executed_action_text",
        "is_action_valid",
        "prompt_was_truncated",
        "untruncated_prompt_token_count",
        "generation_finish_reason",
        "ended_with_eos",
        "terminal_action_text",
        "terminal_reward",
        "terminal_success",
        "episode_length",
        "tool_callings",
        "orchestration_stop_reason",
        "event_type",
        "verifier_decision",
        "route_target",
        "is_env_action",
        "env_action_text",
        "env_action_owner",
        "env_observation_text",
        "env_reward",
        "env_done",
        "tool_called",
        "tool_name",
        "tool_query",
        "tool_query_valid",
        "tool_observation",
        "tool_status",
        "tool_result_count",
        "tool_error",
        "final_answer_text",
        "protocol_version",
        "protocol_turn_index",
        "remaining_search_slots",
        "query_history_count",
        "retrieval_attempt_count",
        "successful_retrieval_count",
        "evidence_ledger_count",
        "duplicate_query_count",
        "last_tool_status",
        "ledger_snapshot_sha256",
        "protocol_ledger_visible",
        "visible_query_ids",
        "visible_query_hashes",
        "visible_evidence_ids",
        "visible_evidence_sha256s",
        "route_reason",
        "route_forced",
        "prompt_token_ids",
        "response_token_ids",
        "response_mask",
        "rollout_log_probs",
        "prompt_token_count",
        "response_token_count",
    }
)

_SEARCH_NONEMPTY_STRING_FIELDS = (
    "run_id",
    "uid",
    "traj_uid",
    "event_uid",
    "role",
    "agent_id",
    "wg_id",
    "model_id",
    "policy_snapshot_id",
    "task_uid",
    "data_source",
    "generation_finish_reason",
    "protocol_version",
    "ledger_snapshot_sha256",
    "route_reason",
)


def _record_shape_errors(record: Mapping[str, Any], record_index: int) -> list[str]:
    prefix = f"record[{record_index}]"
    errors: list[str] = []
    missing = sorted(_SEARCH_REQUIRED_FIELDS.difference(record.keys()))
    if missing:
        errors.append(f"{prefix}: missing required fields {missing}")

    if record.get("schema_version") != SEARCH_EVENT_TRACE_SCHEMA_VERSION:
        errors.append(
            f"{prefix}: schema_version={record.get('schema_version')!r}, "
            f"expected {SEARCH_EVENT_TRACE_SCHEMA_VERSION!r}"
        )
    if record.get("task_type") != "search":
        errors.append(f"{prefix}: task_type={record.get('task_type')!r}, expected 'search'")
    for field in _SEARCH_NONEMPTY_STRING_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}: {field} must be a non-empty string")

    if record.get("role") != record.get("agent_id"):
        errors.append(f"{prefix}: role and agent_id differ")

    state = record.get("hcapo_state_chat")
    if not isinstance(state, list) or not state:
        errors.append(f"{prefix}: hcapo_state_chat must be a non-empty message list")
    elif any(
        not isinstance(message, Mapping) or "role" not in message or "content" not in message
        for message in state
    ):
        errors.append(f"{prefix}: hcapo_state_chat contains a malformed message")

    for field in ("prompt_token_ids", "response_token_ids", "response_mask", "rollout_log_probs"):
        value = record.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{prefix}: {field} must be a non-empty list")

    prompt_ids = record.get("prompt_token_ids")
    response_ids = record.get("response_token_ids")
    response_mask = record.get("response_mask")
    if isinstance(prompt_ids, list) and prompt_ids:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in prompt_ids):
            errors.append(f"{prefix}: prompt_token_ids must contain integers")
        if record.get("prompt_token_count") != len(prompt_ids):
            errors.append(f"{prefix}: prompt_token_count does not match prompt_token_ids")
    if isinstance(response_ids, list) and response_ids:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in response_ids):
            errors.append(f"{prefix}: response_token_ids must contain integers")
        if record.get("response_token_count") != len(response_ids):
            errors.append(f"{prefix}: response_token_count does not match response_token_ids")
    if isinstance(response_mask, list) and response_mask and any(value != 1 for value in response_mask):
        errors.append(f"{prefix}: response_mask must contain only retained-token markers (1)")

    for field in (
        "is_action_valid",
        "prompt_was_truncated",
        "ended_with_eos",
        "terminal_success",
        "is_env_action",
        "tool_called",
        "route_forced",
        "protocol_ledger_visible",
    ):
        if not isinstance(record.get(field), bool):
            errors.append(f"{prefix}: {field} must be a boolean")

    for field in (
        "event_index",
        "event_count",
        "role_event_index",
        "role_event_count",
        "env_step_index",
        "remaining_search_slots",
        "query_history_count",
        "retrieval_attempt_count",
        "successful_retrieval_count",
        "evidence_ledger_count",
        "duplicate_query_count",
    ):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{prefix}: {field} must be a non-negative integer")
    protocol_turn = record.get("protocol_turn_index")
    if not isinstance(protocol_turn, int) or isinstance(protocol_turn, bool) or protocol_turn < 1:
        errors.append(f"{prefix}: protocol_turn_index must be a positive integer")
    if record.get("query_history_count") != record.get("retrieval_attempt_count"):
        errors.append(f"{prefix}: query_history_count must equal retrieval_attempt_count")
    if isinstance(record.get("successful_retrieval_count"), int) and isinstance(record.get("retrieval_attempt_count"), int):
        if record["successful_retrieval_count"] > record["retrieval_attempt_count"]:
            errors.append(f"{prefix}: successful_retrieval_count exceeds retrieval_attempt_count")
    if isinstance(record.get("duplicate_query_count"), int) and isinstance(record.get("retrieval_attempt_count"), int):
        if record["duplicate_query_count"] > record["retrieval_attempt_count"]:
            errors.append(f"{prefix}: duplicate_query_count exceeds retrieval_attempt_count")
    snapshot_hash = record.get("ledger_snapshot_sha256")
    if isinstance(snapshot_hash, str) and not re.fullmatch(r"[0-9a-f]{64}", snapshot_hash):
        errors.append(f"{prefix}: ledger_snapshot_sha256 must be a lowercase SHA-256 hex digest")
    for field in ("visible_query_ids", "visible_evidence_ids", "visible_evidence_sha256s"):
        values = record.get(field)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            errors.append(f"{prefix}: {field} must be a list of non-empty strings")
    query_hashes = record.get("visible_query_hashes")
    if not isinstance(query_hashes, list) or any(
        value is not None and (not isinstance(value, str) or not value)
        for value in query_hashes
    ):
        errors.append(f"{prefix}: visible_query_hashes must be a list of SHA-256 strings or null")
    query_ids = record.get("visible_query_ids")
    if isinstance(query_ids, list) and isinstance(query_hashes, list) and len(query_ids) != len(query_hashes):
        errors.append(f"{prefix}: visible_query_ids and visible_query_hashes must be positionally aligned")
    evidence_ids = record.get("visible_evidence_ids")
    evidence_hashes = record.get("visible_evidence_sha256s")
    if (
        isinstance(evidence_ids, list)
        and isinstance(evidence_hashes, list)
        and len(evidence_ids) != len(evidence_hashes)
    ):
        errors.append(f"{prefix}: visible_evidence_ids and visible_evidence_sha256s must be positionally aligned")
    for field in ("visible_query_hashes", "visible_evidence_sha256s"):
        values = record.get(field)
        if isinstance(values, list) and any(
            value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value))
            for value in values
        ):
            errors.append(f"{prefix}: {field} must contain lowercase SHA-256 hex digests")
    if record.get("protocol_ledger_visible") is False and any(
        record.get(field)
        for field in (
            "visible_query_ids",
            "visible_query_hashes",
            "visible_evidence_ids",
            "visible_evidence_sha256s",
        )
    ):
        errors.append(f"{prefix}: protocol_ledger_visible=false requires empty visible-entry lists")
    if isinstance(record.get("event_count"), int) and record.get("event_count", 0) == 0:
        errors.append(f"{prefix}: event_count must be positive")
    if isinstance(record.get("role_event_count"), int) and record.get("role_event_count", 0) == 0:
        errors.append(f"{prefix}: role_event_count must be positive")

    for field in ("terminal_reward", "episode_length", "tool_callings"):
        value = record.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            errors.append(f"{prefix}: {field} must be a finite number")

    return errors


def load_event_trace_records(trace_dir: str | Path, split: str = "val") -> tuple[list[dict[str, Any]], list[str]]:
    trace_path = Path(trace_dir).expanduser()
    files = sorted(trace_path.glob(f"{split}_step_*_part_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No {split} event trace JSONL files found under {trace_path}")

    records: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(record)
    return records, [str(path) for path in files]


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def _state_text(record: Mapping[str, Any]) -> str:
    messages = record.get("hcapo_state_chat") or []
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, Mapping)
    )


def audit_search_event_records(
    records: Iterable[Mapping[str, Any]],
    *,
    min_effective_trajectories: int = 100,
    advantage_eps: float = 1e-8,
    expected_tasks: int | None = None,
    expected_rollouts_per_task: int | None = None,
    min_valid_search_events: int = 0,
) -> dict[str, Any]:
    input_records = list(records)
    if not input_records:
        raise ValueError("No event trace records were found")

    search_records: list[dict[str, Any]] = []
    structural_errors: list[str] = []
    for record_index, record in enumerate(input_records):
        if not isinstance(record, Mapping):
            structural_errors.append(f"record[{record_index}]: expected a JSON object")
            continue
        copied_record = dict(record)
        search_records.append(copied_record)
        structural_errors.extend(_record_shape_errors(copied_record, record_index))

    errors: list[str] = []
    warnings: list[str] = []

    if structural_errors:
        displayed_errors = structural_errors[:200]
        if len(structural_errors) > len(displayed_errors):
            displayed_errors.append(
                f"... omitted {len(structural_errors) - len(displayed_errors)} additional structural errors"
            )
        return {
            "run_ids": sorted(
                {
                    str(record.get("run_id"))
                    for record in search_records
                    if isinstance(record.get("run_id"), str) and record.get("run_id")
                }
            ),
            "integrity": {
                "passed": False,
                "error_count": len(structural_errors),
                "errors": displayed_errors,
                "warnings": warnings,
            },
            "counts": {
                "events": len(search_records),
                "tasks": 0,
                "trajectories": 0,
                "successful_trajectories": 0,
                "repeated_trajectory_role_groups": 0,
                "repeated_trajectories": 0,
                "nonzero_repeat_trajectory_role_groups": 0,
                "raw_effective_success_trajectory_role_groups": 0,
                "effective_success_trajectory_role_groups": 0,
                "effective_success_trajectories": 0,
                "excluded_truncated_trajectories": 0,
                "tool_error_events": 0,
                "search_tool_events": 0,
                "valid_search_tool_events": 0,
                "invalid_search_action_events": 0,
                "no_result_events": 0,
                "truncated_events": 0,
            },
            "coverage_gate": {
                "min_effective_trajectories": int(min_effective_trajectories),
                "ready_for_offline_hindsight": False,
            },
            "distributions": {
                "rollouts_per_task": {},
                "episode_length": {},
                "tool_callings": {},
                "tool_status": {},
                "stop_reason": {},
            },
            "data_sources": {},
        }

    run_ids = {str(record.get("run_id")) for record in search_records}
    if len(run_ids) != 1:
        errors.append(f"Expected one run_id, found {sorted(run_ids)}")
    schema_versions = {str(record.get("schema_version")) for record in search_records}
    if schema_versions != {SEARCH_EVENT_TRACE_SCHEMA_VERSION}:
        errors.append(
            f"Expected schema {SEARCH_EVENT_TRACE_SCHEMA_VERSION}, found {sorted(schema_versions)}"
        )

    event_uids = [str(record.get("event_uid")) for record in search_records]
    duplicate_event_uids = len(event_uids) - len(set(event_uids))
    if duplicate_event_uids:
        errors.append(f"Found {duplicate_event_uids} duplicate event_uid values")

    by_trajectory: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in search_records:
        by_trajectory[str(record.get("traj_uid"))].append(record)

    trajectory_summaries: dict[str, dict[str, Any]] = {}
    tool_error_events = 0
    missing_tool_queries = 0
    invalid_search_action_events = 0
    no_result_events = 0
    tool_status_counts: Counter[str] = Counter()
    truncated_events = 0
    full_action_leak_events = 0

    for traj_uid, trajectory_records in by_trajectory.items():
        events = sorted(trajectory_records, key=lambda record: int(record.get("event_index", -1)))
        event_indices = [int(event.get("event_index", -1)) for event in events]
        if event_indices != list(range(len(events))):
            errors.append(f"{traj_uid}: non-contiguous event_index {event_indices}")

        declared_counts = {int(event.get("event_count", -1)) for event in events}
        if declared_counts != {len(events)}:
            errors.append(f"{traj_uid}: event_count values {sorted(declared_counts)} != {len(events)}")

        role_counts = Counter(str(event.get("role")) for event in events)
        for role, role_events in _group_by_role(events).items():
            role_indices = [int(event.get("role_event_index", -1)) for event in role_events]
            if role_indices != list(range(len(role_events))):
                errors.append(f"{traj_uid}/{role}: non-contiguous role_event_index {role_indices}")
            declared_role_counts = {int(event.get("role_event_count", -1)) for event in role_events}
            if declared_role_counts != {len(role_events)}:
                errors.append(
                    f"{traj_uid}/{role}: role_event_count values {sorted(declared_role_counts)} != {len(role_events)}"
                )

        env_steps = [int(event.get("env_step_index", -1)) for event in events]
        if env_steps != sorted(env_steps):
            errors.append(f"{traj_uid}: env_step_index is not monotonic: {env_steps}")

        events_by_step: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            events_by_step[int(event.get("env_step_index", -1))].append(event)

        for env_step, step_events in sorted(events_by_step.items()):
            step_roles = [str(event.get("role")) for event in step_events]
            allowed = (
                ["Verifier Agent", "Search Agent"],
                ["Verifier Agent", "Answer Agent"],
            )
            if step_roles not in allowed:
                errors.append(f"{traj_uid}/step-{env_step}: illegal role sequence {step_roles}")

            decisions = {str(event.get("verifier_decision")) for event in step_events}
            route_targets = {str(event.get("route_target")) for event in step_events}
            route_reasons = {str(event.get("route_reason")) for event in step_events}
            route_forced_values = {bool(event.get("route_forced")) for event in step_events}
            action_owners = {str(event.get("env_action_owner")) for event in step_events}
            protocol_fields = (
                "protocol_version",
                "protocol_turn_index",
                "remaining_search_slots",
                "query_history_count",
                "retrieval_attempt_count",
                "successful_retrieval_count",
                "evidence_ledger_count",
                "duplicate_query_count",
                "last_tool_status",
                "ledger_snapshot_sha256",
                "protocol_ledger_visible",
            )
            for field in protocol_fields:
                if len({event.get(field) for event in step_events}) != 1:
                    errors.append(f"{traj_uid}/step-{env_step}: inconsistent {field}")
            # These are intentionally lists because an invalid query has an
            # observable query id but no normalized-query hash.  Serialize
            # them for comparison rather than placing unhashable lists in a
            # set, and require every role at the same environment state to
            # have seen the exact same bounded ledger snapshot.
            for field in (
                "visible_query_ids",
                "visible_query_hashes",
                "visible_evidence_ids",
                "visible_evidence_sha256s",
            ):
                serialized_values = {
                    json.dumps(event.get(field), ensure_ascii=False, sort_keys=True)
                    for event in step_events
                }
                if len(serialized_values) != 1:
                    errors.append(f"{traj_uid}/step-{env_step}: inconsistent {field}")
            if any(event.get("protocol_version") != "drmas.search_protocol.v2" for event in step_events):
                errors.append(f"{traj_uid}/step-{env_step}: unsupported protocol_version")
            if any(event.get("protocol_turn_index") != env_step + 1 for event in step_events):
                errors.append(
                    f"{traj_uid}/step-{env_step}: protocol_turn_index must equal env_step_index + 1"
                )
            if len(decisions) != 1 or not decisions.issubset({"yes", "no", "invalid"}):
                errors.append(f"{traj_uid}/step-{env_step}: invalid/inconsistent verifier decisions {decisions}")
            if len(route_targets) != 1 or not route_targets.issubset({"Search Agent", "Answer Agent"}):
                errors.append(f"{traj_uid}/step-{env_step}: invalid/inconsistent route targets {route_targets}")
            if len(route_reasons) != 1:
                errors.append(f"{traj_uid}/step-{env_step}: inconsistent route reasons {route_reasons}")
            if len(route_forced_values) != 1:
                errors.append(f"{traj_uid}/step-{env_step}: inconsistent route_forced values {route_forced_values}")
            if len(action_owners) != 1:
                errors.append(f"{traj_uid}/step-{env_step}: inconsistent action owners {action_owners}")

            decision = next(iter(decisions)) if len(decisions) == 1 else None
            route_target = next(iter(route_targets)) if len(route_targets) == 1 else None
            route_reason = next(iter(route_reasons)) if len(route_reasons) == 1 else None
            route_forced = next(iter(route_forced_values)) if len(route_forced_values) == 1 else None
            action_owner = next(iter(action_owners)) if len(action_owners) == 1 else None
            if decision == "yes" and (
                route_target != "Answer Agent" or route_reason != "verifier_yes" or route_forced
            ):
                errors.append(
                    f"{traj_uid}/step-{env_step}: verifier_yes must be an unforced Answer route"
                )
            if decision == "no" and route_target == "Search Agent" and (
                route_reason != "verifier_no" or route_forced
            ):
                errors.append(
                    f"{traj_uid}/step-{env_step}: ordinary verifier_no must be an unforced Search route"
                )
            if decision == "no" and route_target == "Answer Agent" and (
                route_reason != "forced_last_step" or not route_forced
            ):
                errors.append(
                    f"{traj_uid}/step-{env_step}: no->Answer requires forced_last_step metadata"
                )
            if decision == "invalid" and route_target == "Search Agent" and (
                route_reason != "invalid_verifier_fallback_search" or route_forced
            ):
                errors.append(
                    f"{traj_uid}/step-{env_step}: invalid->Search requires explicit unforced fallback metadata"
                )
            if decision == "invalid" and route_target == "Answer Agent" and (
                (route_reason == "invalid_verifier_fallback_answer" and route_forced)
                or (route_reason == "forced_last_step_after_invalid_verifier" and not route_forced)
                or route_reason not in {
                    "invalid_verifier_fallback_answer",
                    "forced_last_step_after_invalid_verifier",
                }
            ):
                errors.append(
                    f"{traj_uid}/step-{env_step}: invalid->Answer has inconsistent fallback/forced metadata"
                )
            if route_forced and any(event.get("remaining_search_slots") != 0 for event in step_events):
                errors.append(f"{traj_uid}/step-{env_step}: forced route must have zero remaining search slots")
            if route_target is not None and action_owner != route_target:
                errors.append(
                    f"{traj_uid}/step-{env_step}: route target {route_target!r} != action owner {action_owner!r}"
                )

            expected_event_types = {
                "Verifier Agent": "verifier",
                "Search Agent": "search_query",
                "Answer Agent": "final_answer",
            }
            for event in step_events:
                role = str(event.get("role"))
                if event.get("event_type") != expected_event_types.get(role):
                    errors.append(
                        f"{traj_uid}/step-{env_step}: role {role!r} has event_type={event.get('event_type')!r}"
                    )

            env_action_events = [event for event in step_events if bool(event.get("is_env_action", False))]
            transition_events = [event for event in step_events if event.get("env_action_text") is not None]
            if len(env_action_events) != 1 or len(transition_events) != 1:
                errors.append(
                    f"{traj_uid}/step-{env_step}: expected one env action/transition, found "
                    f"{len(env_action_events)}/{len(transition_events)}"
                )

            for event in transition_events:
                role = str(event.get("role"))
                tool_called = bool(event.get("tool_called", False))
                tool_status = event.get("tool_status")
                tool_query_valid = event.get("tool_query_valid")
                if bool(event.get("is_env_action", False)):
                    if event.get("env_action_owner") != role:
                        errors.append(
                            f"{traj_uid}/step-{env_step}: action owner {event.get('env_action_owner')!r} != {role!r}"
                        )
                    if event.get("executed_action_text") is not None and (
                        event.get("executed_action_text") != event.get("env_action_text")
                    ):
                        errors.append(f"{traj_uid}/step-{env_step}: executed action differs from env action")
                if role == "Search Agent" and not tool_called:
                    errors.append(f"{traj_uid}/step-{env_step}: Search Agent transition is not marked as a tool call")
                if role == "Answer Agent" and tool_called:
                    errors.append(f"{traj_uid}/step-{env_step}: Answer Agent transition is marked as a tool call")
                if role == "Search Agent":
                    tool_status_counts[str(tool_status)] += 1
                    if not isinstance(tool_query_valid, bool):
                        errors.append(
                            f"{traj_uid}/step-{env_step}: tool_query_valid must be a boolean for Search Agent"
                        )
                    elif not tool_query_valid:
                        invalid_search_action_events += 1
                    allowed_statuses = {
                        "success",
                        "no_results",
                        "invalid_query",
                        "api_error",
                        "processing_error",
                        "unknown_api_state",
                        "execution_error",
                    }
                    if tool_status not in allowed_statuses:
                        errors.append(
                            f"{traj_uid}/step-{env_step}: unsupported tool_status={tool_status!r}"
                        )
                    result_count = event.get("tool_result_count")
                    if not isinstance(result_count, int) or isinstance(result_count, bool) or result_count < 0:
                        errors.append(
                            f"{traj_uid}/step-{env_step}: tool_result_count must be a non-negative integer"
                        )
                    infrastructure_statuses = {
                        "api_error",
                        "processing_error",
                        "unknown_api_state",
                        "execution_error",
                    }
                    if tool_status in infrastructure_statuses and not event.get("tool_error"):
                        errors.append(
                            f"{traj_uid}/step-{env_step}: infrastructure status {tool_status!r} has no tool_error"
                        )
                    if tool_status == "success" and isinstance(result_count, int) and result_count == 0:
                        errors.append(f"{traj_uid}/step-{env_step}: success status has zero results")
                    if tool_status in {"no_results", "invalid_query"} and result_count != 0:
                        errors.append(
                            f"{traj_uid}/step-{env_step}: {tool_status} status has result_count={result_count!r}"
                        )
                    if tool_status == "no_results":
                        no_result_events += 1
                if tool_called and not event.get("tool_query") and tool_query_valid is not False:
                    missing_tool_queries += 1
                if tool_called and not event.get("tool_observation") and tool_status != "invalid_query":
                    errors.append(f"{traj_uid}/step-{env_step}: tool call has no observation")
                if tool_called and event.get("tool_query"):
                    query_match = re.search(
                        r"<search>(.*?)</search>",
                        str(event.get("env_action_text")),
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if query_match and query_match.group(1).strip() != str(event.get("tool_query")).strip():
                        errors.append(f"{traj_uid}/step-{env_step}: tool_query differs from executed search tag")
                if event.get("tool_error"):
                    tool_error_events += 1

        episode_lengths = {int(float(event.get("episode_length", -1))) for event in events}
        tool_callings = {int(float(event.get("tool_callings", -1))) for event in events}
        terminal_rewards = {float(event.get("terminal_reward", 0.0)) for event in events}
        terminal_successes = {bool(event.get("terminal_success", False)) for event in events}
        task_uids = {str(event.get("task_uid")) for event in events}
        group_uids = {str(event.get("uid")) for event in events}
        data_sources = {str(event.get("data_source")) for event in events}
        final_answers = {event.get("final_answer_text") for event in events}
        stop_reasons = {str(event.get("orchestration_stop_reason")) for event in events}
        for name, values in (
            ("episode_length", episode_lengths),
            ("tool_callings", tool_callings),
            ("terminal_reward", terminal_rewards),
            ("terminal_success", terminal_successes),
            ("task_uid", task_uids),
            ("uid", group_uids),
            ("data_source", data_sources),
            ("final_answer_text", final_answers),
            ("orchestration_stop_reason", stop_reasons),
        ):
            if len(values) != 1:
                errors.append(f"{traj_uid}: inconsistent {name}: {values}")
        if not all(math.isfinite(value) for value in terminal_rewards):
            errors.append(f"{traj_uid}: non-finite terminal_reward")

        stop_reason = next(iter(stop_reasons)) if len(stop_reasons) == 1 else None
        allowed_stop_reasons = {
            "answer_submitted",
            "forced_answer_submitted",
            "forced_answer_invalid",
            "answer_invalid",
            "max_steps_exhausted",
            "env_done",
        }
        if stop_reason not in allowed_stop_reasons:
            errors.append(f"{traj_uid}: unsupported orchestration_stop_reason={stop_reason!r}")
        forced_answer_events = [
            event
            for event in events
            if event.get("event_type") == "final_answer" and bool(event.get("route_forced", False))
        ]
        if forced_answer_events and stop_reason not in {
            "forced_answer_submitted",
            "forced_answer_invalid",
        }:
            errors.append(f"{traj_uid}: forced Answer route has incompatible stop_reason={stop_reason!r}")
        if not forced_answer_events and stop_reason in {
            "forced_answer_submitted",
            "forced_answer_invalid",
        }:
            errors.append(f"{traj_uid}: forced stop_reason without a forced Answer route")

        episode_length = next(iter(episode_lengths))
        tool_call_count = next(iter(tool_callings))
        if role_counts["Verifier Agent"] != episode_length:
            errors.append(
                f"{traj_uid}: Verifier count {role_counts['Verifier Agent']} != episode_length {episode_length}"
            )
        if role_counts["Search Agent"] != tool_call_count:
            errors.append(
                f"{traj_uid}: Search count {role_counts['Search Agent']} != tool_callings {tool_call_count}"
            )
        answer_events = [event for event in events if event.get("role") == "Answer Agent"]
        for answer_event in answer_events:
            # A projected valid <answer>...</answer> action terminates SearchEnv.
            # Repeated Answer calls are still legal when an earlier projection
            # was invalid and the environment therefore continued.
            if not bool(answer_event.get("env_done")) and bool(answer_event.get("is_action_valid")):
                errors.append(
                    f"{traj_uid}: a valid Answer Agent action did not terminate the environment"
                )

        final_answer = next(iter(final_answers))
        if final_answer:
            terminal_event_index = max(event_indices)
            for event in events:
                if int(event.get("event_index", -1)) >= terminal_event_index:
                    continue
                if str(final_answer) in _state_text(event):
                    full_action_leak_events += 1

        for event in events:
            truncated_events += int(bool(event.get("prompt_was_truncated", False)))
            response_ids = event.get("response_token_ids") or []
            response_mask = event.get("response_mask") or []
            rollout_log_probs = event.get("rollout_log_probs") or []
            if not (len(response_ids) == len(response_mask) == len(rollout_log_probs)):
                errors.append(f"{event.get('event_uid')}: response token/mask/log-prob lengths differ")
            if not all(math.isfinite(float(value)) for value in rollout_log_probs):
                errors.append(f"{event.get('event_uid')}: non-finite rollout log-prob")

        trajectory_summaries[traj_uid] = {
            "task_uid": next(iter(task_uids)),
            "uid": next(iter(group_uids)),
            "data_source": next(iter(data_sources)),
            "terminal_reward": next(iter(terminal_rewards)),
            "terminal_success": next(iter(terminal_successes)),
            "episode_length": episode_length,
            "tool_callings": tool_call_count,
            "stop_reason": next(iter(stop_reasons)),
            "has_truncated_prompt": any(bool(event.get("prompt_was_truncated")) for event in events),
            "role_counts": dict(role_counts),
        }

    if missing_tool_queries:
        errors.append(f"Found {missing_tool_queries} tool calls without a normalized tool_query")
    if tool_error_events:
        errors.append(f"Found {tool_error_events} tool transitions with tool_error")
    if invalid_search_action_events:
        warnings.append(
            f"Found {invalid_search_action_events} model-generated invalid/empty Search actions; "
            "these are behavior data, not retriever infrastructure failures"
        )
    if no_result_events:
        warnings.append(f"Found {no_result_events} valid queries with no retrieval results")
    search_tool_events = sum(tool_status_counts.values())
    valid_search_tool_events = search_tool_events - invalid_search_action_events
    if valid_search_tool_events < min_valid_search_events:
        errors.append(
            f"Expected at least {min_valid_search_events} valid Search Agent tool events, "
            f"found {valid_search_tool_events}"
        )
    if full_action_leak_events:
        errors.append(f"Found {full_action_leak_events} earlier states containing the full final action text")
    if any("offline_eval_info" in record for record in search_records):
        warnings.append("offline_eval_info is present; do not feed it to the hindsight scorer")
    if truncated_events:
        warnings.append(f"{truncated_events} events have prompt_was_truncated=true and should be excluded offline")

    role_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for traj_uid, events in by_trajectory.items():
        for role, role_events in _group_by_role(events).items():
            summary = trajectory_summaries[traj_uid]
            role_groups[(traj_uid, role)] = {
                **summary,
                "traj_uid": traj_uid,
                "role": role,
                "event_count": len(role_events),
                "repeated": len(role_events) >= 2,
                "macro_advantage": 0.0,
                "group_has_reward_variance": False,
            }

    by_prompt_role: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for role_group in role_groups.values():
        by_prompt_role[(role_group["uid"], role_group["role"])].append(role_group)

    for prompt_role_groups in by_prompt_role.values():
        rewards = np.array([group["terminal_reward"] for group in prompt_role_groups], dtype=np.float64)
        mean = float(rewards.mean())
        std = float(rewards.std())
        has_variance = std > advantage_eps
        for group, reward in zip(prompt_role_groups, rewards):
            group["group_has_reward_variance"] = has_variance
            group["macro_advantage"] = float((reward - mean) / (std + 1e-6)) if has_variance else 0.0

    repeated_role_groups = [group for group in role_groups.values() if group["repeated"]]
    nonzero_repeat_groups = [
        group for group in repeated_role_groups if group["group_has_reward_variance"]
    ]
    raw_effective_success_groups = [
        group
        for group in nonzero_repeat_groups
        if group["terminal_success"] and abs(group["macro_advantage"]) > advantage_eps
    ]
    effective_success_groups = [
        group for group in raw_effective_success_groups if not group["has_truncated_prompt"]
    ]
    repeated_trajectories = {group["traj_uid"] for group in repeated_role_groups}
    raw_effective_trajectories = {group["traj_uid"] for group in raw_effective_success_groups}
    effective_trajectories = {group["traj_uid"] for group in effective_success_groups}
    excluded_truncated_trajectories = raw_effective_trajectories.difference(effective_trajectories)

    task_to_trajectories: defaultdict[str, set[str]] = defaultdict(set)
    for traj_uid, summary in trajectory_summaries.items():
        task_to_trajectories[summary["task_uid"]].add(traj_uid)

    if expected_tasks is not None and len(task_to_trajectories) != expected_tasks:
        errors.append(f"Expected {expected_tasks} tasks, found {len(task_to_trajectories)}")
    if expected_rollouts_per_task is not None:
        wrong_rollout_counts = {
            task_uid: len(trajectory_ids)
            for task_uid, trajectory_ids in task_to_trajectories.items()
            if len(trajectory_ids) != expected_rollouts_per_task
        }
        if wrong_rollout_counts:
            preview = dict(list(sorted(wrong_rollout_counts.items()))[:20])
            errors.append(
                f"Expected {expected_rollouts_per_task} rollouts per task; mismatches={preview}"
            )

    data_source_summary: dict[str, dict[str, int]] = {}
    for data_source in sorted({summary["data_source"] for summary in trajectory_summaries.values()}):
        source_trajectories = {
            traj_uid for traj_uid, summary in trajectory_summaries.items() if summary["data_source"] == data_source
        }
        data_source_summary[data_source] = {
            "trajectories": len(source_trajectories),
            "successful_trajectories": sum(
                int(trajectory_summaries[traj_uid]["terminal_success"]) for traj_uid in source_trajectories
            ),
            "repeated_trajectories": len(source_trajectories.intersection(repeated_trajectories)),
            "effective_success_trajectories": len(source_trajectories.intersection(effective_trajectories)),
        }

    counts = {
        "events": len(search_records),
        "tasks": len(task_to_trajectories),
        "trajectories": len(trajectory_summaries),
        "successful_trajectories": sum(
            int(summary["terminal_success"]) for summary in trajectory_summaries.values()
        ),
        "repeated_trajectory_role_groups": len(repeated_role_groups),
        "repeated_trajectories": len(repeated_trajectories),
        "nonzero_repeat_trajectory_role_groups": len(nonzero_repeat_groups),
        "raw_effective_success_trajectory_role_groups": len(raw_effective_success_groups),
        "effective_success_trajectory_role_groups": len(effective_success_groups),
        "effective_success_trajectories": len(effective_trajectories),
        "excluded_truncated_trajectories": len(excluded_truncated_trajectories),
        "tool_error_events": tool_error_events,
        "search_tool_events": search_tool_events,
        "valid_search_tool_events": valid_search_tool_events,
        "invalid_search_action_events": invalid_search_action_events,
        "no_result_events": no_result_events,
        "truncated_events": truncated_events,
    }

    return {
        "run_ids": sorted(run_ids),
        "integrity": {
            "passed": not errors,
            "error_count": len(errors),
            "errors": errors,
            "warnings": warnings,
        },
        "counts": counts,
        "coverage_gate": {
            "min_effective_trajectories": int(min_effective_trajectories),
            "ready_for_offline_hindsight": len(effective_trajectories) >= min_effective_trajectories,
        },
        "distributions": {
            "rollouts_per_task": _distribution(len(value) for value in task_to_trajectories.values()),
            "episode_length": _distribution(
                summary["episode_length"] for summary in trajectory_summaries.values()
            ),
            "tool_callings": _distribution(
                summary["tool_callings"] for summary in trajectory_summaries.values()
            ),
            "tool_status": {str(key): int(value) for key, value in sorted(tool_status_counts.items())},
            "stop_reason": _distribution(
                summary["stop_reason"] for summary in trajectory_summaries.values()
            ),
        },
        "data_sources": data_source_summary,
    }


def _group_by_role(events: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("role"))].append(event)
    return dict(grouped)


def render_search_audit_markdown(report: Mapping[str, Any]) -> str:
    integrity = report["integrity"]
    counts = report["counts"]
    gate = report["coverage_gate"]
    lines = [
        "# Dr.MAS Search event trace audit",
        "",
        f"- Integrity: **{'PASS' if integrity['passed'] else 'FAIL'}**",
        f"- Events: {counts['events']}",
        f"- Tasks: {counts['tasks']}",
        f"- Trajectories: {counts['trajectories']}",
        f"- Successful trajectories: {counts['successful_trajectories']}",
        f"- Repeated trajectories: {counts['repeated_trajectories']}",
        f"- Effective successful trajectories: {counts['effective_success_trajectories']}",
        f"- Effective trajectories excluded for truncation: {counts['excluded_truncated_trajectories']}",
        f"- Search tool events: {counts['search_tool_events']}",
        f"- Valid Search tool events: {counts['valid_search_tool_events']}",
        f"- Ready for offline hindsight: **{gate['ready_for_offline_hindsight']}** "
        f"(threshold={gate['min_effective_trajectories']})",
        "",
        "## Distributions",
        "",
        "```json",
        json.dumps(report["distributions"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Data sources",
        "",
        "```json",
        json.dumps(report["data_sources"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    if integrity["errors"]:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in integrity["errors"]])
    if integrity["warnings"]:
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in integrity["warnings"]])
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "audit_search_event_records",
    "load_event_trace_records",
    "render_search_audit_markdown",
]
