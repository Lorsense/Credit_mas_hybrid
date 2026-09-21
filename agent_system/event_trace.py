# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior-neutral event tracing for offline multi-agent credit audits.

The trace is deliberately built from the raw rollout batch, before balancing or
``adjust_batch`` can duplicate and reorder events.  It is a data collection
facility only: enabling it must not change rewards, advantages, or actor inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch


MATH_EVENT_TRACE_SCHEMA_VERSION = "drmas.math_event.v1"
SEARCH_EVENT_TRACE_SCHEMA_VERSION = "drmas.search_event.v3"
# Kept as the Math default for callers written against the first trace release.
EVENT_TRACE_SCHEMA_VERSION = MATH_EVENT_TRACE_SCHEMA_VERSION
EVENT_TRACE_MANIFEST_SCHEMA_VERSION = "drmas.agent_event_manifest.v2"


def _mapping_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def get_event_trace_config(config: Any) -> Any:
    """Return ``trainer.event_trace`` for dict and OmegaConf configurations."""

    trainer_config = _mapping_get(config, "trainer", None)
    return _mapping_get(trainer_config, "event_trace", {})


def event_trace_enabled(config: Any) -> bool:
    """Whether event tracing is explicitly enabled."""

    return bool(_mapping_get(get_event_trace_config(config), "enabled", False))


def event_metadata_enabled(config: Any) -> bool:
    """Whether rollout events must carry the online credit-assignment schema.

    Raw trace dumping remains independently controlled by
    :func:`event_trace_enabled`.  ``team_event_gae`` consumes the same immutable
    pre-action state and transition ownership fields in memory, so it enables
    metadata capture without forcing JSONL output.
    """

    algorithm_config = _mapping_get(config, "algorithm", None)
    estimator = _mapping_get(algorithm_config, "adv_estimator", "")
    estimator_value = getattr(estimator, "value", estimator)
    return event_trace_enabled(config) or str(estimator_value).lower() == "team_event_gae"


def event_trace_option(config: Any, key: str, default: Any = None) -> Any:
    """Read one event-trace option without assuming a config implementation."""

    return _mapping_get(get_event_trace_config(config), key, default)


def parse_search_verifier_decisions(
    text_responses: list[str],
    active_mask: np.ndarray,
    action_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Parse one unambiguous, projection-valid Search router decision.

    The route parser and the actor's invalid-action mask must agree.  A raw
    string that happens to contain a tag but failed projection is still an
    ``invalid`` router output and goes through the configured fallback.
    """

    active = np.asarray(active_mask, dtype=bool)
    if len(text_responses) != len(active):
        raise ValueError(
            f"text_responses length {len(text_responses)} != active_mask length {len(active)}"
        )
    if action_valid_mask is None:
        action_valid = np.ones(len(active), dtype=bool)
    else:
        action_valid = np.asarray(action_valid_mask, dtype=bool)
        if action_valid.shape != active.shape:
            raise ValueError(
                "action_valid_mask must have the same shape as active_mask: "
                f"{action_valid.shape} != {active.shape}"
            )

    decisions = []
    for response, is_active, is_valid in zip(text_responses, active, action_valid):
        normalized = str(response).lower()
        if not is_active:
            decisions.append("inactive")
        elif not is_valid:
            decisions.append("invalid")
        elif "<verify>yes</verify>" in normalized and "<verify>no</verify>" not in normalized:
            decisions.append("yes")
        elif "<verify>no</verify>" in normalized and "<verify>yes</verify>" not in normalized:
            decisions.append("no")
        else:
            decisions.append("invalid")
    return np.array(decisions, dtype=object)


def build_search_route_targets(
    active_mask: np.ndarray,
    verification_vector: np.ndarray,
    *,
    step: int,
    max_steps: int,
) -> np.ndarray:
    """Legacy boolean route helper used by older call sites and unit tests.

    New Search execution uses :func:`agent_system.search_protocol.plan_search_routes`
    because it preserves invalid Verifier syntax.  This helper still enforces
    the v2 finite-horizon rule: a final ``False`` routes to Answer, never to
    an empty orchestra fallback.
    """

    targets = []
    for is_active, is_sufficient in zip(active_mask, verification_vector):
        if not is_active:
            targets.append("inactive")
        elif is_sufficient:
            targets.append("Answer Agent")
        elif step < max_steps:
            targets.append("Search Agent")
        else:
            targets.append("Answer Agent")
    return np.array(targets, dtype=object)


def resolve_search_env_action_owners(
    agent_selections: list[tuple[str, Any]],
    active_mask: np.ndarray,
    route_targets: Any,
) -> np.ndarray:
    """Resolve exactly one Search/Answer action owner for the v2 protocol."""

    batch_size = len(active_mask)
    if len(route_targets) != batch_size:
        raise ValueError(f"route_targets length {len(route_targets)} != batch size {batch_size}")
    owners = np.empty(batch_size, dtype=object)
    owners[:] = None
    owners[np.logical_not(active_mask)] = "inactive"

    for agent_id, selected_mask in agent_selections:
        if len(selected_mask) != batch_size:
            raise ValueError(f"{agent_id} selection length {len(selected_mask)} != batch size {batch_size}")
        for item, is_selected in enumerate(selected_mask):
            if not bool(is_selected):
                continue
            if owners[item] is not None:
                raise ValueError(f"Multiple Search env actions selected for row {item}")
            owners[item] = agent_id

    for item, is_active in enumerate(active_mask):
        if not is_active:
            if owners[item] != "inactive":
                raise ValueError(f"Inactive Search row {item} unexpectedly selected owner {owners[item]!r}")
            if route_targets[item] != "inactive":
                raise ValueError(
                    f"Inactive Search row {item} has route_target={route_targets[item]!r}, expected 'inactive'"
                )
            continue
        if owners[item] is not None:
            if route_targets[item] != owners[item]:
                raise ValueError(
                    f"Search route/action mismatch for row {item}: "
                    f"route_target={route_targets[item]!r}, owner={owners[item]!r}"
                )
            continue
        raise ValueError(
            f"No Search env action owner for active row {item}; route_target={route_targets[item]!r}"
        )
    return owners


def build_search_transition_fields(
    event: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    action_owner: str,
    text_action: str,
    reward: Any,
    done: Any,
) -> dict[str, Any]:
    """Bind one environment transition to its selected actor event.

    Search trace v2 has exactly one actor-owned environment action per active
    step.  There is deliberately no synthetic orchestra-fallback transition.
    """

    carries_transition = bool(event.get("is_env_action", False))
    # Only Search Agent env actions are retrieval tool calls. An Answer Agent
    # projection is a terminal-answer env action, not a tool call, even though
    # the underlying search env routes it through the search-action parser and
    # therefore reports tool_calling=True / tool_status for it.
    is_search_tool = carries_transition and action_owner == "Search Agent"
    tool_query = info.get("tool_query", info.get("tool_input")) if is_search_tool else None
    if isinstance(tool_query, list) and len(tool_query) == 1:
        tool_query = tool_query[0]
    return {
        "env_action_owner": action_owner,
        "env_action_text": text_action if carries_transition else None,
        "env_observation_text": info.get("tool_observation_text") if carries_transition else None,
        "env_reward": reward if carries_transition else None,
        "env_done": bool(done) if carries_transition else None,
        "tool_called": bool(is_search_tool),
        "tool_name": info.get("tool_name") if is_search_tool else None,
        "tool_query": tool_query,
        "tool_query_valid": info.get("tool_query_valid") if is_search_tool else None,
        "tool_observation": info.get("tool_observation_text") if is_search_tool else None,
        "tool_status": info.get("tool_status") if is_search_tool else None,
        "tool_result_count": info.get("tool_result_count") if is_search_tool else None,
        "tool_error": info.get("tool_error") if is_search_tool else None,
    }


def build_task_uid(env_kwargs: Mapping[str, Any]) -> str:
    """Build a stable, answer-free identifier for a Math or Search task."""

    if not isinstance(env_kwargs, Mapping) or "question" not in env_kwargs:
        raise ValueError("Event tracing requires env_kwargs with a question field")
    identity = {
        "data_source": env_kwargs.get("data_source", "unknown"),
        "question": env_kwargs["question"],
    }
    canonical_identity = json.dumps(_to_jsonable(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()


def build_offline_eval_info(env_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist private task fields; never copy arbitrary env kwargs."""

    allowed_keys = ("question", "data_source", "ground_truth", "task_id")
    return {key: _to_jsonable(env_kwargs[key]) for key in allowed_keys if key in env_kwargs}


# Backwards-compatible names for the Math trace tooling shipped in v1.  New
# call sites should use the task-generic functions above.
build_math_task_uid = build_task_uid
build_math_offline_eval_info = build_offline_eval_info


def build_event_trace_manifest(
    config: Any,
    tokenizers: Mapping[str, Any],
    wg_to_agents_mapping: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Build the minimum provenance needed to reproduce offline scoring."""

    data_config = _mapping_get(config, "data", {})
    agent_config = _mapping_get(config, "agent", {})
    env_config = _mapping_get(config, "env", {})
    rollout_config = _mapping_get(_mapping_get(config, "actor_rollout_ref", {}), "rollout", {})
    validation_config = _mapping_get(rollout_config, "val_kwargs", {})
    orchestra_type = str(_mapping_get(agent_config, "orchestra_type", "unknown"))
    orchestra_configs = _mapping_get(agent_config, "orchestra", {})
    math_orchestra_config = _mapping_get(orchestra_configs, "math", {})
    search_config = _mapping_get(env_config, "search", {})
    search_protocol_config = _mapping_get(env_config, "search_protocol", {})

    tokenizer_manifest = {}
    for wg_id, tokenizer in tokenizers.items():
        chat_template = getattr(tokenizer, "chat_template", None)
        tokenizer_manifest[str(wg_id)] = {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "chat_template_sha256": (
                hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest() if chat_template is not None else None
            ),
        }

    wg_models = {}
    for wg_id, agents in wg_to_agents_mapping.items():
        wg_models[str(wg_id)] = [
            {
                "agent_id": _mapping_get(agent, "agent_id", None),
                "model_id": _mapping_get(agent, "model_id", None),
            }
            for agent in agents
        ]

    event_schema_version = (
        SEARCH_EVENT_TRACE_SCHEMA_VERSION if orchestra_type == "search" else MATH_EVENT_TRACE_SCHEMA_VERSION
    )

    return {
        "schema_version": EVENT_TRACE_MANIFEST_SCHEMA_VERSION,
        "event_schema_version": event_schema_version,
        "run_id": str(run_id),
        "source_revision": event_trace_option(config, "source_revision", None),
        "retriever_revision": event_trace_option(config, "retriever_revision", None),
        "dataset_sha256": event_trace_option(config, "dataset_sha256", None),
        "checkpoint": {
            "resume_mode": _mapping_get(_mapping_get(config, "trainer", {}), "resume_mode", None),
            "resume_from_path": _mapping_get(_mapping_get(config, "trainer", {}), "resume_from_path", None),
        },
        "agent_ids": list(_mapping_get(agent_config, "agent_ids", [])),
        "model_ids": list(_mapping_get(agent_config, "model_ids", [])),
        "model_sharing": bool(_mapping_get(agent_config, "model_sharing", False)),
        "orchestra_type": orchestra_type,
        "wg_models": wg_models,
        "tokenizers": tokenizer_manifest,
        "math_max_loop_num": int(_mapping_get(math_orchestra_config, "max_loop_num", 0)),
        "env_max_steps": int(_mapping_get(env_config, "max_steps", 0)),
        "env_seed": int(_mapping_get(env_config, "seed", 0)),
        "search": {
            "search_url": _to_jsonable(_mapping_get(search_config, "search_url", None)),
            "topk": _mapping_get(search_config, "topk", None),
            "timeout": _mapping_get(search_config, "timeout", None),
            "history_length": _mapping_get(env_config, "history_length", None),
            "protocol": _to_jsonable(search_protocol_config),
        },
        "rollout_group_size": int(_mapping_get(_mapping_get(env_config, "rollout", {}), "n", 1)),
        "validation_group_size": int(_mapping_get(_mapping_get(env_config, "rollout", {}), "val_n", 1)),
        "max_prompt_length": int(_mapping_get(data_config, "max_prompt_length", 0)),
        "max_response_length": int(_mapping_get(data_config, "max_response_length", 0)),
        "truncation": _mapping_get(data_config, "truncation", None),
        "apply_chat_template_kwargs": _to_jsonable(_mapping_get(data_config, "apply_chat_template_kwargs", {})),
        "rollout_backend": _mapping_get(rollout_config, "name", None),
        "rollout_dtype": _mapping_get(rollout_config, "dtype", None),
        "train_sampling": {
            "temperature": _mapping_get(rollout_config, "temperature", None),
            "top_p": _mapping_get(rollout_config, "top_p", None),
            "top_k": _mapping_get(rollout_config, "top_k", None),
            "do_sample": _mapping_get(rollout_config, "do_sample", None),
        },
        "validation_sampling": {
            "temperature": _mapping_get(validation_config, "temperature", None),
            "top_p": _mapping_get(validation_config, "top_p", None),
            "top_k": _mapping_get(validation_config, "top_k", None),
            "do_sample": _mapping_get(validation_config, "do_sample", None),
        },
    }


def dump_event_trace_manifest(manifest: Mapping[str, Any], output_dir: str) -> str:
    """Atomically write one immutable manifest per trace run."""

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    run_id = str(manifest["run_id"])
    output_path = os.path.join(output_dir, f"manifest_{run_id}.json")
    if os.path.exists(output_path):
        return output_path

    temporary_path = f"{output_path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "x", encoding="utf-8") as file:
            json.dump(_to_jsonable(manifest), file, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return output_path


def annotate_trajectory_events(
    events: list[dict[str, Any]],
    traj_uid: str,
    *,
    schema_version: str = EVENT_TRACE_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Attach stable zero-based event and role-event identities in buffer order.

    ``events`` must already contain only real (active) agent calls.  The function
    mutates metadata dictionaries but never reorders, copies, or removes an event.
    """

    role_counts = Counter(str(event["agent_id"]) for event in events)
    next_role_index: defaultdict[str, int] = defaultdict(int)
    event_count = len(events)

    for event_index, event in enumerate(events):
        event_traj_uid = str(event.get("traj_uid", traj_uid))
        if event_traj_uid != str(traj_uid):
            raise ValueError(f"Event trajectory mismatch: expected {traj_uid}, got {event_traj_uid}")

        role = str(event["agent_id"])
        role_event_index = next_role_index[role]
        next_role_index[role] += 1

        event["schema_version"] = schema_version
        event["event_uid"] = f"{traj_uid}:{event_index}"
        event["event_index"] = event_index
        event["role_event_index"] = role_event_index
        event["event_count"] = event_count
        event["role_event_count"] = role_counts[role]
        event["original_row_index"] = event.get("index")

    return events


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _batch_size(batch: Any) -> int:
    try:
        return len(batch)
    except TypeError:
        pass

    for value in batch.batch.values():
        return int(value.shape[0])
    for value in batch.non_tensor_batch.values():
        return int(value.shape[0])
    return 0


def _tensor_row(batch: Any, key: str, row_index: int) -> Any:
    tensor_batch = batch.batch
    if tensor_batch is None or key not in tensor_batch.keys():
        return None
    return tensor_batch[key][row_index]


def _non_tensor_row(batch: Any, key: str, row_index: int) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return None
    return values[row_index]


def _flatten_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        value = np.asarray(value).reshape(-1).tolist()
    else:
        raise TypeError(f"Expected a tensor-like sequence, got {type(value)}")
    return [_to_jsonable(item) for item in value]


def _masked_list(value: Any, mask: Any) -> list[Any]:
    values = _flatten_list(value)
    masks = [bool(item) for item in _flatten_list(mask)]
    if len(values) != len(masks):
        raise ValueError(f"Value/mask length mismatch: {len(values)} != {len(masks)}")
    return [item for item, keep in zip(values, masks) if keep]


def _validate_trace_batch(
    batch: Any,
    include_token_ids: bool,
    require_rollout_log_probs: bool,
    require_generation_finish_reason: bool,
) -> None:
    required_non_tensor = {
        "schema_version",
        "task_type",
        "uid",
        "traj_uid",
        "event_uid",
        "event_index",
        "role_event_index",
        "event_count",
        "role_event_count",
        "agent_id",
        "wg_id",
        "model_id",
        "env_step_index",
        "sampling_try",
        "task_uid",
        "hcapo_state_chat",
        "raw_action_text",
        "executed_action_text",
        "is_action_valid",
        "prompt_was_truncated",
        "untruncated_prompt_token_count",
        "ended_with_eos",
        "terminal_action_text",
        "episode_rewards",
        "pass",
    }
    if require_generation_finish_reason:
        required_non_tensor.add("generation_finish_reason")
    missing_non_tensor = sorted(required_non_tensor.difference(batch.non_tensor_batch.keys()))
    if missing_non_tensor:
        raise ValueError(f"Event trace batch is missing non-tensor fields: {missing_non_tensor}")

    required_tensor = {"prompts", "responses", "attention_mask"} if include_token_ids else set()
    if require_rollout_log_probs:
        required_tensor.add("rollout_log_probs")
    tensor_keys = set(batch.batch.keys()) if batch.batch is not None else set()
    missing_tensor = sorted(required_tensor.difference(tensor_keys))
    if missing_tensor:
        raise ValueError(f"Event trace batch is missing tensor fields: {missing_tensor}")

    event_uids = [_to_jsonable(value) for value in batch.non_tensor_batch["event_uid"]]
    if len(event_uids) != len(set(event_uids)):
        raise ValueError("Event trace batch contains duplicate event_uid values")
    active_masks = batch.non_tensor_batch.get("active_masks")
    if active_masks is not None and not all(bool(value) for value in active_masks):
        raise ValueError("Event trace batch contains inactive filler rows")

    task_types = {str(value) for value in batch.non_tensor_batch["task_type"]}
    unknown_task_types = task_types.difference({"math", "search"})
    if unknown_task_types:
        raise ValueError(f"Unsupported event trace task types: {sorted(unknown_task_types)}")

    if "math" in task_types:
        required_math_fields = {
            "loop_index",
            "verifier_decision",
            "approved_before",
            "approved_after",
            "orchestration_stop_reason",
        }
        missing_math_fields = sorted(required_math_fields.difference(batch.non_tensor_batch.keys()))
        if missing_math_fields:
            raise ValueError(f"Math event trace batch is missing fields: {missing_math_fields}")

    if "search" in task_types:
        required_search_fields = {
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
            "orchestration_stop_reason",
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
        }
        missing_search_fields = sorted(required_search_fields.difference(batch.non_tensor_batch.keys()))
        if missing_search_fields:
            raise ValueError(f"Search event trace batch is missing fields: {missing_search_fields}")


def iter_event_trace_records(
    batch: Any,
    *,
    split: str,
    global_step: int,
    run_id: str,
    actor_update_count: int,
    include_token_ids: bool = True,
    include_rollout_log_probs: bool = True,
    require_rollout_log_probs: bool = True,
    require_generation_finish_reason: bool = True,
    include_offline_eval_info: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield compact JSON-ready records from an unadjusted rollout batch.

    Prompt left-padding and response right-padding are removed with the rollout
    attention mask.  The retained response IDs are therefore the exact generated
    teacher-forcing labels, not IDs obtained by re-tokenizing projected text.
    """

    if require_rollout_log_probs and not include_rollout_log_probs:
        raise ValueError("require_rollout_log_probs=True requires include_rollout_log_probs=True")
    if include_rollout_log_probs and not include_token_ids:
        raise ValueError("include_rollout_log_probs=True requires include_token_ids=True")

    _validate_trace_batch(
        batch,
        include_token_ids=include_token_ids,
        require_rollout_log_probs=require_rollout_log_probs,
        require_generation_finish_reason=require_generation_finish_reason,
    )

    direct_fields = {
        "schema_version": "schema_version",
        "task_type": "task_type",
        "uid": "uid",
        "traj_uid": "traj_uid",
        "event_uid": "event_uid",
        "event_index": "event_index",
        "role_event_index": "role_event_index",
        "event_count": "event_count",
        "role_event_count": "role_event_count",
        "agent_id": "agent_id",
        "wg_id": "wg_id",
        "model_id": "model_id",
        "env_step": "env_step",
        "env_step_index": "env_step_index",
        "loop_index": "loop_index",
        "sampling_try": "sampling_try",
        "task_uid": "task_uid",
        "data_source": "data_source",
        "original_row_index": "original_row_index",
        "hcapo_state_chat": "hcapo_state_chat",
        "raw_action_text": "raw_action_text",
        "executed_action_text": "executed_action_text",
        "is_action_valid": "is_action_valid",
        "prompt_was_truncated": "prompt_was_truncated",
        "untruncated_prompt_token_count": "untruncated_prompt_token_count",
        "generation_finish_reason": "generation_finish_reason",
        "ended_with_eos": "ended_with_eos",
        "verifier_decision": "verifier_decision",
        "approved_before": "approved_before",
        "approved_after": "approved_after",
        "terminal_action_text": "terminal_action_text",
        "terminal_reward": "episode_rewards",
        "terminal_success": "pass",
        "episode_length": "episode_lengths",
        "tool_callings": "tool_callings",
        "orchestration_stop_reason": "orchestration_stop_reason",
        "protocol_version": "protocol_version",
        "protocol_turn_index": "protocol_turn_index",
        "remaining_search_slots": "remaining_search_slots",
        "query_history_count": "query_history_count",
        "retrieval_attempt_count": "retrieval_attempt_count",
        "successful_retrieval_count": "successful_retrieval_count",
        "evidence_ledger_count": "evidence_ledger_count",
        "duplicate_query_count": "duplicate_query_count",
        "last_tool_status": "last_tool_status",
        "ledger_snapshot_sha256": "ledger_snapshot_sha256",
        "protocol_ledger_visible": "protocol_ledger_visible",
        "visible_query_ids": "visible_query_ids",
        "visible_query_hashes": "visible_query_hashes",
        "visible_evidence_ids": "visible_evidence_ids",
        "visible_evidence_sha256s": "visible_evidence_sha256s",
        "route_reason": "route_reason",
        "route_forced": "route_forced",
        "event_type": "event_type",
        "route_target": "route_target",
        "is_env_action": "is_env_action",
        "env_action_text": "env_action_text",
        "env_action_owner": "env_action_owner",
        "env_observation_text": "env_observation_text",
        "env_reward": "env_reward",
        "env_done": "env_done",
        "tool_called": "tool_called",
        "tool_name": "tool_name",
        "tool_query": "tool_query",
        "tool_query_valid": "tool_query_valid",
        "tool_observation": "tool_observation",
        "tool_status": "tool_status",
        "tool_result_count": "tool_result_count",
        "tool_error": "tool_error",
        "final_answer_text": "final_answer_text",
    }

    # Recompute per-trajectory tool_callings from search_query events. The env's
    # counter (non-tensor "tool_callings") also counts Answer projections that the
    # underlying search env routes through its search-action parser, so it can
    # exceed the true Search-retrieval count. The hindsight/audit contract is
    # tool_callings == number of Search Agent retrieval events. No-op for Math
    # traces (no search_query events -> count stays 0).
    corrected_tool_callings: dict[Any, int] = {}
    _traj_uids = batch.non_tensor_batch.get("traj_uid") if batch.non_tensor_batch is not None else None
    _event_types = batch.non_tensor_batch.get("event_type") if batch.non_tensor_batch is not None else None
    if _traj_uids is not None and _event_types is not None:
        for _i in range(_batch_size(batch)):
            _tu = _to_jsonable(_traj_uids[_i])
            if _tu not in corrected_tool_callings:
                corrected_tool_callings[_tu] = 0
            if str(_event_types[_i]) == "search_query":
                corrected_tool_callings[_tu] += 1

    for row_index in range(_batch_size(batch)):
        record = {
            "run_id": str(run_id),
            "split": str(split),
            "global_step": int(global_step),
            "actor_update_count": int(actor_update_count),
        }
        for output_key, batch_key in direct_fields.items():
            record[output_key] = _to_jsonable(_non_tensor_row(batch, batch_key, row_index))
        # terminal_success is sourced from the float success_rate (the 'pass'
        # non-tensor column written at rollout_loop.py); the hindsight scorer and
        # the trace audit require a strict bool, so cast it here.
        record["terminal_success"] = bool(record.get("terminal_success", False))
        # Override the env's over-counting tool_callings with the true Search-event count.
        record["tool_callings"] = corrected_tool_callings.get(record.get("traj_uid"), record.get("tool_callings"))
        record["role"] = record["agent_id"]
        record["policy_snapshot_id"] = f"{run_id}:actor_update_{int(actor_update_count)}:{record['wg_id']}"

        if include_token_ids:
            prompts = _tensor_row(batch, "prompts", row_index)
            responses = _tensor_row(batch, "responses", row_index)
            attention_mask = _tensor_row(batch, "attention_mask", row_index)

            prompt_length = int(prompts.shape[-1])
            response_length = int(responses.shape[-1])
            prompt_mask = attention_mask[..., :prompt_length]
            response_mask = attention_mask[..., -response_length:]

            prompt_token_ids = _masked_list(prompts, prompt_mask)
            response_token_ids = _masked_list(responses, response_mask)
            record["prompt_token_ids"] = prompt_token_ids
            record["response_token_ids"] = response_token_ids
            record["response_mask"] = [1] * len(response_token_ids)
            record["prompt_token_count"] = len(prompt_token_ids)
            record["response_token_count"] = len(response_token_ids)

            rollout_log_probs = _tensor_row(batch, "rollout_log_probs", row_index)
            if include_rollout_log_probs and rollout_log_probs is not None:
                valid_rollout_log_probs = _masked_list(rollout_log_probs, response_mask)
                if len(valid_rollout_log_probs) != len(response_token_ids):
                    raise ValueError(f"Token/log-prob length mismatch for {record['event_uid']}")
                if not all(math.isfinite(float(value)) for value in valid_rollout_log_probs):
                    raise ValueError(f"Non-finite rollout log-prob for {record['event_uid']}")
                record["rollout_log_probs"] = valid_rollout_log_probs

        if include_offline_eval_info:
            record["offline_eval_info"] = _to_jsonable(_non_tensor_row(batch, "offline_eval_info", row_index))

        yield record


def dump_event_trace_jsonl(
    batch: Any,
    *,
    output_dir: str,
    split: str,
    global_step: int,
    part_index: int,
    run_id: str,
    actor_update_count: int,
    include_token_ids: bool = True,
    include_rollout_log_probs: bool = True,
    require_rollout_log_probs: bool = True,
    require_generation_finish_reason: bool = True,
    include_offline_eval_info: bool = False,
) -> tuple[str, int, int]:
    """Atomically dump one raw rollout batch and return path/count/part index."""

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    used_part_index = int(part_index)
    while True:
        filename = f"{split}_step_{int(global_step):08d}_part_{used_part_index:05d}.jsonl"
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):
            break
        used_part_index += 1

    temporary_path = f"{output_path}.tmp-{uuid.uuid4().hex}"
    record_count = 0
    try:
        with open(temporary_path, "x", encoding="utf-8") as file:
            for record in iter_event_trace_records(
                batch,
                split=split,
                global_step=global_step,
                run_id=run_id,
                actor_update_count=actor_update_count,
                include_token_ids=include_token_ids,
                include_rollout_log_probs=include_rollout_log_probs,
                require_rollout_log_probs=require_rollout_log_probs,
                require_generation_finish_reason=require_generation_finish_reason,
                include_offline_eval_info=include_offline_eval_info,
            ):
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                record_count += 1
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise

    return output_path, record_count, used_part_index


__all__ = [
    "EVENT_TRACE_SCHEMA_VERSION",
    "EVENT_TRACE_MANIFEST_SCHEMA_VERSION",
    "MATH_EVENT_TRACE_SCHEMA_VERSION",
    "SEARCH_EVENT_TRACE_SCHEMA_VERSION",
    "annotate_trajectory_events",
    "build_offline_eval_info",
    "build_event_trace_manifest",
    "build_math_offline_eval_info",
    "build_math_task_uid",
    "build_search_route_targets",
    "build_search_transition_fields",
    "build_task_uid",
    "dump_event_trace_jsonl",
    "dump_event_trace_manifest",
    "event_metadata_enabled",
    "event_trace_enabled",
    "event_trace_option",
    "get_event_trace_config",
    "iter_event_trace_records",
    "parse_search_verifier_decisions",
    "resolve_search_env_action_owners",
]
