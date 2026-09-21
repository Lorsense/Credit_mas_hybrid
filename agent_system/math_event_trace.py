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

"""Math team-event transition adapter and chain contract.

The Math orchestra runs an internal Solver/Verifier loop and executes exactly
one ``env.step`` with the latest Solver answer per trajectory.  This module
binds that single authoritative environment transition to the real events in
buffer (call) order, records submitted/transition references as explicit
event UIDs, and validates the complete chain contract before training.

Facts come from the actual rollout only: the environment reward/done are the
values returned by ``env.step``; approval flags are the orchestra's executed
decisions; nothing is inferred from tags, pass/fail, or chain length.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from verl.trainer.ppo.team_event_math_value import allowed_math_chains

MATH_EVENT_TRACE_ADAPTER_VERSION = "drmas.math_team_event.v1"
MATH_EVENT_TYPE_BY_AGENT = {
    "Solver Agent": "math_solution",
    "Verifier Agent": "math_verifier",
}


def math_event_type(agent_id: str) -> str:
    try:
        return MATH_EVENT_TYPE_BY_AGENT[str(agent_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Math producer {agent_id!r}; expected one of {sorted(MATH_EVENT_TYPE_BY_AGENT)}"
        ) from exc


def resolve_math_terminal_positions(
    multiagent_batch_buffer: Sequence[Mapping[str, Any]],
    active_masks: np.ndarray,
) -> List[Dict[str, Optional[int]]]:
    """Locate, per trajectory, the last active event and the last Solver event.

    ``multiagent_batch_buffer`` is in real agent-call order (S0, V0, S1, V1,
    S2).  For each rollout row the last *active* event carries the single env
    transition; the most recent active Solver event is the submitted answer.
    Inactive filler rows never own a transition or a submission.
    """

    active = np.asarray(active_masks, dtype=bool)
    plan: List[Dict[str, Optional[int]]] = []
    rows = None
    for position, entry in enumerate(multiagent_batch_buffer):
        agent_batch = entry["batch"]
        masks = agent_batch.non_tensor_batch.get("agent_active_mask")
        if masks is None:
            raise ValueError(f"Math trace adapter: {entry['agent_id']} batch lacks agent_active_mask")
        if rows is None:
            rows = len(masks)
        elif len(masks) != rows:
            raise ValueError("Math trace adapter: agent batches disagree on batch size")
    if rows is None:
        raise ValueError("Math trace adapter: empty multiagent buffer")
    if len(active) != rows:
        raise ValueError(f"Math trace adapter: active_masks length {len(active)} != rows {rows}")

    for row in range(rows):
        terminal_position: Optional[int] = None
        submitted_position: Optional[int] = None
        if active[row]:
            for position, entry in enumerate(multiagent_batch_buffer):
                masks = entry["batch"].non_tensor_batch["agent_active_mask"]
                if not bool(masks[row]):
                    continue
                terminal_position = position
                if str(entry["agent_id"]) == "Solver Agent":
                    submitted_position = position
        plan.append({
            "transition_owner_position": terminal_position,
            "submitted_solution_position": submitted_position,
        })
    return plan


def annotate_math_step_events(
    multiagent_batch_buffer: Sequence[Mapping[str, Any]],
    active_masks: np.ndarray,
    rewards: Any,
    dones: Any,
) -> None:
    """Attach per-event Math transition facts for the one executed env.step.

    Called in the rollout loop immediately after ``env.step`` returned the
    authoritative reward/done.  Only the trajectory's last active event of the
    step is the env action owner; every other event is explicitly non-owner
    with zero reward, not done, and a None owner — the producer names S0/S1/S2
    are identical, so a blanket owner field would misclassify non-terminal
    Solvers.
    """

    plan = resolve_math_terminal_positions(multiagent_batch_buffer, active_masks)
    rewards_np = np.asarray(rewards, dtype=np.float64)
    dones_np = np.asarray(dones, dtype=bool)
    rows = len(plan)
    if rewards_np.shape[0] != rows or dones_np.shape[0] != rows:
        raise ValueError("Math trace adapter: reward/done must cover every rollout row")
    for position, entry in enumerate(multiagent_batch_buffer):
        agent_id = str(entry["agent_id"])
        event_type = math_event_type(agent_id)
        agent_batch = entry["batch"]
        masks = agent_batch.non_tensor_batch["agent_active_mask"]
        is_owner = np.zeros(rows, dtype=bool)
        owner_values = np.empty(rows, dtype=object)
        owner_values[:] = None
        env_reward = np.zeros(rows, dtype=np.float64)
        env_done = np.zeros(rows, dtype=bool)
        env_step_index = np.zeros(rows, dtype=np.int64)
        buffer_position = np.full(rows, position, dtype=np.int64)
        submitted_position = np.empty(rows, dtype=object)
        transition_position = np.empty(rows, dtype=object)
        for row in range(rows):
            submitted_position[row] = plan[row]["submitted_solution_position"]
            transition_position[row] = plan[row]["transition_owner_position"]
            if not bool(masks[row]) or not bool(active_masks[row]):
                continue
            terminal = plan[row]["transition_owner_position"] == position
            if terminal:
                is_owner[row] = True
                owner_values[row] = agent_id
                env_reward[row] = float(rewards_np[row])
                env_done[row] = bool(dones_np[row])
        agent_batch.non_tensor_batch["event_type"] = np.array([event_type] * rows, dtype=object)
        agent_batch.non_tensor_batch["is_env_action"] = is_owner
        agent_batch.non_tensor_batch["env_action_owner"] = owner_values
        agent_batch.non_tensor_batch["env_reward"] = env_reward
        agent_batch.non_tensor_batch["env_done"] = env_done
        agent_batch.non_tensor_batch["env_step_index"] = env_step_index
        agent_batch.non_tensor_batch["math_buffer_position"] = buffer_position
        agent_batch.non_tensor_batch["math_submitted_solution_position"] = submitted_position
        agent_batch.non_tensor_batch["math_transition_owner_position"] = transition_position


def finalize_math_trajectory_events(events: List[dict], traj_uid: str, max_loop_num: int = 3) -> List[dict]:
    """Resolve submitted/transition UIDs and enforce the complete chain contract.

    ``events`` is one trajectory's real (active) event list in buffer order,
    already annotated with ``event_uid``/``event_index``/``role_event_index``.
    Raises on any incomplete chain, post-done event, missing authoritative
    transition, or non-alternating producer sequence.  Never rewrites the
    recorded approval decisions or the environment's reward/done.
    """

    if not events:
        raise ValueError(f"Math trajectory {traj_uid} has no events")
    if [event.get("event_index", i) for i, event in enumerate(events)] != list(range(len(events))):
        raise ValueError(f"Math trajectory {traj_uid}: events are not a contiguous annotated chain")
    if any(str(event.get("traj_uid")) != str(traj_uid) for event in events):
        raise ValueError(f"Math trajectory {traj_uid}: mixed trajectory identity")

    kinds = [str(event.get("event_type")) for event in events]
    for kind in kinds:
        if kind not in MATH_EVENT_TYPE_BY_AGENT.values():
            raise ValueError(f"Math trajectory {traj_uid}: unsupported event_type {kind!r}")
    # Alternation contract: S, then V, then S, ... ; first event is a Solver.
    for position, kind in enumerate(kinds):
        want = "math_solution" if position % 2 == 0 else "math_verifier"
        if kind != want:
            raise ValueError(
                f"Math trajectory {traj_uid}: non-alternating chain {kinds} at position {position}"
            )
    if tuple(kinds) not in set(allowed_math_chains(max_loop_num)):
        raise ValueError(
            f"Math trajectory {traj_uid}: event chain {kinds} is not a legal "
            f"max_loop_num={max_loop_num} Math chain"
        )

    terminal = events[-1]
    owners = [event for event in events if bool(event.get("is_env_action", False))]
    if len(owners) != 1 or owners[0] is not terminal:
        raise ValueError(
            f"Math trajectory {traj_uid}: exactly the last event must own the single env.step"
        )
    if any(bool(event.get("env_done", False)) for event in events[:-1]):
        raise ValueError(f"Math trajectory {traj_uid}: events after env_done")
    if not bool(terminal.get("env_done", False)):
        raise ValueError(
            f"Math trajectory {traj_uid}: missing authoritative env_done from env.step; "
            "refusing to infer terminal placement"
        )
    for event in events[:-1]:
        if event.get("env_action_owner") is not None or float(event.get("env_reward", 0.0) or 0.0) != 0.0:
            raise ValueError(f"Math trajectory {traj_uid}: non-terminal event carries owner/reward")
    if float(terminal.get("env_reward", 0.0)) not in (0.0, 1.0):
        raise ValueError(
            f"Math trajectory {traj_uid}: terminal env_reward "
            f"{terminal.get('env_reward')!r} is not the binary task signal"
        )
    if any(int(event.get("env_step_index", -1)) != 0 for event in events):
        raise ValueError(f"Math trajectory {traj_uid}: env_step_index must be 0 for the single env.step")

    # Solver/Verifier rounds: role_event_index must be the per-role call order.
    for role in ("Solver Agent", "Verifier Agent"):
        role_events = [event for event in events if str(event.get("agent_id")) == role]
        indices = [int(event.get("role_event_index", -1)) for event in role_events]
        if indices != list(range(len(role_events))):
            raise ValueError(f"Math trajectory {traj_uid}/{role}: role_event_index {indices} is not 0..T-1")

    # Resolve trajectory-level references to explicit event UIDs.
    by_position = {int(event["math_buffer_position"]): event for event in events if "math_buffer_position" in event}
    submitted_position = events[0].get("math_submitted_solution_position")
    transition_position = events[0].get("math_transition_owner_position")
    submitted_uid = None
    if submitted_position is not None:
        submitted_event = by_position.get(int(submitted_position))
        if submitted_event is None or submitted_event.get("event_type") != "math_solution":
            raise ValueError(f"Math trajectory {traj_uid}: submitted reference does not point at a Solver event")
        submitted_uid = str(submitted_event["event_uid"])
    transition_event = by_position.get(int(transition_position)) if transition_position is not None else None
    if transition_event is None or transition_event is not terminal:
        raise ValueError(
            f"Math trajectory {traj_uid}: transition owner reference must be the last completed event"
        )
    for event in events:
        event["submitted_solution_event_uid"] = submitted_uid
        event["transition_owner_event_uid"] = str(terminal["event_uid"])
        event["math_event_adapter_version"] = MATH_EVENT_TRACE_ADAPTER_VERSION
    return events


def math_trajectory_is_complete(events: List[dict]) -> bool:
    """Cheap post-annotation completeness probe used before any batch update."""

    try:
        finalize_math_trajectory_events(events, str(events[0].get("traj_uid")) if events else "")
    except ValueError:
        return False
    return True


__all__ = [
    "MATH_EVENT_TRACE_ADAPTER_VERSION",
    "MATH_EVENT_TYPE_BY_AGENT",
    "annotate_math_step_events",
    "finalize_math_trajectory_events",
    "math_event_type",
    "math_trajectory_is_complete",
    "resolve_math_terminal_positions",
]
