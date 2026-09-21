"""Tests for the Math team-event transition adapter (math_event_trace)."""

import numpy as np
import pytest

from agent_system.math_event_trace import (
    MATH_EVENT_TRACE_ADAPTER_VERSION,
    annotate_math_step_events,
    finalize_math_trajectory_events,
    math_event_type,
    resolve_math_terminal_positions,
)


class FakeAgentBatch:
    def __init__(self, agent_active_mask):
        self.non_tensor_batch = {"agent_active_mask": np.asarray(agent_active_mask, dtype=bool)}


def _buffer(active_masks):
    """Full five-event chain S0,V0,S1,V1,S2 with per-event activity masks."""

    solver_active = np.asarray(active_masks, dtype=bool)  # placeholder, replaced below
    return solver_active


def _make_buffer(chain, rows):
    """Build a multiagent buffer where every row runs `chain` events."""

    entries = []
    for position, agent_id in enumerate(chain):
        mask = np.ones(rows, dtype=bool)
        entries.append({"agent_id": agent_id, "batch": FakeAgentBatch(mask)})
    return entries


def _make_buffer_rowwise(chains_per_row):
    """Build a buffer where each row executes its own producer sequence."""

    length = max(len(chain) for chain in chains_per_row)
    rows = len(chains_per_row)
    entries = []
    for position in range(length):
        agent_id = "Solver Agent" if position % 2 == 0 else "Verifier Agent"
        mask = np.zeros(rows, dtype=bool)
        for row, chain in enumerate(chains_per_row):
            if position < len(chain):
                assert chain[position] == agent_id
                mask[row] = True
        entries.append({"agent_id": agent_id, "batch": FakeAgentBatch(mask)})
    return entries


def _extract_trajectory(buffer, row, active_masks):
    events = []
    for position, entry in enumerate(buffer):
        if not entry["batch"].non_tensor_batch["agent_active_mask"][row]:
            continue
        if not active_masks[row]:
            continue
        record = {}
        for key, values in entry["batch"].non_tensor_batch.items():
            record[key] = values[row]
        events.append(record)
    return events


def _annotate(events, traj_uid):
    for index, event in enumerate(events):
        event["event_uid"] = f"{traj_uid}:{index}"
        event["event_index"] = index
        event["event_count"] = len(events)
        event["traj_uid"] = traj_uid
        role = event["agent_id"] = "Solver Agent" if event["event_type"] == "math_solution" else "Verifier Agent"
        event.setdefault("role_event_index", None)
    role_counters = {}
    for event in events:
        role = event["agent_id"]
        event["role_event_index"] = role_counters.get(role, 0)
        role_counters[role] = role_counters.get(role, 0) + 1
    for event in events:
        event["role_event_count"] = role_counters[event["agent_id"]]
    return events


def test_event_type_mapping_and_rejection():
    assert math_event_type("Solver Agent") == "math_solution"
    assert math_event_type("Verifier Agent") == "math_verifier"
    with pytest.raises(ValueError, match="Unknown Math producer"):
        math_event_type("Answer Agent")


def test_full_chain_terminal_owner_is_last_solver():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=3)
    active = np.ones(3, dtype=bool)
    rewards = np.array([1.0, 0.0, 1.0])
    dones = np.ones(3, dtype=bool)
    annotate_math_step_events(buffer, active, rewards, dones)

    plan = resolve_math_terminal_positions(buffer, active)
    assert plan[0]["transition_owner_position"] == 4
    assert plan[0]["submitted_solution_position"] == 4

    solver_batch = buffer[0]["batch"].non_tensor_batch
    assert solver_batch["event_type"][0] == "math_solution"
    assert solver_batch["is_env_action"][0] == False
    assert solver_batch["env_action_owner"][0] is None
    assert solver_batch["env_reward"][0] == 0.0
    assert solver_batch["env_done"][0] == False
    terminal_batch = buffer[4]["batch"].non_tensor_batch
    assert terminal_batch["is_env_action"][0] == True
    assert terminal_batch["env_action_owner"][0] == "Solver Agent"
    assert terminal_batch["env_reward"][0] == 1.0
    assert terminal_batch["env_done"][0] == True
    assert terminal_batch["env_step_index"][0] == 0


def test_early_stop_after_verifier_owns_transition():
    chains = [
        ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"],
        ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"],
        ["Solver Agent", "Verifier Agent"],  # approved after V0
    ]
    buffer = _make_buffer_rowwise(chains)
    active = np.ones(3, dtype=bool)
    rewards = np.array([1.0, 0.0, 1.0])
    annotate_math_step_events(buffer, active, rewards, np.ones(3, dtype=bool))

    events = _extract_trajectory(buffer, 2, active)
    assert [e["event_type"] for e in events] == ["math_solution", "math_verifier"]
    # Early stop: the Verifier owns the transition, the last Solver submitted.
    assert events[-1]["is_env_action"] == True
    assert events[-1]["env_action_owner"] == "Verifier Agent"
    assert events[-1]["env_reward"] == 1.0
    assert events[0]["is_env_action"] == False
    assert events[0]["env_action_owner"] is None
    _annotate(events, "t2")
    finalized = finalize_math_trajectory_events(events, "t2")
    assert finalized[0]["submitted_solution_event_uid"] == "t2:0"
    assert finalized[0]["transition_owner_event_uid"] == "t2:1"


def test_finalize_full_chain_references_and_contract():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    annotate_math_step_events(buffer, active, np.array([1.0]), np.array([True]))
    events = _extract_trajectory(buffer, 0, active)
    _annotate(events, "t0")
    finalized = finalize_math_trajectory_events(events, "t0")
    for event in finalized:
        assert event["submitted_solution_event_uid"] == "t0:4"
        assert event["transition_owner_event_uid"] == "t0:4"
        assert event["math_event_adapter_version"] == MATH_EVENT_TRACE_ADAPTER_VERSION


def test_finalize_rejects_missing_authoritative_done():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    # env.step has not run: done/reward are falsy everywhere.
    annotate_math_step_events(buffer, active, np.array([0.0]), np.array([False]))
    events = _extract_trajectory(buffer, 0, active)
    _annotate(events, "t0")
    with pytest.raises(ValueError, match="missing authoritative env_done"):
        finalize_math_trajectory_events(events, "t0")


def test_finalize_rejects_non_binary_reward():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    annotate_math_step_events(buffer, active, np.array([0.5]), np.array([True]))
    events = _extract_trajectory(buffer, 0, active)
    _annotate(events, "t0")
    with pytest.raises(ValueError, match="binary"):
        finalize_math_trajectory_events(events, "t0")


def test_finalize_rejects_non_alternating_chain():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    annotate_math_step_events(buffer, active, np.array([1.0]), np.array([True]))
    events = _extract_trajectory(buffer, 0, active)
    events = events[1:]  # drop S0 → chain starts with Verifier
    _annotate(events, "t0")
    with pytest.raises(ValueError, match="non-alternating"):
        finalize_math_trajectory_events(events, "t0")


def test_finalize_rejects_reward_on_nonterminal_event():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    annotate_math_step_events(buffer, active, np.array([1.0]), np.array([True]))
    events = _extract_trajectory(buffer, 0, active)
    _annotate(events, "t0")
    events[0]["env_reward"] = 1.0  # a duplicated pass on a non-terminal Solver
    with pytest.raises(ValueError, match="owner/reward"):
        finalize_math_trajectory_events(events, "t0")


def test_finalize_rejects_bare_mid_chain_solver_terminal():
    chain = ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]
    buffer = _make_buffer(chain, rows=1)
    active = np.ones(1, dtype=bool)
    annotate_math_step_events(buffer, active, np.array([1.0]), np.array([True]))
    events = _extract_trajectory(buffer, 0, active)
    events = events[:3]  # S0,V0,S1 — a bare S1 terminal is not a legal Math chain
    _annotate(events, "t0")
    # Mark the fabricated last event as the owner to test the shape contract.
    events[-1]["is_env_action"] = True
    events[-1]["env_action_owner"] = "Solver Agent"
    with pytest.raises(ValueError, match="legal"):
        finalize_math_trajectory_events(events, "t0")


def test_inactive_rows_never_own_transitions():
    chains = [["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"]]
    buffer = _make_buffer_rowwise(chains + [[]])  # second row inactive
    active = np.array([True, False])
    annotate_math_step_events(buffer, active, np.array([1.0, 0.0]), np.array([True, False]))
    plan = resolve_math_terminal_positions(buffer, active)
    assert plan[1]["transition_owner_position"] is None
    assert plan[1]["submitted_solution_position"] is None


def test_differing_stop_points_in_one_batch():
    chains = [
        ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent", "Solver Agent"],
        ["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent"],
        ["Solver Agent", "Verifier Agent"],
    ]
    buffer = _make_buffer_rowwise(chains)
    active = np.ones(3, dtype=bool)
    rewards = np.array([1.0, 0.0, 1.0])
    annotate_math_step_events(buffer, active, rewards, np.ones(3, dtype=bool))
    for row, expected_terminal in enumerate((4, 3, 1)):
        events = _extract_trajectory(buffer, row, active)
        assert len(events) == expected_terminal + 1
        assert events[-1]["is_env_action"] == True
        assert events[-1]["env_reward"] == rewards[row]
        assert all(not event["is_env_action"] for event in events[:-1])
        _annotate(events, f"t{row}")
        finalize_math_trajectory_events(events, f"t{row}")
