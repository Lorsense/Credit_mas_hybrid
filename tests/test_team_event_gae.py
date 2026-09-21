"""CPU tests for cross-agent Search-event TD/GAE credit assignment."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.utils import (
    adjust_batch,
    combine_batches,
    prepare_team_event_optimizer_batch,
    split_batch_by_wg_ids,
)
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss,
    compute_sample_weight_normalizer,
    compute_team_event_gae_advantage,
)
from verl.trainer.ppo.metric_utils import _compute_team_event_credit_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    validate_team_event_sample_weights,
)


def _opaque(values):
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def _run(records, *, gamma=1.0, lam=1.0, normalize=False, scores=None, rewards=None, **kwargs):
    width = 3
    batch_size = len(records)
    token_scores = torch.zeros((batch_size, width), dtype=torch.float32)
    token_rewards = torch.zeros((batch_size, width), dtype=torch.float32)
    if scores is not None:
        token_scores[:, -1] = torch.tensor(scores, dtype=torch.float32)
    if rewards is not None:
        token_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    mask = torch.ones((batch_size, width), dtype=torch.float32)
    unique_records = {}
    for record in records:
        unique_records.setdefault(record["event_uid"], record)
    role_indices = {}
    role_counts = {}
    records_by_trajectory = {}
    for record in unique_records.values():
        records_by_trajectory.setdefault(record["traj"], []).append(record)
    for trajectory_records in records_by_trajectory.values():
        trajectory_records.sort(key=lambda record: record["index"])
        per_role = {}
        for record in trajectory_records:
            per_role.setdefault(record["agent"], []).append(record)
        for role_records in per_role.values():
            for role_index, record in enumerate(role_records):
                role_indices[record["event_uid"]] = role_index
                role_counts[record["event_uid"]] = len(role_records)
    advantages, returns, diagnostics = compute_team_event_gae_advantage(
        token_level_rewards=token_rewards,
        token_level_scores=token_scores,
        response_mask=mask,
        uid=_opaque([record.get("uid", "task") for record in records]),
        traj_index=_opaque([record["traj"] for record in records]),
        event_uid=_opaque([record["event_uid"] for record in records]),
        event_index=_opaque([record["index"] for record in records]),
        event_count=_opaque([record["count"] for record in records]),
        role_event_index=_opaque(
            [record.get("role_index", role_indices[record["event_uid"]]) for record in records]
        ),
        role_event_count=_opaque(
            [record.get("role_count", role_counts[record["event_uid"]]) for record in records]
        ),
        agent_id=_opaque([record["agent"] for record in records]),
        event_type=_opaque([record["type"] for record in records]),
        env_step_index=_opaque([record["step"] for record in records]),
        is_env_action=_opaque([record["owner"] for record in records]),
        env_action_owner=_opaque(
            [
                record.get(
                    "env_owner",
                    record["agent"] if record["owner"] else "Answer Agent",
                )
                for record in records
            ]
        ),
        env_reward=_opaque([record.get("reward") for record in records]),
        env_done=_opaque([record.get("done") for record in records]),
        state_chats=_opaque([record["state"] for record in records]),
        is_action_valid=_opaque([record.get("valid", True) for record in records]),
        gamma=gamma,
        lam=lam,
        normalize_advantages=normalize,
        **kwargs,
    )
    return advantages[:, 0], returns[:, 0], diagnostics


def _chain(traj="t1", terminal_reward=1.0):
    roles = ["Verifier Agent", "Search Agent", "Verifier Agent", "Answer Agent"]
    types = ["verifier", "search_query", "verifier", "final_answer"]
    owners = [False, True, False, True]
    steps = [0, 0, 1, 1]
    records = []
    for index in range(4):
        records.append(
            {
                "traj": traj,
                "event_uid": f"{traj}:{index}",
                "index": index,
                "count": 4,
                "agent": roles[index],
                "type": types[index],
                "step": steps[index],
                "owner": owners[index],
                "env_owner": roles[index + 1] if index in (0, 2) else roles[index],
                "reward": terminal_reward if index == 3 else None,
                "done": True if index == 3 else None,
                "state": [{"role": "user", "content": f"{traj}-state-{index}"}],
            }
        )
    return records


def test_global_chain_uses_internal_and_environment_boundaries():
    advantages, returns, _ = _run(_chain(), gamma=0.5, lam=0.5)
    # A(answer)=1; internal V->A keeps it at 1; the Search env boundary applies
    # gamma*lambda=.25; the first internal V->Search keeps .25.
    expected = torch.tensor([0.25, 0.25, 1.0, 1.0])
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(returns, expected, atol=1e-6)


def test_loto_value_excludes_the_current_trajectory():
    records = [
        {
            "traj": "success",
            "event_uid": "success:0",
            "index": 0,
            "count": 1,
            "agent": "Answer Agent",
            "type": "final_answer",
            "step": 0,
            "owner": True,
            "reward": 1.0,
            "done": True,
            "state": ["same-state"],
        },
        {
            "traj": "failure",
            "event_uid": "failure:0",
            "index": 0,
            "count": 1,
            "agent": "Answer Agent",
            "type": "final_answer",
            "step": 0,
            "owner": True,
            "reward": 0.0,
            "done": True,
            "state": ["same-state"],
        },
    ]
    advantages, _, diagnostics = _run(records)
    assert torch.allclose(diagnostics["event_values"], torch.tensor([0.0, 1.0]))
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0]))


def test_singleton_exact_state_falls_back_to_role_type_and_depth():
    records = []
    for trajectory, reward in (("success", 1.0), ("failure", 0.0)):
        records.append(
            {
                "traj": trajectory,
                "event_uid": f"{trajectory}:0",
                "index": 0,
                "count": 1,
                "agent": "Answer Agent",
                "type": "final_answer",
                "step": 2,
                "owner": True,
                "reward": reward,
                "done": True,
                "state": [f"unique-{trajectory}"],
            }
        )
    advantages, _, diagnostics = _run(records)
    assert torch.allclose(diagnostics["event_values"], torch.tensor([0.0, 1.0]))
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0]))


def test_adjust_batch_duplicates_and_reordering_do_not_recompute_events():
    base_records = _chain("t1", 1.0) + _chain("t2", 0.0)
    base_adv, _, _ = _run(base_records, gamma=0.9, lam=0.8)
    base_by_uid = {
        record["event_uid"]: float(base_adv[index])
        for index, record in enumerate(base_records)
    }

    copied = base_records + [dict(base_records[1]), dict(base_records[7])]
    order = [8, 3, 5, 0, 9, 2, 7, 1, 6, 4]
    shuffled = [copied[index] for index in order]
    shuffled_adv, _, _ = _run(shuffled, gamma=0.9, lam=0.8)
    for index, record in enumerate(shuffled):
        assert abs(float(shuffled_adv[index]) - base_by_uid[record["event_uid"]]) < 1e-6


def test_invalid_and_kl_penalties_are_event_local():
    record = {
        "traj": "t1",
        "event_uid": "t1:0",
        "index": 0,
        "count": 1,
        "agent": "Answer Agent",
        "type": "final_answer",
        "step": 0,
        "owner": True,
        "reward": 1.0,
        "done": True,
        "state": ["state"],
        "valid": False,
    }
    advantages, _, diagnostics = _run(
        [record],
        scores=[0.0],
        rewards=[-0.2],
        invalid_action_penalty_coef=0.1,
    )
    assert torch.allclose(diagnostics["event_rewards"], torch.tensor([0.7]), atol=1e-6)
    assert torch.allclose(advantages, torch.tensor([0.7]), atol=1e-6)


def test_non_owner_cannot_carry_team_reward():
    record = _chain()[0]
    record["count"] = 1
    record["reward"] = 1.0
    record["done"] = True
    with pytest.raises(ValueError, match="Non-owner event"):
        _run([record])


def test_incomplete_trajectory_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        _run(_chain()[:-1])


def _long_search_chain(traj="t1", terminal_reward=1.0):
    roles = [
        "Verifier Agent",
        "Search Agent",
        "Verifier Agent",
        "Search Agent",
        "Verifier Agent",
        "Answer Agent",
    ]
    types = ["verifier", "search_query", "verifier", "search_query", "verifier", "final_answer"]
    owners = [False, True, False, True, False, True]
    steps = [0, 0, 1, 1, 2, 2]
    records = []
    for index, (role, event_type, owner, step) in enumerate(zip(roles, types, owners, steps)):
        records.append(
            {
                "traj": traj,
                "event_uid": f"{traj}:{index}",
                "index": index,
                "count": len(roles),
                "agent": role,
                "type": event_type,
                "step": step,
                "owner": owner,
                "env_owner": role if owner else roles[index + 1],
                "reward": terminal_reward if index == len(roles) - 1 else None,
                "done": True if index == len(roles) - 1 else None,
                "state": [{"role": "user", "content": f"{traj}-state-{index}"}],
            }
        )
    return records


def test_agent_local_smdp_accumulates_cross_agent_segments_and_terminal_tail():
    records = _long_search_chain()
    advantages, _, diagnostics = _run(
        records,
        gamma=0.5,
        lam=0.5,
        internal_gamma=1.0,
        internal_lam=1.0,
        agent_local_mode="shadow",
        agent_local_lam=0.5,
        agent_local_internal_lam=1.0,
    )

    expected_team = torch.tensor([0.0625, 0.0625, 0.25, 0.25, 1.0, 1.0])
    expected_local = torch.tensor([0.0625, 0.125, 0.25, 0.5, 1.0, 1.0])
    assert torch.allclose(advantages, expected_team, atol=1e-6)
    assert torch.allclose(diagnostics["event_team_raw_advantages"], expected_team, atol=1e-6)
    assert torch.allclose(diagnostics["event_local_raw_advantages"], expected_local, atol=1e-6)
    # Search S1's local segment crosses V2 and reaches terminal Answer, so its
    # accumulated reward is discounted once by the Search environment edge.
    assert diagnostics["event_local_segment_rewards"][3] == pytest.approx(0.5)
    assert diagnostics["event_local_span_events"][3] == pytest.approx(3.0)


def test_agent_local_shadow_and_zero_alpha_preserve_team_actor_advantage():
    # Three distinct trajectories force the trajectory-balanced normalization
    # path to run; this is a strict alpha=0 compatibility test, not a
    # singleton shortcut.
    records = (
        _long_search_chain("t1", 1.0)
        + _long_search_chain("t2", 0.3)
        + _long_search_chain("t3", 0.0)
    )
    base_advantages, base_returns, base_diagnostics = _run(
        records,
        gamma=0.7,
        lam=0.6,
        normalize=True,
        agent_local_mode="off",
    )
    for mode in ("shadow", "blend"):
        actual_advantages, actual_returns, diagnostics = _run(
            records,
            gamma=0.7,
            lam=0.6,
            normalize=True,
            agent_local_mode=mode,
            agent_local_mix_alpha=0.0,
            agent_local_lam=0.4,
        )
        assert torch.equal(actual_advantages, base_advantages)
        assert torch.equal(actual_returns, base_returns)
        assert torch.equal(diagnostics["event_raw_advantages"], base_diagnostics["event_raw_advantages"])


def test_agent_local_blends_raw_advantages_before_normalization():
    records = _long_search_chain()
    advantages, returns, diagnostics = _run(
        records,
        gamma=0.5,
        lam=0.5,
        normalize=False,
        agent_local_mode="blend",
        agent_local_mix_alpha=0.25,
        agent_local_lam=0.5,
    )
    expected = 0.75 * diagnostics["event_team_raw_advantages"] + 0.25 * diagnostics[
        "event_local_raw_advantages"
    ]
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(diagnostics["event_raw_advantages"], expected, atol=1e-6)
    assert torch.allclose(returns, expected + diagnostics["event_values"], atol=1e-6)


def test_agent_local_equals_team_when_all_trace_coefficients_are_one():
    _, _, diagnostics = _run(
        _long_search_chain(),
        gamma=1.0,
        lam=1.0,
        internal_gamma=1.0,
        internal_lam=1.0,
        agent_local_mode="shadow",
        agent_local_lam=1.0,
        agent_local_internal_lam=1.0,
    )
    assert torch.allclose(
        diagnostics["event_local_raw_advantages"],
        diagnostics["event_team_raw_advantages"],
        atol=1e-6,
    )


def test_agent_local_all_one_equivalence_with_nonzero_loto_values():
    # Independent closed-form anchor with two peer trajectories. Every event
    # in the success trajectory uses V=0 from the failure peer; every event in
    # the failure trajectory uses V=1 from the success peer. With every trace
    # coefficient equal to one, Team and same-agent SMDP returns must still be
    # identical despite repeated Verifier/Search events and nonzero baselines.
    records = _long_search_chain("success", 1.0) + _long_search_chain("failure", 0.0)
    _, _, diagnostics = _run(
        records,
        gamma=1.0,
        lam=1.0,
        internal_gamma=1.0,
        internal_lam=1.0,
        agent_local_mode="shadow",
        agent_local_lam=1.0,
        agent_local_internal_lam=1.0,
    )
    expected_values = torch.tensor([0.0] * 6 + [1.0] * 6)
    expected_raw = torch.tensor([1.0] * 6 + [-1.0] * 6)
    assert torch.allclose(diagnostics["event_values"], expected_values, atol=1e-6)
    assert torch.allclose(diagnostics["event_team_raw_advantages"], expected_raw, atol=1e-6)
    assert torch.allclose(diagnostics["event_local_raw_advantages"], expected_raw, atol=1e-6)


def test_agent_local_lambda_zero_keeps_only_current_smdp_td_segment():
    _, _, diagnostics = _run(
        _long_search_chain(),
        gamma=0.5,
        lam=0.5,
        agent_local_mode="shadow",
        agent_local_lam=0.0,
        agent_local_internal_lam=1.0,
    )
    assert torch.allclose(
        diagnostics["event_local_raw_advantages"],
        torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]),
        atol=1e-6,
    )


def test_agent_local_internal_lambda_is_multiplied_across_internal_edges():
    _, _, diagnostics = _run(
        _long_search_chain(),
        gamma=0.5,
        lam=0.5,
        agent_local_mode="shadow",
        agent_local_lam=0.5,
        agent_local_internal_lam=0.25,
    )
    assert torch.allclose(
        diagnostics["event_local_raw_advantages"],
        torch.tensor([0.00390625, 0.03125, 0.0625, 0.5, 1.0, 1.0]),
        atol=1e-6,
    )


def test_agent_local_duplicate_and_reorder_do_not_change_unique_event_diagnostics():
    base_records = _long_search_chain("t1", 1.0) + _long_search_chain("t2", 0.2)
    _, _, base = _run(
        base_records,
        gamma=0.8,
        lam=0.7,
        agent_local_mode="blend",
        agent_local_mix_alpha=0.25,
        agent_local_lam=0.6,
    )
    diagnostic_names = (
        "event_team_raw_advantages",
        "event_local_raw_advantages",
        "event_raw_advantages",
        "event_local_segment_rewards",
        "event_local_segment_gammas",
        "event_local_segment_lambdas",
    )
    expected_by_uid = {
        name: {
            record["event_uid"]: float(base[name][row])
            for row, record in enumerate(base_records)
        }
        for name in diagnostic_names
    }

    copied = base_records + [dict(base_records[1]), dict(base_records[3]), dict(base_records[9])]
    order = np.random.default_rng(31).permutation(len(copied))
    shuffled = [copied[index] for index in order]
    _, _, actual = _run(
        shuffled,
        gamma=0.8,
        lam=0.7,
        agent_local_mode="blend",
        agent_local_mix_alpha=0.25,
        agent_local_lam=0.6,
    )
    for name in diagnostic_names:
        for row, record in enumerate(shuffled):
            assert float(actual[name][row]) == pytest.approx(
                expected_by_uid[name][record["event_uid"]], abs=1e-6
            )


def test_agent_local_blend_is_normalized_once_after_raw_mixing():
    records = (
        _long_search_chain("t1", 1.0)
        + _long_search_chain("t2", 0.3)
        + _long_search_chain("t3", 0.0)
    )
    alpha = 0.25
    advantages, _, diagnostics = _run(
        records,
        gamma=0.8,
        lam=0.7,
        normalize=True,
        agent_local_mode="blend",
        agent_local_mix_alpha=alpha,
        agent_local_lam=0.5,
    )
    mixed_raw = (1.0 - alpha) * diagnostics["event_team_raw_advantages"] + alpha * diagnostics[
        "event_local_raw_advantages"
    ]
    expected = mixed_raw.clone()
    agents = [record["agent"] for record in records]
    trajectories = [record["traj"] for record in records]
    for agent in sorted(set(agents)):
        indices = [index for index, value in enumerate(agents) if value == agent]
        counts = {
            trajectory: sum(
                1
                for index in indices
                if trajectories[index] == trajectory
            )
            for trajectory in set(trajectories[index] for index in indices)
        }
        weights = torch.tensor(
            [1.0 / counts[trajectories[index]] for index in indices], dtype=torch.float32
        )
        samples = mixed_raw[indices]
        mean = torch.sum(weights * samples) / torch.sum(weights)
        centered = samples - mean
        variance = torch.sum(weights * centered.square()) / torch.sum(weights)
        expected[indices] = centered / (torch.sqrt(variance) + 1e-6)

    assert torch.allclose(diagnostics["event_raw_advantages"], mixed_raw, atol=1e-6)
    assert torch.allclose(advantages, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"agent_local_mode": "unknown"}, "agent_local_mode"),
        ({"agent_local_mode": "blend", "agent_local_mix_alpha": 1.1}, "mix_alpha"),
        ({"agent_local_mode": "shadow", "agent_local_lam": -0.1}, "agent_local_lam"),
    ],
)
def test_agent_local_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _run(_chain(), **kwargs)


def test_agent_local_validates_role_event_metadata():
    records = _long_search_chain()
    records[2]["role_index"] = 0
    with pytest.raises(ValueError, match="role-event indices"):
        _run(records, agent_local_mode="shadow")


def test_agent_local_metrics_use_unique_events_instead_of_adjustment_rows():
    data = DataProto.from_dict(
        tensors={
            # e0 is copied twice; its diagnostics must still count once.
            "event_team_raw_advantages": torch.tensor([1.0, 1.0, -1.0]),
            "event_local_raw_advantages": torch.tensor([-1.0, -1.0, -1.0]),
            "event_raw_advantages": torch.tensor([0.5, 0.5, -1.0]),
            "event_local_span_events": torch.tensor([3.0, 3.0, 1.0]),
            "event_local_span_env_steps": torch.tensor([2.0, 2.0, 1.0]),
            "event_mix_alpha": torch.tensor([0.25, 0.25, 0.25]),
        },
        non_tensors={
            "event_uid": np.array(["e0", "e0", "e1"], dtype=object),
            "role_event_count": np.array([2, 2, 1], dtype=object),
        },
    )

    metrics = _compute_team_event_credit_metrics(data, "search")
    assert metrics["credit/search/unique_event_count"] == 2.0
    assert metrics["credit/search/team_raw/mean"] == pytest.approx(0.0)
    assert metrics["credit/search/local_raw/mean"] == pytest.approx(-1.0)
    assert metrics["credit/search/team_local/sign_conflict_rate"] == pytest.approx(0.5)
    assert metrics["credit/search/local_span/events_mean"] == pytest.approx(2.0)
    assert metrics["credit/search/role_sequence/repeated_event_ratio"] == pytest.approx(0.5)
    assert metrics["credit/search/mix_alpha"] == pytest.approx(0.25)


def test_team_only_mode_still_reports_unique_event_credit_metrics():
    data = DataProto.from_dict(
        tensors={
            "event_team_raw_advantages": torch.tensor([1.0, 1.0, -1.0]),
            "event_raw_advantages": torch.tensor([1.0, 1.0, -1.0]),
        },
        non_tensors={
            "event_uid": np.array(["e0", "e0", "e1"], dtype=object),
            "role_event_count": np.array([2, 2, 1], dtype=object),
        },
    )
    metrics = _compute_team_event_credit_metrics(data, "search")
    assert metrics["credit/search/unique_event_count"] == 2.0
    assert metrics["credit/search/team_raw/mean"] == pytest.approx(0.0)
    assert metrics["credit/search/team_raw/rms"] == pytest.approx(1.0)
    assert metrics["credit/search/mixed_raw/rms"] == pytest.approx(1.0)
    assert "credit/search/local_raw/rms" not in metrics


def test_compute_advantage_routes_nested_agent_local_configuration():
    records = _long_search_chain()
    role_indices = {}
    role_counts = {}
    per_role = {}
    for record in records:
        per_role.setdefault(record["agent"], []).append(record)
    for role_records in per_role.values():
        for role_index, record in enumerate(role_records):
            role_indices[record["event_uid"]] = role_index
            role_counts[record["event_uid"]] = len(role_records)

    rows = len(records)
    data = DataProto.from_dict(
        tensors={
            "responses": torch.ones((rows, 3), dtype=torch.long),
            "attention_mask": torch.ones((rows, 4), dtype=torch.long),
            "response_mask": torch.ones((rows, 3), dtype=torch.float32),
            "token_level_scores": torch.zeros((rows, 3), dtype=torch.float32),
            "token_level_rewards": torch.zeros((rows, 3), dtype=torch.float32),
        },
        non_tensors={
            "task_type": np.array(["search"] * rows, dtype=object),
            "uid": np.array(["task"] * rows, dtype=object),
            "traj_uid": _opaque([record["traj"] for record in records]),
            "event_uid": _opaque([record["event_uid"] for record in records]),
            "event_index": _opaque([record["index"] for record in records]),
            "event_count": _opaque([record["count"] for record in records]),
            "role_event_index": _opaque(
                [role_indices[record["event_uid"]] for record in records]
            ),
            "role_event_count": _opaque(
                [role_counts[record["event_uid"]] for record in records]
            ),
            "agent_id": _opaque([record["agent"] for record in records]),
            "event_type": _opaque([record["type"] for record in records]),
            "env_step_index": _opaque([record["step"] for record in records]),
            "is_env_action": _opaque([record["owner"] for record in records]),
            "env_action_owner": _opaque([record["env_owner"] for record in records]),
            "env_reward": _opaque([record.get("reward") for record in records]),
            "env_done": _opaque([record.get("done") for record in records]),
            "hcapo_state_chat": _opaque([record["state"] for record in records]),
            "is_action_valid": np.ones(rows, dtype=bool),
        },
    )
    configured = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.TEAM_EVENT_GAE,
        gamma=0.5,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        team_event_gae=OmegaConf.create(
            {
                "internal_gamma": 1.0,
                "internal_lam": 1.0,
                "include_kl_shaping": True,
                "normalize_advantages": False,
                "agent_local": {
                    "mode": "blend",
                    "mix_alpha": 0.25,
                    "local_lambda": 0.5,
                    "local_internal_lambda": 1.0,
                },
            }
        ),
        use_invalid_action_penalty=True,
        invalid_action_penalty_coef=0.0,
    )
    expected, _, expected_diagnostics = _run(
        records,
        gamma=0.5,
        lam=0.5,
        normalize=False,
        agent_local_mode="blend",
        agent_local_mix_alpha=0.25,
        agent_local_lam=0.5,
    )
    assert torch.allclose(configured.batch["advantages"][:, 0], expected, atol=1e-6)
    assert torch.allclose(
        configured.batch["event_local_raw_advantages"],
        expected_diagnostics["event_local_raw_advantages"],
        atol=1e-6,
    )


def _adjust_config(divisor=4, *, adaptive=False, updates=1, world_size=1):
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "team_event_gae", "use_kl_in_reward": False},
            "trainer": {"n_gpus_per_node": world_size, "nnodes": 1},
            "actor_rollout_ref": {
                "actor": {
                    "use_adaptive_ppo_mini_batch_size": adaptive,
                    "ppo_mini_update_num": updates,
                    "ppo_epochs": 1,
                    "ppo_micro_batch_size_per_gpu": divisor,
                    "ppo_mini_batch_size": divisor,
                    "use_kl_loss": False,
                },
                "rollout": {
                    "log_prob_micro_batch_size_per_gpu": divisor,
                    "multi_turn": {"enable": False},
                },
                "ref": {"log_prob_micro_batch_size_per_gpu": divisor},
            },
            "critic": {"ulysses_sequence_parallel_size": 1},
        }
    )


def test_adjust_batch_assigns_inverse_event_multiplicity_weights():
    data = DataProto.from_dict(
        tensors={"input_ids": torch.arange(6, dtype=torch.long).reshape(3, 2)},
        non_tensors={"event_uid": np.array(["e0", "e1", "e2"], dtype=object)},
    )
    np.random.seed(7)
    adjusted = adjust_batch(_adjust_config(), data, wg_id="search")

    assert len(adjusted) == 4
    assert adjusted.meta_info["search/unique_event_count"] == 3
    assert adjusted.meta_info["search/adjusted_row_count"] == 4
    assert adjusted.meta_info["search/sample_weight_sum"] == pytest.approx(3.0)
    assert adjusted.non_tensor_batch["is_adjustment_copy"].tolist() == [False, False, False, True]

    weights = adjusted.batch["sample_weight"].cpu().numpy()
    event_uids = adjusted.non_tensor_batch["event_uid"]
    dup_counts = adjusted.non_tensor_batch["sample_dup_count"]
    for event_uid in np.unique(event_uids):
        rows = event_uids == event_uid
        assert weights[rows].sum() == pytest.approx(1.0)
        assert np.all(dup_counts[rows] == rows.sum())


def test_adjust_batch_float32_weight_sum_is_stable_for_large_padding_ratio():
    unique_rows = 43
    data = DataProto.from_dict(
        tensors={"input_ids": torch.arange(unique_rows * 2, dtype=torch.long).reshape(unique_rows, 2)},
        non_tensors={"event_uid": np.array([f"e{i}" for i in range(unique_rows)], dtype=object)},
    )
    np.random.seed(11)
    adjusted = adjust_batch(_adjust_config(divisor=256), data, wg_id="verifier")

    assert len(adjusted) == 256
    assert adjusted.meta_info["verifier/unique_event_count"] == unique_rows
    assert adjusted.meta_info["verifier/sample_weight_sum"] == pytest.approx(unique_rows, rel=1e-6)
    for event_uid in np.unique(adjusted.non_tensor_batch["event_uid"]):
        rows = adjusted.non_tensor_batch["event_uid"] == event_uid
        assert adjusted.batch["sample_weight"].cpu().numpy()[rows].sum() == pytest.approx(1.0, rel=1e-6)


def _optimizer_schedule_data(unique_rows=13):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(unique_rows * 3, dtype=torch.long).reshape(unique_rows, 3),
            "attention_mask": torch.tensor(
                [[1, 1, 1] if index % 3 else [1, 1, 0] for index in range(unique_rows)],
                dtype=torch.long,
            ),
            "responses": torch.ones((unique_rows, 2), dtype=torch.long),
            "feature": torch.linspace(-1.0, 1.0, unique_rows, dtype=torch.float64),
            "target": torch.linspace(0.5, -0.5, unique_rows, dtype=torch.float64),
        },
        non_tensors={
            "event_uid": np.array([f"e{index:02d}" for index in range(unique_rows)], dtype=object),
            "wg_id": np.array(["search"] * unique_rows, dtype=object),
            "agent_id": np.array(["Search Agent"] * unique_rows, dtype=object),
        },
    )


def test_uid_coherent_five_mini_schedule_is_rank_aligned_and_duplicate_invariant():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    np.random.seed(9)
    adjusted = adjust_batch(config, _optimizer_schedule_data(), wg_id="search")
    scheduled = prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=17)

    assert scheduled.meta_info["search/optimizer_mini_batch_count"] == 5
    assert scheduled.meta_info["search/optimizer_unique_event_count"] == 13
    assert len(scheduled) == 20
    assert scheduled.meta_info["search/ppo_mini_batch_size"] == 2
    validate_team_event_sample_weights(scheduled, wg_id="search")

    mini_ids = scheduled.batch["optimizer_mini_batch_id"].cpu().numpy()
    uids = scheduled.non_tensor_batch["event_uid"].astype(str)
    uid_to_mini = {}
    for uid_value, mini_id in zip(uids, mini_ids):
        uid_to_mini.setdefault(uid_value, set()).add(int(mini_id))
    assert set(uid_to_mini) == {f"e{index:02d}" for index in range(13)}
    assert all(len(mini_set) == 1 for mini_set in uid_to_mini.values())

    mini_uid_sets = [set(uids[mini_ids == mini_id]) for mini_id in range(5)]
    assert all(mini_uid_sets)
    assert all(
        mini_uid_sets[left].isdisjoint(mini_uid_sets[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )

    for rank_batch in scheduled.chunk(2):
        local_ids = rank_batch.batch["optimizer_mini_batch_id"].cpu().tolist()
        assert local_ids == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_uid_coherent_padding_matches_unique_event_adam_trajectory_for_five_steps():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    source = _optimizer_schedule_data()
    np.random.seed(13)
    adjusted = adjust_batch(config, source, wg_id="search")
    scheduled = prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=23)

    scheduled_ids = scheduled.batch["optimizer_mini_batch_id"].cpu().numpy()
    scheduled_uids = scheduled.non_tensor_batch["event_uid"].astype(str)
    uid_to_mini = {}
    for uid_value, mini_id in zip(scheduled_uids, scheduled_ids):
        uid_to_mini.setdefault(uid_value, int(mini_id))

    source_uids = source.non_tensor_batch["event_uid"].astype(str)
    source_row = {uid_value: row for row, uid_value in enumerate(source_uids)}
    unique_parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    padded_parameter = unique_parameter.detach().clone().requires_grad_(True)
    unique_optimizer = torch.optim.Adam([unique_parameter], lr=0.03)
    padded_optimizer = torch.optim.Adam([padded_parameter], lr=0.03)

    for mini_id in range(5):
        unique_optimizer.zero_grad()
        padded_optimizer.zero_grad()
        mini_uids = sorted(uid for uid, owner in uid_to_mini.items() if owner == mini_id)
        rows = torch.tensor([source_row[uid] for uid in mini_uids], dtype=torch.long)
        unique_loss = torch.mean(
            (unique_parameter * source.batch["feature"][rows] - source.batch["target"][rows]) ** 2
        )

        scheduled_rows = torch.from_numpy(np.flatnonzero(scheduled_ids == mini_id))
        row_losses = (
            padded_parameter * scheduled.batch["feature"][scheduled_rows]
            - scheduled.batch["target"][scheduled_rows]
        ) ** 2
        row_weights = scheduled.batch["sample_weight"][scheduled_rows].to(dtype=torch.float64)
        padded_loss = torch.sum(row_losses * row_weights) / torch.sum(row_weights)

        unique_loss.backward()
        padded_loss.backward()
        assert torch.allclose(unique_loss, padded_loss, atol=1e-10)
        assert torch.allclose(unique_parameter.grad, padded_parameter.grad, atol=1e-10)
        unique_optimizer.step()
        padded_optimizer.step()
        assert torch.allclose(unique_parameter, padded_parameter, atol=1e-10)

    unique_state = unique_optimizer.state[unique_parameter]
    padded_state = padded_optimizer.state[padded_parameter]
    assert unique_state["step"] == padded_state["step"]
    assert torch.allclose(unique_state["exp_avg"], padded_state["exp_avg"], atol=1e-10)
    assert torch.allclose(unique_state["exp_avg_sq"], padded_state["exp_avg_sq"], atol=1e-10)


def test_uid_coherent_schedule_rejects_more_optimizer_minis_than_unique_events():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=1)
    adjusted = adjust_batch(config, _optimizer_schedule_data(unique_rows=4), wg_id="search")
    with pytest.raises(ValueError, match="unique_events=4"):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


def test_uid_coherent_multi_mini_rejects_multiple_ppo_epochs():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    config.actor_rollout_ref.actor.ppo_epochs = 2
    adjusted = adjust_batch(config, _optimizer_schedule_data(), wg_id="search")
    with pytest.raises(ValueError, match="ppo_epochs=1"):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


def test_uid_coherent_schedule_rejects_rows_without_valid_response_tokens():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    source = _optimizer_schedule_data()
    source.batch["attention_mask"][0, -2:] = 0
    adjusted = adjust_batch(config, source, wg_id="search")
    with pytest.raises(ValueError, match="valid response token"):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


def test_uid_coherent_valid_token_preflight_uses_current_wg_multi_turn_mask():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    config.actor_rollout_ref.rollout.multi_turn.enable = True
    source = _optimizer_schedule_data()
    source.batch["loss_mask"] = source.batch["attention_mask"].clone()
    # attention_mask remains valid; only the current WG's actor loss mask is
    # empty, so this fails only if the scheduler honors multi_turn semantics.
    source.batch["loss_mask"][0, -2:] = 0
    adjusted = adjust_batch(config, source, wg_id="search")
    with pytest.raises(ValueError, match="valid response token"):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


def test_uid_coherent_schedule_rejects_multimodal_actor_rows():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    source = _optimizer_schedule_data()
    source.non_tensor_batch["multi_modal_inputs"] = np.array(
        [{"image": object()} for _ in range(len(source))], dtype=object
    )
    adjusted = adjust_batch(config, source, wg_id="search")
    with pytest.raises(ValueError, match="text batches only"):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: setattr(config.actor_rollout_ref.actor, "use_dynamic_bsz", True), "use_dynamic_bsz=False"),
        (
            lambda config: setattr(
                config.actor_rollout_ref.actor, "ulysses_sequence_parallel_size", 2
            ),
            "ulysses_sequence_parallel_size=1",
        ),
    ],
)
def test_uid_coherent_multi_mini_rejects_unsupported_parallel_layouts(mutation, message):
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    mutation(config)
    adjusted = adjust_batch(config, _optimizer_schedule_data(), wg_id="search")
    with pytest.raises(ValueError, match=message):
        prepare_team_event_optimizer_batch(config, adjusted, wg_id="search", seed=0)


def test_uid_coherent_schedule_rejects_actor_world_size_mismatch():
    config = _adjust_config(divisor=2, adaptive=True, updates=5, world_size=2)
    adjusted = adjust_batch(config, _optimizer_schedule_data(), wg_id="search")
    with pytest.raises(ValueError, match="does not match"):
        prepare_team_event_optimizer_batch(
            config,
            adjusted,
            wg_id="search",
            seed=0,
            actor_world_size=4,
        )


def test_combine_batches_preserves_every_workgroup_adaptive_batch_size():
    config = _adjust_config(divisor=4, adaptive=True)
    specifications = {
        "verifier": ("v", "Verifier Agent", 3, 4),
        "search": ("s", "Search Agent", 5, 8),
        "answer": ("a", "Answer Agent", 9, 12),
    }
    batches = {}
    for offset, (wg_id, (prefix, agent_id, unique_rows, adjusted_rows)) in enumerate(specifications.items()):
        batches[wg_id] = DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(
                    offset * 100,
                    offset * 100 + unique_rows * 2,
                    dtype=torch.long,
                ).reshape(unique_rows, 2)
            },
            non_tensors={
                "event_uid": np.array([f"{prefix}{i}" for i in range(unique_rows)], dtype=object),
                "wg_id": np.array([wg_id] * unique_rows, dtype=object),
                "agent_id": np.array([agent_id] * unique_rows, dtype=object),
            },
        )
        batches[wg_id] = adjust_batch(config, batches[wg_id], wg_id=wg_id)
        batches[wg_id].meta_info[f"{wg_id}/global_token_num"] = [offset + 1] * adjusted_rows

    for ordered_ids in (("verifier", "search", "answer"), ("answer", "search", "verifier")):
        combined = combine_batches({wg_id: batches[wg_id] for wg_id in ordered_ids})
        assert len(combined) == sum(spec[3] for spec in specifications.values())
        split_again = split_batch_by_wg_ids(combined, list(specifications))

        for wg_id, (prefix, agent_id, unique_rows, adjusted_rows) in specifications.items():
            assert combined.meta_info[f"{wg_id}/ppo_mini_batch_size"] == adjusted_rows
            assert combined.meta_info[f"{wg_id}/unique_event_count"] == unique_rows
            assert combined.meta_info[f"{wg_id}/adjusted_row_count"] == adjusted_rows
            assert combined.meta_info[f"{wg_id}/sample_weight_sum"] == pytest.approx(unique_rows, rel=1e-6)
            assert combined.meta_info[f"{wg_id}/global_token_num"] == [list(specifications).index(wg_id) + 1] * adjusted_rows

            routed = split_again[wg_id]
            assert len(routed) == adjusted_rows
            assert set(routed.non_tensor_batch["wg_id"]) == {wg_id}
            assert set(routed.non_tensor_batch["agent_id"]) == {agent_id}
            assert all(str(uid).startswith(prefix) for uid in routed.non_tensor_batch["event_uid"])
            assert routed.meta_info[f"{wg_id}/ppo_mini_batch_size"] == adjusted_rows
            for event_uid in np.unique(routed.non_tensor_batch["event_uid"]):
                rows = routed.non_tensor_batch["event_uid"] == event_uid
                assert routed.batch["sample_weight"].cpu().numpy()[rows].sum() == pytest.approx(1.0)


def test_in_reward_kl_uses_multi_turn_mask_and_unique_event_weights():
    class RecordingKLController:
        value = 0.5

        def __init__(self):
            self.last_update = None

        def update(self, current_kl, n_steps):
            self.last_update = (current_kl, n_steps)

    data = DataProto.from_dict(
        tensors={
            "responses": torch.ones((3, 2), dtype=torch.long),
            "attention_mask": torch.ones((3, 3), dtype=torch.long),
            "loss_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.long),
            "token_level_scores": torch.zeros((3, 2), dtype=torch.float32),
            # The second response token is deliberately huge but excluded by
            # loss_mask. Event e0 is a copied pair whose weights sum to one.
            "old_log_probs": torch.tensor([[1.0, 100.0], [1.0, 100.0], [3.0, 100.0]]),
            "ref_log_prob": torch.zeros((3, 2), dtype=torch.float32),
            "sample_weight": torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32),
        },
        non_tensors={"event_uid": np.array(["e0", "e0", "e1"], dtype=object)},
    )
    controller = RecordingKLController()
    adjusted, metrics = apply_kl_penalty(
        data,
        kl_ctrl=controller,
        kl_penalty="kl",
        multi_turn=True,
    )

    assert metrics["actor/reward_kl_penalty"] == pytest.approx(2.0)
    assert controller.last_update == pytest.approx((2.0, 2))
    assert torch.allclose(
        adjusted.batch["token_level_rewards"],
        torch.tensor([[-0.5, 0.0], [-0.5, 0.0], [-1.5, 0.0]]),
    )


def test_invalid_action_ratio_uses_unique_event_weights():
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((3, 1), dtype=torch.long),
            "attention_mask": torch.ones((3, 3), dtype=torch.long),
            "token_level_scores": torch.zeros((3, 2), dtype=torch.float32),
            "sample_weight": torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32),
        },
        non_tensors={
            "event_uid": np.array(["e0", "e0", "e1"], dtype=object),
            "is_action_valid": np.array([True, True, False], dtype=bool),
        },
    )
    adjusted, metrics = apply_invalid_action_penalty(data, invalid_action_penalty_coef=0.1)

    assert metrics["episode/valid_action_ratio"] == pytest.approx(0.5)
    assert torch.allclose(
        adjusted.batch["token_level_scores"],
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, -0.1]]),
    )


def _weight_validation_batch(weights):
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    rows = tensor.shape[0]
    return DataProto.from_dict(
        tensors={"sample_weight": tensor},
        non_tensors={"event_uid": np.array(["e0", "e0", "e1"][:rows], dtype=object)},
    )


def test_team_event_weight_validation_accepts_only_aligned_unit_uid_mass():
    validate_team_event_sample_weights(_weight_validation_batch([0.5, 0.5, 1.0]), wg_id="search")

    missing = DataProto.from_dict(
        tensors={"input_ids": torch.ones((3, 1), dtype=torch.long)},
        non_tensors={"event_uid": np.array(["e0", "e0", "e1"], dtype=object)},
    )
    with pytest.raises(ValueError, match="requires sample_weight"):
        validate_team_event_sample_weights(missing, wg_id="search")
    with pytest.raises(ValueError, match="finite and positive"):
        validate_team_event_sample_weights(_weight_validation_batch([0.5, 0.5, float("nan")]), wg_id="search")
    with pytest.raises(ValueError, match="finite and positive"):
        validate_team_event_sample_weights(_weight_validation_batch([0.5, 0.5, 0.0]), wg_id="search")
    with pytest.raises(ValueError, match="sum to weight 1"):
        validate_team_event_sample_weights(_weight_validation_batch([0.6, 0.5, 1.0]), wg_id="search")
    with pytest.raises(ValueError, match="Misaligned"):
        validate_team_event_sample_weights(
            _weight_validation_batch([[0.5, 0.5], [0.5, 0.5], [1.0, 1.0]]),
            wg_id="search",
        )


def _duplicated_loss_inputs():
    loss_mat = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 6.0],
            [8.0, 0.0, 0.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Event 1 appears three times in the adjusted batch.
    duplicate_indices = torch.tensor([0, 1, 1, 2, 1])
    duplicate_weights = torch.tensor([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0 / 3.0])
    return loss_mat, loss_mask, duplicate_indices, duplicate_weights


def test_duplicate_aware_loss_is_invariant_across_micro_batches_and_dp_ranks():
    loss_mat, loss_mask, duplicate_indices, duplicate_weights = _duplicated_loss_inputs()
    expected = agg_loss(loss_mat, loss_mask, "seq-mean-token-mean")
    duplicated_loss = loss_mat[duplicate_indices]
    duplicated_mask = loss_mask[duplicate_indices]
    global_normalizer = compute_sample_weight_normalizer(
        duplicated_mask,
        duplicate_weights,
        "seq-mean-token-mean",
    )

    # A single adjusted mini-batch is exactly equivalent to one row per event.
    actual = agg_loss(
        duplicated_loss,
        duplicated_mask,
        "seq-mean-token-mean",
        sample_weight=duplicate_weights,
        sample_weight_normalizer=global_normalizer,
    )
    assert torch.allclose(actual, expected, atol=1e-7)

    # Micro-batches contribute numerators against the same full-mini-batch
    # denominator; their backward contributions sum to the same objective.
    micro_slices = (slice(0, 2), slice(2, 5))
    micro_total = sum(
        agg_loss(
            duplicated_loss[micro_slice],
            duplicated_mask[micro_slice],
            "seq-mean-token-mean",
            sample_weight=duplicate_weights[micro_slice],
            sample_weight_normalizer=global_normalizer,
        )
        for micro_slice in micro_slices
    )
    assert torch.allclose(micro_total, expected, atol=1e-7)

    # Simulate two DP ranks. Each rank scales its local contribution by the
    # world size before DDP/FSDP averages gradients across ranks.
    rank_slices = (torch.tensor([0, 2, 4]), torch.tensor([1, 3]))
    rank_losses = []
    for rank_indices in rank_slices:
        rank_losses.append(
            agg_loss(
                duplicated_loss[rank_indices],
                duplicated_mask[rank_indices],
                "seq-mean-token-mean",
                sample_weight=duplicate_weights[rank_indices],
                sample_weight_normalizer=global_normalizer,
                sample_weight_scale=2.0,
            )
        )
    ddp_average = torch.stack(rank_losses).mean()
    assert torch.allclose(ddp_average, expected, atol=1e-7)


def test_duplicate_aware_policy_entropy_and_kl_gradients_are_invariant():
    _, response_mask, duplicate_indices, duplicate_weights = _duplicated_loss_inputs()
    advantages = torch.tensor(
        [
            [0.7, 0.7, 0.0],
            [-0.3, -0.3, -0.3],
            [1.2, 0.0, 0.0],
        ]
    )
    old_log_prob = torch.zeros_like(advantages)
    base_parameter = torch.tensor([0.02, -0.03, 0.01], requires_grad=True)
    duplicate_parameter = base_parameter.detach().clone().requires_grad_(True)

    base_log_prob = base_parameter.unsqueeze(-1).expand_as(advantages)
    duplicate_log_prob = duplicate_parameter[duplicate_indices].unsqueeze(-1).expand(
        len(duplicate_indices),
        advantages.shape[1],
    )
    duplicate_mask = response_mask[duplicate_indices]
    duplicate_advantages = advantages[duplicate_indices]
    normalizer = compute_sample_weight_normalizer(
        duplicate_mask,
        duplicate_weights,
        "seq-mean-token-mean",
    )

    base_policy_outputs = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=base_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="seq-mean-token-mean",
    )
    metric_normalizer = compute_sample_weight_normalizer(
        duplicate_mask,
        duplicate_weights,
        "token-mean",
    )
    duplicate_policy_outputs = compute_policy_loss(
        old_log_prob=old_log_prob[duplicate_indices],
        log_prob=duplicate_log_prob,
        advantages=duplicate_advantages,
        response_mask=duplicate_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="seq-mean-token-mean",
        sample_weight=duplicate_weights,
        sample_weight_normalizer=normalizer,
        sample_weight_metric_normalizer=metric_normalizer,
    )
    base_pg = base_policy_outputs[0]
    duplicate_pg = duplicate_policy_outputs[0]
    base_grad = torch.autograd.grad(base_pg, base_parameter, retain_graph=True)[0]
    duplicate_grad = torch.autograd.grad(duplicate_pg, duplicate_parameter, retain_graph=True)[0]
    for duplicate_value, base_value in zip(duplicate_policy_outputs, base_policy_outputs):
        assert torch.allclose(duplicate_value, base_value, atol=1e-7)
    assert torch.allclose(duplicate_grad, base_grad, atol=1e-7)

    # The actor sums these full-denominator contributions across micro-batches
    # (and explicitly all-reduces them across DP ranks), so metrics are also
    # invariant to micro-batch boundaries.
    micro_outputs = []
    for micro_slice in (slice(0, 2), slice(2, 5)):
        micro_outputs.append(
            compute_policy_loss(
                old_log_prob=old_log_prob[duplicate_indices][micro_slice],
                log_prob=duplicate_log_prob[micro_slice],
                advantages=duplicate_advantages[micro_slice],
                response_mask=duplicate_mask[micro_slice],
                cliprange=0.2,
                cliprange_low=0.2,
                cliprange_high=0.2,
                clip_ratio_c=3.0,
                loss_agg_mode="seq-mean-token-mean",
                sample_weight=duplicate_weights[micro_slice],
                sample_weight_normalizer=normalizer,
                sample_weight_metric_normalizer=metric_normalizer,
                sample_weight_metric_scale=1.0,
            )
        )
    assert torch.allclose(sum(output[0] for output in micro_outputs), base_policy_outputs[0], atol=1e-7)
    for metric_index in (1, 2, 3):
        assert torch.allclose(
            sum(output[metric_index] for output in micro_outputs),
            base_policy_outputs[metric_index],
            atol=1e-7,
        )

    entropy = torch.tensor([[0.4, 0.3, 0.0], [0.8, 0.7, 0.6], [0.2, 0.0, 0.0]])
    kld = torch.tensor([[0.02, 0.03, 0.0], [0.08, 0.07, 0.06], [0.01, 0.0, 0.0]])
    for regularizer_values in (entropy, kld):
        base_regularizer_parameter = torch.tensor([0.01, -0.02, 0.03], requires_grad=True)
        duplicate_regularizer_parameter = base_regularizer_parameter.detach().clone().requires_grad_(True)
        base_regularizer_mat = regularizer_values + base_regularizer_parameter.unsqueeze(-1)
        duplicate_regularizer_mat = (
            regularizer_values[duplicate_indices]
            + duplicate_regularizer_parameter[duplicate_indices].unsqueeze(-1)
        )
        base_regularizer = agg_loss(base_regularizer_mat, response_mask, "seq-mean-token-mean")
        duplicate_regularizer = agg_loss(
            duplicate_regularizer_mat,
            duplicate_mask,
            "seq-mean-token-mean",
            sample_weight=duplicate_weights,
            sample_weight_normalizer=normalizer,
        )
        base_regularizer_grad = torch.autograd.grad(base_regularizer, base_regularizer_parameter)[0]
        duplicate_regularizer_grad = torch.autograd.grad(
            duplicate_regularizer, duplicate_regularizer_parameter
        )[0]
        assert torch.allclose(duplicate_regularizer, base_regularizer, atol=1e-7)
        assert torch.allclose(duplicate_regularizer_grad, base_regularizer_grad, atol=1e-7)
