"""End-to-end CPU tensor tests for Math answer-LOTO team event GAE.

Runs the production ``compute_team_event_gae_advantage`` with the
``math_value`` engine, checking the guide's exact-fraction oracle, the general
GAE recursion, single-pass normalization, duplicate invariance, and the
fail-closed contracts.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_team_event_gae_advantage
from verl.trainer.ppo.team_event_math_value import MathValueConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "math_loto"
MATH_CFG = {"mode": "math_answer_loto"}


def _opaque(values):
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def _fixture_group(name):
    data = json.loads((FIXTURES / "historical_fixtures.json").read_text(encoding="utf-8"))
    return next(f["group"] for f in data if f["name"] == name)


def _math_records(group):
    records = []
    for trajectory in group:
        chain = trajectory["events"]
        role_counters = {}
        for index, source in enumerate(chain):
            terminal = index == len(chain) - 1
            agent = source["agent_id"]
            role_index = role_counters.get(agent, 0)
            role_counters[agent] = role_index + 1
            records.append({
                "row": None,  # assigned below
                "uid": trajectory["uid"],
                "traj": trajectory["traj_uid"],
                "event_uid": f"{trajectory['traj_uid']}:{index}",
                "index": index,
                "count": len(chain),
                "agent": agent,
                "role_index": role_index,
                "role_count": None,  # derived below
                "type": "math_solution" if index % 2 == 0 else "math_verifier",
                "step": 0,
                "owner": terminal,
                "env_owner": agent if terminal else None,
                "reward": float(trajectory["R"]) if terminal else None,
                "done": terminal,
                "state": [{"role": "user", "content": f"{trajectory['traj_uid']}:{index}"}],
                "executed": source["output"],
            })
    counts = {}
    for record in records:
        counts[(record["traj"], record["agent"])] = counts.get((record["traj"], record["agent"]), 0) + 1
    for row, record in enumerate(records):
        record["row"] = row
        record["role_count"] = counts[(record["traj"], record["agent"])]
    return records


def _math_run(records, *, gamma=1.0, lam=1.0, internal_gamma=1.0, internal_lam=1.0,
              normalize=False, width=3, scores=None, rewards=None, config=None,
              use_invalid_action_penalty=True, invalid_action_penalty_coef=0.0,
              include_kl_shaping=False, executed=None, value_config=None, pad_from=None):
    batch_size = len(records)
    token_scores = torch.zeros((batch_size, width), dtype=torch.float32)
    token_rewards = torch.zeros((batch_size, width), dtype=torch.float32)
    if scores is not None:
        token_scores[:, -1] = torch.tensor(scores, dtype=torch.float32)
    if rewards is not None:
        token_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    mask = torch.ones((batch_size, width), dtype=torch.float32)
    if pad_from is not None:
        mask[:, pad_from:] = 0
    executed_values = executed if executed is not None else [record["executed"] for record in records]
    meta = {}
    advantages, returns, diagnostics = compute_team_event_gae_advantage(
        token_level_rewards=token_rewards,
        token_level_scores=token_scores,
        response_mask=mask,
        uid=_opaque([record["uid"] for record in records]),
        traj_index=_opaque([record["traj"] for record in records]),
        event_uid=_opaque([record["event_uid"] for record in records]),
        event_index=_opaque([record["index"] for record in records]),
        event_count=_opaque([record["count"] for record in records]),
        role_event_index=_opaque([record["role_index"] for record in records]),
        role_event_count=_opaque([record["role_count"] for record in records]),
        agent_id=_opaque([record["agent"] for record in records]),
        event_type=_opaque([record["type"] for record in records]),
        env_step_index=_opaque([record["step"] for record in records]),
        is_env_action=_opaque([record["owner"] for record in records]),
        env_action_owner=_opaque([record["env_owner"] for record in records]),
        env_reward=_opaque([record["reward"] for record in records]),
        env_done=_opaque([record["done"] for record in records]),
        state_chats=_opaque([record["state"] for record in records]),
        is_action_valid=_opaque([record.get("valid", True) for record in records]),
        gamma=gamma,
        lam=lam,
        internal_gamma=internal_gamma,
        internal_lam=internal_lam,
        use_invalid_action_penalty=use_invalid_action_penalty,
        invalid_action_penalty_coef=invalid_action_penalty_coef,
        include_kl_shaping=include_kl_shaping,
        normalize_advantages=normalize,
        value_config=value_config,
        math_value_config=config if config is not None else MATH_CFG,
        executed_action_text=_opaque(executed_values),
        diagnostics_meta=meta,
    )
    return advantages, returns, diagnostics, meta


def _by_event_uid(tensor, records):
    return {record["event_uid"]: tensor[record["row"]].item() for record in records}


# ---------------------------------------------------------------------------
# Exact-fraction oracle through the full tensor path
# ---------------------------------------------------------------------------

def test_fraction_oracle_raw_gae_identity_float32():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    advantages, returns, diagnostics, meta = _math_run(records)
    target_uid = "f8c20da0-b915-46cf-8a2a-fbdcddffcd07"
    expected_raw = [1 / 7, 4 / 7, 4 / 7, 1 / 49, 1 / 49]
    expected_pre = [6 / 7, 3 / 7, 3 / 7, 48 / 49, 48 / 49]
    raw = {record["event_uid"]: diagnostics["event_raw_advantages"][index].item()
           for index, record in enumerate(records)}
    pre = {record["event_uid"]: diagnostics["event_math_value_pre"][index].item()
           for index, record in enumerate(records)}
    post = {record["event_uid"]: diagnostics["event_math_value_post"][index].item()
            for index, record in enumerate(records)}
    for index in range(5):
        uid = f"{target_uid}:{index}"
        assert abs(raw[uid] - expected_raw[index]) <= 1e-6
        assert abs(pre[uid] - expected_pre[index]) <= 1e-6
    assert abs(post[f"{target_uid}:4"] - 0.0) <= 1e-6
    # env-only GAE maps: task raw equals actor raw with aux disabled.
    task_raw = diagnostics["event_math_task_raw_advantages"]
    actor_raw = diagnostics["event_math_raw_advantages"]
    assert torch.allclose(task_raw, actor_raw, atol=1e-7)
    # metadata carries the credit-trace schema
    assert meta["schema_version"] == "team_event_math_credit_v1"
    assert meta["details_by_event"][f"{target_uid}:0"]["parser_status"] == "ok"


def test_lambda_one_telescoping_for_every_trajectory():
    for name in ("all_failure", "all_success", "one_quarter_correction", "parser_last_box_failure",
                 "sole_success", "real_early_stop"):
        group = _fixture_group(name)
        records = _math_records(group)
        advantages, returns, diagnostics, meta = _math_run(records)
        rewards_by_traj = {}
        for record in records:
            if record["done"]:
                rewards_by_traj[record["traj"]] = record["reward"]
        for record in records:
            raw = diagnostics["event_raw_advantages"][record["row"]].item() if "row" in record else None
        raw_by_uid = {record["event_uid"]: diagnostics["event_raw_advantages"][index].item()
                      for index, record in enumerate(records)}
        pre_by_uid = {record["event_uid"]: diagnostics["event_math_value_pre"][index].item()
                      for index, record in enumerate(records)}
        for record in records:
            total = rewards_by_traj[record["traj"]]
            expected = total - pre_by_uid[record["event_uid"]]
            assert abs(raw_by_uid[record["event_uid"]] - expected) <= 1e-6


def test_general_lambda_recursion_is_preserved():
    # internal_lam=0.5 only affects internal (non-terminal) edges.  The
    # recursion A_e = delta_e + gamma*lam*A_{e+1} must hold exactly.
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    advantages, returns, diagnostics, meta = _math_run(records, internal_lam=0.5)
    gamma_map = {record["event_uid"]: (0.0 if record["done"] else 1.0) for record in records}
    lam_map = {record["event_uid"]: (0.0 if record["done"] else 0.5) for record in records}
    pre = {record["event_uid"]: diagnostics["event_math_value_pre"][index].item()
           for index, record in enumerate(records)}
    post = {record["event_uid"]: diagnostics["event_math_value_post"][index].item()
            for index, record in enumerate(records)}
    env_reward = {record["event_uid"]: (record["reward"] or 0.0) for record in records}
    raw = {record["event_uid"]: diagnostics["event_raw_advantages"][index].item()
           for index, record in enumerate(records)}
    by_traj = {}
    for record in records:
        by_traj.setdefault(record["traj"], []).append(record)
    for trajectory_records in by_traj.values():
        trajectory_records.sort(key=lambda record: record["index"])
        next_advantage = 0.0
        for record in reversed(trajectory_records):
            uid = record["event_uid"]
            following = None
            position = trajectory_records.index(record)
            if position + 1 < len(trajectory_records):
                following = trajectory_records[position + 1]["event_uid"]
            next_value = post[uid] if following is None else pre[following]
            delta = env_reward[uid] + gamma_map[uid] * next_value - pre[uid]
            expected = delta + gamma_map[uid] * lam_map[uid] * next_advantage
            assert abs(raw[uid] - expected) <= 1e-6, uid
            next_advantage = expected


def test_normalization_once_per_role_with_equal_trajectory_weights():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    for index, record in enumerate(records):
        record["row"] = index
    advantages, returns, diagnostics, meta = _math_run(records, normalize=True)
    raw = _by_event_uid(diagnostics["event_raw_advantages"], records)
    # Recompute the single-pass reference normalization by hand.
    groups = {}
    for record in records:
        groups.setdefault((record["uid"], record["agent"]), []).append(record)
    normalized = {}
    for cohort in groups.values():
        per_trajectory = {}
        for record in cohort:
            per_trajectory.setdefault(record["traj"], []).append(record)
        if len(per_trajectory) < 2:
            for record in cohort:
                normalized[record["event_uid"]] = raw[record["event_uid"]]
            continue
        mean = sum(
            sum(raw[r["event_uid"]] for r in traj_records) / len(traj_records)
            for traj_records in per_trajectory.values()
        ) / len(per_trajectory)
        variance = sum(
            sum((raw[r["event_uid"]] - mean) ** 2 for r in traj_records) / len(traj_records)
            for traj_records in per_trajectory.values()
        ) / len(per_trajectory)
        std = math.sqrt(variance)
        for traj_records in per_trajectory.values():
            for record in traj_records:
                normalized[record["event_uid"]] = (
                    0.0 if std <= 1e-6 else (raw[record["event_uid"]] - mean) / (std + 1e-6)
                )
    got = {record["event_uid"]: advantages[record["row"], 0].item() for record in records}
    for uid, expected in normalized.items():
        assert abs(got[uid] - expected) <= 1e-5, uid
    # event_math advantages diagnostic equals the actor tensor
    got_math = {record["event_uid"]: diagnostics["event_math_advantages"][record["row"]].item()
                for record in records}
    for uid, expected in got.items():
        assert abs(got_math[uid] - expected) <= 1e-6


def test_all_equal_returns_produce_zero_task_advantages():
    group = _fixture_group("all_success")
    records = _math_records(group)
    advantages, _, diagnostics, _ = _math_run(records, normalize=True)
    for index in range(len(records)):
        assert diagnostics["event_raw_advantages"][index].item() == pytest.approx(0.0, abs=1e-6)
        assert advantages[index, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_duplicates_and_reordering_do_not_change_unique_event_values():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    baseline_adv, _, baseline_diag, _ = _math_run(records, normalize=True)
    duplicated = records + records[::-1]
    dup_adv, _, dup_diag, _ = _math_run(duplicated, normalize=True)
    lookup = {record["event_uid"]: index for index, record in enumerate(records)}
    for index, record in enumerate(duplicated):
        base = lookup[record["event_uid"]]
        assert abs(dup_adv[index, 0].item() - baseline_adv[base, 0].item()) <= 1e-6


def test_padding_tokens_carry_zero_advantage():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    # width=6 with a mask that blanks the last three tokens of every row
    advantages, _, _, _ = _math_run(records, normalize=True, width=6, pad_from=3)
    assert torch.all(advantages[:, 3:] == 0)


def test_missing_executed_text_is_rejected():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    executed = [record["executed"] for record in records]
    executed[0] = None
    with pytest.raises(ValueError, match="executed_action_text"):
        _math_run(records, executed=executed)


def test_conflicting_copied_executed_text_is_rejected():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    conflicting = list(records)
    clone = dict(records[0])
    clone["executed"] = records[0]["executed"] + " altered"
    conflicting.append(clone)
    with pytest.raises(ValueError, match="Copied rows disagree"):
        _math_run(conflicting)


def test_env_only_rejects_kl_shaping_and_invalid_penalty():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    width = 3
    scores = [0.0] * len(records)
    rewards_column = [1.0] * len(records)  # nonzero reward-KL difference
    with pytest.raises(ValueError, match="reward-KL shaping"):
        _math_run(records, scores=scores, rewards=rewards_column, include_kl_shaping=True)
    with pytest.raises(ValueError, match="invalid action penalty"):
        _math_run(records, use_invalid_action_penalty=True, invalid_action_penalty_coef=0.1)


def test_runtime_rejects_discounted_math_profile():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    with pytest.raises(ValueError, match="gamma=internal_gamma=1"):
        _math_run(records, gamma=0.9)


def test_engine_conflict_between_math_and_boundary_is_rejected():
    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _math_run(records, value_config={"mode": "boundary_loto"})


# ---------------------------------------------------------------------------
# Credit records
# ---------------------------------------------------------------------------

class _FakeNT(dict):
    pass


def test_math_credit_records_roundtrip(tmp_path):
    from agent_system.team_event_credit_trace import (
        build_math_run_manifest,
        dump_math_credit_trace,
    )

    group = _fixture_group("one_quarter_correction")
    records = _math_records(group)
    for index, record in enumerate(records):
        record["row"] = index
    advantages, returns, diagnostics, meta = _math_run(records, normalize=True)

    config = {
        "algorithm": {
            "adv_estimator": "team_event_gae",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
            "team_event_gae": {
                "internal_gamma": 1.0,
                "internal_lam": 1.0,
                "include_kl_shaping": False,
                "math_value": dict(MATH_CFG),
                "credit_trace": {"enabled": True, "every_n_steps": 1},
            },
        },
        "agent": {"orchestra_type": "math", "orchestra": {"math": {"max_loop_num": 3}}},
        "env": {"env_name": "math", "rollout": {"n": 8}},
        "actor_rollout_ref": {"rollout": {"n": 1}, "actor": {}},
        "trainer": {"val_only": False},
    }
    manifest = build_math_run_manifest(config, run_id="run-1")
    assert manifest["schema_version"] == "team_event_math_manifest_v1"

    batch = type("Batch", (), {})()
    batch.batch = dict(diagnostics)
    batch.batch["advantages"] = advantages
    batch.batch["response_mask"] = torch.ones((len(records), 3))
    batch.non_tensor_batch = {
        "event_uid": _opaque([r["event_uid"] for r in records]),
        "uid": _opaque([r["uid"] for r in records]),
        "traj_uid": _opaque([r["traj"] for r in records]),
        "event_index": _opaque([r["index"] for r in records]),
        "role_event_index": _opaque([r["role_index"] for r in records]),
        "env_step_index": _opaque([r["step"] for r in records]),
        "agent_id": _opaque([r["agent"] for r in records]),
        "event_type": _opaque([r["type"] for r in records]),
        "wg_id": _opaque(["wg_solver" if r["agent"] == "Solver Agent" else "wg_verifier" for r in records]),
        "loop_index": _opaque([r["index"] // 2 for r in records]),
        "verifier_decision": _opaque(["not_applicable" if r["index"] % 2 == 0 else "reject" for r in records]),
        "submitted_solution_event_uid": _opaque([f"{r['traj']}:0" for r in records]),
        "transition_owner_event_uid": _opaque([f"{r['traj']}:{r['count']-1}" for r in records]),
        "orchestration_stop_reason": _opaque(["max_loop_exhausted"] * len(records)),
        "env_done": _opaque([r["done"] for r in records]),
    }
    batch.meta_info = {"math_answer_loto": meta}
    path, count, _ = dump_math_credit_trace(batch, str(tmp_path), manifest, global_step=1)
    assert count == len(records)
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(records)
    sample = next(line for line in lines if line["event_uid"] == "f8c20da0-b915-46cf-8a2a-fbdcddffcd07:0")
    for field in ("value_pre", "value_post", "question_parent", "same_round_parent", "kernel_mass",
                  "effective_ess", "task_raw_advantage", "actor_advantage", "parser_status",
                  "branch", "answer_key_sha256", "submitted_solution_event_uid",
                  "transition_owner_event_uid"):
        assert field in sample
    assert sample["question_parent"] == pytest.approx(6 / 7, abs=1e-6)


# ---------------------------------------------------------------------------
# Startup config contract
# ---------------------------------------------------------------------------

def _training_config(**overrides):
    config = {
        "algorithm": {
            "adv_estimator": "team_event_gae",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
            "team_event_gae": {
                "internal_gamma": 1.0,
                "internal_lam": 1.0,
                "include_kl_shaping": False,
                "normalize_advantages": True,
                "math_value": dict(MATH_CFG),
                "value": {"mode": "legacy_loto"},
                "credit_trace": {"enabled": True, "every_n_steps": 1},
            },
        },
        "agent": {"orchestra_type": "math", "orchestra": {"math": {"max_loop_num": 3}}},
        "env": {"env_name": "math", "rollout": {"n": 8}},
        "actor_rollout_ref": {"rollout": {"n": 1}, "actor": {"use_invalid_action_penalty": False}},
        "trainer": {"val_only": False},
    }
    config.update(overrides)
    return config


def test_validate_math_training_config_accepts_the_run_contract():
    from agent_system.team_event_credit_trace import validate_math_training_config

    value = validate_math_training_config(_training_config())
    assert value is not None and value.mode == "math_answer_loto"


def test_validate_math_training_config_rejects_misconfigurations():
    from agent_system.team_event_credit_trace import validate_math_training_config

    bad_n = _training_config()
    bad_n["env"]["rollout"]["n"] = 4
    with pytest.raises(ValueError, match="env.rollout.n"):
        validate_math_training_config(bad_n)
    bad_loop = _training_config()
    bad_loop["agent"]["orchestra"]["math"]["max_loop_num"] = 2
    with pytest.raises(ValueError, match="max_loop_num"):
        validate_math_training_config(bad_loop)
    conflict = _training_config()
    conflict["algorithm"]["team_event_gae"]["value"]["mode"] = "boundary_loto"
    with pytest.raises(ValueError, match="conflicts"):
        validate_math_training_config(conflict)
    penalty = _training_config()
    penalty["actor_rollout_ref"]["actor"]["use_invalid_action_penalty"] = True
    penalty["actor_rollout_ref"]["actor"]["invalid_action_penalty_coef"] = 0.1
    with pytest.raises(ValueError, match="invalid-action penalty"):
        validate_math_training_config(penalty)
    kl = _training_config()
    kl["algorithm"]["use_kl_in_reward"] = True
    kl["algorithm"]["team_event_gae"]["include_kl_shaping"] = True
    with pytest.raises(ValueError, match="reward-KL"):
        validate_math_training_config(kl)
    orchestra = _training_config()
    orchestra["agent"]["orchestra_type"] = "search"
    with pytest.raises(ValueError, match="Math orchestra"):
        validate_math_training_config(orchestra)
    off = _training_config()
    off["algorithm"]["team_event_gae"]["math_value"]["mode"] = "off"
    assert validate_math_training_config(off) is None


def test_val_only_allows_rollout_n_one():
    from agent_system.team_event_credit_trace import validate_math_training_config

    config = _training_config()
    config["trainer"]["val_only"] = True
    config["env"]["rollout"]["n"] = 1
    assert validate_math_training_config(config) is None


def test_checkpoint_manifest_roundtrip_and_incompatibility(tmp_path):
    from agent_system.team_event_credit_trace import (
        build_math_run_manifest,
        check_team_event_checkpoint_manifest,
        save_team_event_run_manifest,
    )

    manifest = build_math_run_manifest(_training_config(), run_id="run-1")
    checkpoint = tmp_path / "global_step_1"
    checkpoint.mkdir()
    save_team_event_run_manifest(manifest, str(checkpoint))
    check_team_event_checkpoint_manifest(manifest, str(checkpoint))
    changed = build_math_run_manifest(_training_config(), run_id="run-1")
    changed["compatibility"]["value"]["beta"] = 2.0
    with pytest.raises(ValueError, match="differs"):
        check_team_event_checkpoint_manifest(changed, str(checkpoint))
    with pytest.raises(ValueError, match="requires the checkpoint's"):
        check_team_event_checkpoint_manifest(manifest, str(tmp_path / "missing"))
