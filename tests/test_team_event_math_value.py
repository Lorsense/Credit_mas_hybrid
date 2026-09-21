"""CPU tests for the Math answer-LOTO value engine (task_event_math_value).

Pure float64 checks against the audited reference semantics and the frozen
golden fixtures; no Ray/DataProto involved.
"""

import json
import math
from pathlib import Path

import pytest

from verl.trainer.ppo.team_event_math_value import (
    MATH_VALUE_PARSER_VERSION,
    BoundedAnswerCache,
    MathValueConfig,
    allowed_math_chains,
    canonical_answer_key,
    estimate_math_values,
    extract_latest_box,
    parse_answer_feature,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "math_loto"
CFG = MathValueConfig(mode="math_answer_loto")


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_status", [
    (r"the answer is \boxed{42}", "ok"),
    (r"\fbox{9}", "ok"),
    (r"\boxed{a} then \boxed{b}", "ok"),           # latest box wins
    (r"\boxed{1+\frac{1}{2}}", "ok"),               # nested braces
    (r"\boxed{\frac{1}{2}}", "ok"),
    (r"\boxed{unclosed \{ thing", "unclosed_last_box"),
    (r"\boxed{42", "unclosed_last_box"),
    (r"\boxed{}", "empty_box"),
    (r"\boxed{   }", "empty_box"),
    (r"no box at all", "no_box"),
    (r"\boxedSomething{42}", "no_box"),             # \boxedSomething is not a command
    (r"\boxed{a} \boxed{b", "unclosed_last_box"),   # last box unclosed: no fallback
])
def test_latest_box_statuses(text, expected_status):
    assert extract_latest_box(text)["status"] == expected_status


def test_escaped_braces_do_not_close():
    assert extract_latest_box(r"\boxed{a\}b}")["raw"] == "a\\}b"
    assert extract_latest_box(r"\boxed{\\}")["status"] == "ok"


def test_literal_key_preserves_tokens_and_boundaries():
    assert canonical_answer_key(r"-1") != canonical_answer_key("1")
    assert canonical_answer_key("A") != canonical_answer_key("a")
    assert canonical_answer_key("12") != canonical_answer_key("1 2")
    assert canonical_answer_key(r"3 \text{cm}") != canonical_answer_key("3")
    assert canonical_answer_key(r"\dfrac{1}{2}") == canonical_answer_key(r"\tfrac{1}{2}")
    assert canonical_answer_key(r"\sin x") != canonical_answer_key(r"\sinx")
    assert canonical_answer_key("") == ""


def test_display_commands_are_dropped_but_content_kept():
    assert canonical_answer_key(r"\left( x \right)") == canonical_answer_key("(x)")
    # \text{...} content is preserved as one text token.
    assert "math" in canonical_answer_key(r"\text{math}")


def test_parse_answer_feature_shapes():
    feature = parse_answer_feature(r"x = 3, so \boxed{\frac{1}{2}}")
    assert feature["status"] == "ok"
    assert feature["commands"] == 1
    assert feature["key"].startswith("literal:")
    assert parse_answer_feature("no box")["key"] == ""


def test_bounded_cache_never_outgrows_limit():
    cache = BoundedAnswerCache(size=4)
    for i in range(20):
        cache.feature(f"\\boxed{{{i}}}")
    assert len(cache._cache) <= 4


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_keys_and_wrong_types():
    with pytest.raises(ValueError, match="Unknown"):
        MathValueConfig.from_mapping({"mode": "math_answer_loto", "beta_x": 1.0})
    with pytest.raises(ValueError, match="boolean"):
        MathValueConfig(mode="math_answer_loto", strict="yes")
    with pytest.raises(ValueError, match="Unsupported math_value.mode"):
        MathValueConfig(mode="q_only")
    with pytest.raises(ValueError, match="max_loop_num=3"):
        MathValueConfig(mode="math_answer_loto", expected_max_loop_num=2)
    with pytest.raises(ValueError, match="expected_rollout_n>=2"):
        MathValueConfig(mode="math_answer_loto", expected_rollout_n=1)
    with pytest.raises(ValueError, match="finite"):
        MathValueConfig(mode="math_answer_loto", beta=float("nan"))


def test_config_runtime_requires_undiscounted_profile():
    CFG.validate_runtime(gamma=1.0, internal_gamma=1.0)
    with pytest.raises(ValueError, match="gamma=internal_gamma=1"):
        CFG.validate_runtime(gamma=0.9, internal_gamma=1.0)
    with pytest.raises(ValueError, match="agent_local.mode=off"):
        CFG.validate_runtime(gamma=1.0, internal_gamma=1.0, agent_local_mode="blend")


def test_allowed_chains_for_three_loops():
    assert set(allowed_math_chains(3)) == {
        ("math_solution", "math_verifier"),
        ("math_solution", "math_verifier", "math_solution", "math_verifier"),
        ("math_solution", "math_verifier", "math_solution", "math_verifier", "math_solution"),
    }


# ---------------------------------------------------------------------------
# Fixture conversion and golden alignment
# ---------------------------------------------------------------------------

def _convert(group):
    events, trajectories = {}, {}
    for trajectory in group:
        keys = []
        chain = trajectory["events"]
        for index, source in enumerate(chain):
            terminal = index == len(chain) - 1
            uid = f"{trajectory['traj_uid']}:{index}"
            events[uid] = {
                "event_uid": uid,
                "uid": trajectory["uid"],
                "traj_uid": trajectory["traj_uid"],
                "event_index": index,
                "event_count": len(chain),
                "role_event_index": source["role_turn_index"],
                "role_event_count": sum(1 for item in chain if item["agent_id"] == source["agent_id"]),
                "agent_id": source["agent_id"],
                "event_type": "math_solution" if source["pos"].startswith("S") else "math_verifier",
                "env_step_index": 0,
                "is_env_action": terminal,
                "env_action_owner": source["agent_id"] if terminal else None,
                "env_reward": float(trajectory["R"]) if terminal else 0.0,
                "env_done": terminal,
                "is_action_valid": True,
                "executed_action_text": source["output"],
                "task_type": "math",
            }
            keys.append(uid)
        trajectories[trajectory["traj_uid"]] = keys
    return events, trajectories


def _load_fixtures():
    data = json.loads((FIXTURES / "historical_fixtures.json").read_text(encoding="utf-8"))
    golden = json.loads((FIXTURES / "golden_values.json").read_text(encoding="utf-8"))
    return data, {entry["name"]: entry for entry in golden["groups"]}


def test_golden_values_match_production_engine_float64():
    fixtures, golden = _load_fixtures()
    worst = 0.0
    for fixture in fixtures:
        events, trajectories = _convert(fixture["group"])
        result = estimate_math_values(events, trajectories, CFG)
        case = golden[fixture["name"]]["cases"]["math_answer_loto:1.0"]
        for entry in case:
            for index in range(len(entry["pre"])):
                uid = f"{entry['traj_uid']}:{index}"
                worst = max(worst, abs(result.pre[uid] - entry["pre"][index]))
                worst = max(worst, abs(result.post[uid] - entry["post"][index]))
    assert worst <= 1e-10, f"float64 mismatch {worst}"


def test_manual_fraction_oracle_group():
    fixtures, _ = _load_fixtures()
    group = next(f["group"] for f in fixtures if f["name"] == "one_quarter_correction")
    target = next(t for t in group if t["traj_uid"] == "f8c20da0-b915-46cf-8a2a-fbdcddffcd07")
    events, trajectories = _convert(group)
    result = estimate_math_values(events, trajectories, CFG)
    keys = trajectories[target["traj_uid"]]
    expected_pre = [6 / 7, 3 / 7, 3 / 7, 48 / 49, 48 / 49]
    expected_post = [3 / 7, 3 / 7, 48 / 49, 48 / 49, 0.0]
    for index, key in enumerate(keys):
        assert abs(result.pre[key] - expected_pre[index]) <= 1e-10
        assert abs(result.post[key] - expected_post[index]) <= 1e-10


def test_boundary_identity_and_initial_question_mean():
    fixtures, _ = _load_fixtures()
    for fixture in fixtures:
        events, trajectories = _convert(fixture["group"])
        result = estimate_math_values(events, trajectories, CFG)
        rewards = result.returns
        for trajectory, keys in trajectories.items():
            peers = [other for other in trajectories if other != trajectory]
            q_mean = sum(rewards[other] for other in peers) / len(peers)
            assert abs(result.pre[keys[0]] - q_mean) <= 1e-12
            for current, following in zip(keys, keys[1:]):
                assert result.pre[following] == result.post[current]
            assert result.post[keys[-1]] == 0.0


def test_all_equal_returns_produce_zero_raw_signals():
    fixtures, _ = _load_fixtures()
    for name, level in (("all_failure", 0.0), ("all_success", 1.0)):
        fixture = next(f for f in fixtures if f["name"] == name)
        events, trajectories = _convert(fixture["group"])
        result = estimate_math_values(events, trajectories, CFG)
        for key, value in result.pre.items():
            assert value == pytest.approx(level, abs=1e-12)
        # raw GAE A_e = R - pre is identically zero for every event.
        for trajectory, keys in trajectories.items():
            reward = result.returns[trajectory]
            for key in keys:
                assert reward - result.pre[key] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# LOTO time-position and peer semantics (synthetic groups)
# ---------------------------------------------------------------------------

def _synthetic_group(answers, rewards, chains=None):
    """answers: per trajectory, per solver round answer letter ('' = none)."""

    chains = chains or [(0, 1, 2)] * len(rewards)
    events, trajectories = {}, {}
    question = "synthetic audit question"
    for index, reward in enumerate(rewards):
        chain = []
        for solver_round in chains[index]:
            chain.append(("math_solution", solver_round))
            if solver_round < 2:
                chain.append(("math_verifier", solver_round))
        keys = []
        for position, (kind, solver_round) in enumerate(chain):
            terminal = position == len(chain) - 1
            uid = f"t{index}:{position}"
            role_count = sum(1 for item in chain if item[0] == kind)
            events[uid] = {
                "event_uid": uid,
                "uid": "question",
                "traj_uid": f"t{index}",
                "event_index": position,
                "event_count": len(chain),
                "role_event_index": solver_round,
                "role_event_count": role_count,
                "agent_id": "Solver Agent" if kind == "math_solution" else "Verifier Agent",
                "event_type": kind,
                "env_step_index": 0,
                "is_env_action": terminal,
                "env_action_owner": ("Solver Agent" if kind == "math_solution" else "Verifier Agent") if terminal else None,
                "env_reward": float(reward) if terminal else 0.0,
                "env_done": terminal,
                "is_action_valid": True,
                "executed_action_text": (
                    f"\\boxed{{{answers[index][solver_round]}}}" if kind == "math_solution" else "reject"
                ),
                "task_type": "math",
            }
            keys.append(uid)
        trajectories[f"t{index}"] = keys
    return events, trajectories


def test_own_reward_never_enters_own_value():
    events, trajectories = _synthetic_group(
        answers=[["a", "b", "c"], ["b", "b", "b"], ["c", "c", "c"], ["b", "b", "b"],
                 ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]],
        rewards=[1, 0, 0, 0, 0, 1, 1, 1],
    )
    before = estimate_math_values(events, trajectories, CFG)
    flipped = _synthetic_group(
        answers=[["a", "b", "c"], ["b", "b", "b"], ["c", "c", "c"], ["b", "b", "b"],
                 ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]],
        rewards=[0, 0, 0, 0, 0, 1, 1, 1],
    )
    after = estimate_math_values(flipped[0], flipped[1], CFG)
    for key in events:
        if key.startswith("t0:"):
            # Only trajectories sharing t0's group change peers; t0's own flip
            # must not move any of its own pre/post values.
            assert before.pre[key] == after.pre[key]
            assert before.post[key] == after.post[key]


def test_current_answer_only_affects_post_and_later():
    # Peers with the same S0 answer succeed; the others mostly fail, so a
    # same-round match genuinely moves the S0 post away from the parent.
    peer_answers = [["b", "b", "b"], ["a", "b", "b"], ["a", "b", "b"], ["b", "b", "b"],
                    ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]]
    rewards = [1, 1, 1, 0, 0, 1, 0, 0]
    matched = [["a"] + peer_answers[0][1:]] + peer_answers[1:]
    unmatched = [["z"] + peer_answers[0][1:]] + peer_answers[1:]
    events, trajectories = _synthetic_group(matched, rewards)
    before = estimate_math_values(events, trajectories, CFG)
    events2, trajectories2 = _synthetic_group(unmatched, rewards)
    after = estimate_math_values(events2, trajectories2, CFG)
    keys = trajectories["t0"]
    assert before.pre[keys[0]] == after.pre[keys[0]]        # pre uses only prior state
    # matched post = (2*1 + P)/3 with P = 3/7; unmatched post = P.
    assert before.post[keys[0]] == pytest.approx((2 + 3 / 7) / 3)
    assert after.post[keys[0]] == pytest.approx(3 / 7)
    # The carried Verifier pre inherits the changed S0 post; the next Solver
    # pre is governed by its own answer and need not stay different.
    assert before.pre[keys[1]] == pytest.approx((2 + 3 / 7) / 3)
    assert after.pre[keys[1]] == pytest.approx(3 / 7)


def test_future_own_output_does_not_change_past_pre():
    rewards = [1, 0, 1, 0, 1, 0, 1, 0]
    events, trajectories = _synthetic_group([["b", "b", "b"]] * 8, rewards)
    before = estimate_math_values(events, trajectories, CFG)
    events2, trajectories2 = _synthetic_group([["b", "b", "x"]] * 8, rewards)
    after = estimate_math_values(events2, trajectories2, CFG)
    for trajectory in trajectories:
        keys = trajectories[trajectory]
        for key in keys[:4]:  # S0..V1 pre must not depend on the future S2 text
            assert before.pre[key] == after.pre[key]


def test_terminal_solver_answer_is_not_peer_state():
    rewards = [1, 0, 1, 0, 1, 0, 1, 0]
    answers = [["b", "b", "b"]] * 8
    events, trajectories = _synthetic_group(answers, rewards)
    before = estimate_math_values(events, trajectories, CFG)
    rewritten = [list(pair) for pair in answers]
    for pair in rewritten:
        pair[2] = "a"  # every terminal S2 changes
    events2, trajectories2 = _synthetic_group(rewritten, rewards)
    after = estimate_math_values(events2, trajectories2, CFG)
    assert before.pre == after.pre
    assert before.post == after.post


def test_whole_query_same_round_gating():
    # t0 S0 answer 'a': only one same-round match (t1). t2 matches 'a' only at
    # S1. The whole query uses only the same-round peer; t2 never contributes.
    answers = [["a", "b", "b"], ["a", "c", "c"], ["b", "a", "a"], ["b", "b", "b"],
               ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]]
    rewards = [0, 1, 1, 1, 0, 1, 0, 1]
    events, trajectories = _synthetic_group(answers, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    key = trajectories["t0"][0]
    q = sum(rewards[1:]) / 7
    # C is existence-based: all seven peers reached a non-terminal round 0.
    parent = (sum(rewards[1:]) + 2 * q) / (7 + 2)
    expected = (1 * 1 + 1 * parent) / (1 + 1)
    assert abs(result.post[key] - expected) <= 1e-12
    detail = result.details[key]
    assert detail["branch"] == "same"
    assert detail["used_peer_count"] == 1
    assert detail["cross_round_peer_count"] == 0
    assert detail["kernel_mass"] == pytest.approx(1.0)


def test_cross_round_fallback_only_when_no_same_round_match():
    answers = [["a", "b", "b"], ["c", "a", "c"], ["b", "b", "b"], ["b", "b", "b"],
               ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]]
    rewards = [0, 1, 1, 1, 0, 1, 0, 1]
    events, trajectories = _synthetic_group(answers, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    key = trajectories["t0"][0]
    detail = result.details[key]
    assert detail["branch"] == "cross"
    # t1 matches 'a' only at round 1 → weight 0.5, never S0+S1 combined.
    assert detail["kernel_mass"] == pytest.approx(0.5)
    assert detail["used_peer_count"] == 1


def test_single_direct_peer_effective_coefficients_and_ess():
    answers = [["a", "b", "b"], ["a", "c", "c"], ["b", "b", "b"], ["b", "b", "b"],
               ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]]
    rewards = [0, 1, 1, 1, 1, 1, 1, 1]  # all seven peers reached round 0
    events, trajectories = _synthetic_group(answers, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    detail = result.details[trajectories["t0"][0]]
    assert detail["max_effective_coefficient"] == pytest.approx(4 / 7)
    assert detail["effective_ess"] == pytest.approx(2.8)
    assert detail["kernel_ess"] == pytest.approx(1.0)


def test_no_match_falls_back_to_parent_and_m0_to_question_mean():
    answers = [["a", "b", "b"], ["c", "d", "e"], ["f", "g", "h"], ["i", "j", "k"],
               ["l", "m", "n"], ["o", "p", "q"], ["r", "s", "t"], ["u", "v", "w"]]
    rewards = [1, 1, 0, 1, 0, 0, 1, 0]
    events, trajectories = _synthetic_group(answers, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    key = trajectories["t0"][0]
    detail = result.details[key]
    assert detail["branch"] == "parent"
    q = sum(rewards[1:]) / 7
    assert result.post[key] == pytest.approx(q)
    # Round 2 (S2) never has non-terminal peers: m=0 → parent = question mean.
    s2_key = trajectories["t0"][4]
    assert result.details[s2_key]["parent"] == pytest.approx(q)


def test_sole_success_is_not_subdivided():
    rewards = [0, 0, 1, 0, 0, 0, 0, 0]
    events, trajectories = _synthetic_group([["b", "b", "b"]] * 8, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    winner = trajectories["t2"]
    loser = trajectories["t0"]
    for key in winner:
        if result.details[key]["branch"] != "terminal":
            assert result.pre[key] == 0.0    # peers all failed
    for key in loser:
        if result.details[key]["branch"] not in ("terminal",):
            assert 0.0 < result.pre[key] <= 1.0 / 7 + 1e-12


def test_peer_contributes_at_most_one_maximal_weight():
    # t1 has the same answer at S0 and S1; it must still contribute one weight.
    answers = [["a", "z", "z"], ["a", "a", "z"], ["b", "b", "b"], ["b", "b", "b"],
               ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"]]
    rewards = [0, 1, 0, 0, 0, 0, 0, 0]
    events, trajectories = _synthetic_group(answers, rewards)
    result = estimate_math_values(events, trajectories, CFG)
    detail = result.details[trajectories["t0"][0]]
    assert detail["kernel_mass"] == pytest.approx(1.0)
    assert detail["used_peer_count"] == 1


def test_early_stop_round_parent_uses_arrived_set_only():
    # t7 stops after V0: its S0 exists (non-terminal) but its S1/S2 do not.
    answers = [["a", "b", "b"], ["a", "b", "b"], ["b", "b", "b"], ["b", "b", "b"],
               ["b", "b", "b"], ["b", "b", "b"], ["b", "b", "b"], ["a", None, None]]
    rewards = [1, 0, 1, 0, 1, 0, 1, 0]
    chains = [(0, 1, 2)] * 7 + [(0,)]
    events, trajectories = _synthetic_group(answers, rewards, chains=chains)
    result = estimate_math_values(events, trajectories, CFG)
    s1_key = trajectories["t0"][2]
    detail = result.details[s1_key]
    # Only trajectories 1..6 arrived at round 1: m = 6 of 7 peers.
    assert detail["same_round_peer_count"] == 6


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

def _valid_events():
    return _synthetic_group([["b", "b", "b"]] * 8, [1, 0, 1, 0, 1, 0, 1, 0])


def test_incomplete_group_is_rejected():
    events, trajectories = _valid_events()
    del trajectories["t7"]
    for key in [k for k in events if k.startswith("t7:")]:
        events.pop(key)
    with pytest.raises(ValueError, match="trajectories"):
        estimate_math_values(events, trajectories, CFG)


def test_missing_authoritative_terminal_is_rejected():
    events, trajectories = _valid_events()
    terminal = trajectories["t0"][-1]
    events[terminal]["env_done"] = False
    events[terminal]["env_reward"] = 0.0
    with pytest.raises(ValueError, match="missing true terminal boundary"):
        estimate_math_values(events, trajectories, CFG)


def test_reward_after_done_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][1]]["env_reward"] = 0.5
    events[trajectories["t0"][1]]["env_done"] = False
    with pytest.raises(ValueError, match="reward or done"):
        estimate_math_values(events, trajectories, CFG)


def test_nonterminal_reward_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][0]]["env_reward"] = 1.0
    with pytest.raises(ValueError):
        estimate_math_values(events, trajectories, CFG)


def test_nonbinary_terminal_reward_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][-1]]["env_reward"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        estimate_math_values(events, trajectories, CFG)


def test_cross_question_group_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][0]]["uid"] = "other-question"
    with pytest.raises(ValueError, match="crosses question groups"):
        estimate_math_values(events, trajectories, CFG)


def test_missing_event_breaks_chain_contract():
    events, trajectories = _valid_events()
    key = trajectories["t0"][2]
    events.pop(key)
    trajectories["t0"].remove(key)
    for index, remaining in enumerate(trajectories["t0"]):
        events[remaining]["event_index"] = index
    for remaining in trajectories["t0"]:
        events[remaining]["event_count"] = len(trajectories["t0"])
    with pytest.raises(ValueError, match="invalid Math event chain|incomplete"):
        estimate_math_values(events, trajectories, CFG)


def test_duplicate_trajectory_identity_is_rejected():
    events, trajectories = _valid_events()
    trajectories["t0-copy"] = list(trajectories["t0"])
    with pytest.raises(ValueError):
        estimate_math_values(events, trajectories, CFG)


def test_wrong_task_type_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][0]]["task_type"] = "search"
    with pytest.raises(ValueError, match="task_type=Math"):
        estimate_math_values(events, trajectories, CFG)


def test_missing_executed_text_is_rejected():
    events, trajectories = _valid_events()
    events[trajectories["t0"][0]].pop("executed_action_text")
    with pytest.raises(ValueError, match="missing required fields"):
        estimate_math_values(events, trajectories, CFG)


def test_engine_refuses_legacy_mode():
    events, trajectories = _valid_events()
    with pytest.raises(ValueError, match="math_answer_loto"):
        estimate_math_values(events, trajectories, MathValueConfig(mode="off"))
