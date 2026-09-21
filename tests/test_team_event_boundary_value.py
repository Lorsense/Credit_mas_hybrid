"""CPU-only independent oracles for stage-1 boundary LOTO (no verl import)."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/team_event_boundary_value.py"
MODULE_NAME = "team_event_boundary_value"
if MODULE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
engine = sys.modules[MODULE_NAME]
Config = engine.BoundaryValueConfig
Kernel = engine.FrozenTfidfKernel


def trajectory(tid, actions, reward=0, *, group="q", reasons=None):
    """actions: current Search observation strings/None; 'ANSWER' is Answer."""
    records = []
    for k, action in enumerate(actions):
        answer = action == "ANSWER"
        owner = "Answer Agent" if answer else "Search Agent"
        for kind, producer, is_owner in (("verifier", "Verifier Agent", False),
                                         ("final_answer" if answer else "search_query", owner, True)):
            record = dict(event_uid=f"{tid}:{2*k+int(is_owner)}", uid=group, traj_uid=tid,
                          event_index=2*k+int(is_owner), event_count=2*len(actions), agent_id=producer,
                          event_type=kind, env_step_index=k, is_env_action=is_owner,
                          env_action_owner=owner, env_reward=float(reward) if is_owner and k == len(actions)-1 else 0.0,
                          env_done=is_owner and k == len(actions)-1, is_action_valid=True,
                          task_type="Search")
            if kind == "verifier":
                record.update(route_target=owner, route_reason=(reasons[k] if reasons else "normal"))
            elif kind == "search_query":
                record["tool_observation"] = action
            records.append(record)
    for role in {row["agent_id"] for row in records}:
        role_rows = [row for row in records if row["agent_id"] == role]
        for index, row in enumerate(role_rows):
            row.update(role_event_index=index, role_event_count=len(role_rows))
    return records


def batch(*chains):
    events = {row["event_uid"]: row for chain in chains for row in chain}
    trajectories = {chain[0]["traj_uid"]: [row["event_uid"] for row in chain] for chain in chains}
    return events, trajectories


def estimate(events, chains, **settings):
    kernel = settings.pop("kernel", None) or Kernel.fit(["alpha beta", "beta gamma", "delta epsilon"])
    settings.setdefault("expected_rollout_n", len(chains))
    settings.setdefault("record_matched_peers", True)
    config = Config.from_mapping(dict(mode="boundary_loto", **settings))
    return engine.estimate_boundary_values(events, chains, config, kernel)


def gae(events, chain, result, lambda_env=0.95, lambda_internal=1.0):
    out, following = {}, 0.0
    for key in reversed(chain):
        row = events[key]
        lam = 0 if row["env_done"] else lambda_env if row["is_env_action"] else lambda_internal
        delta = row["env_reward"] + result.post[key] - result.pre[key]
        following = delta + lam * following
        out[key] = following
    return out


@pytest.mark.parametrize(("raw", "expected"), [
    (None, ""), ("", ""),
    ("<information>Doc 1: Alpha\\nBeta</information>", "alpha beta"),
    ('{"result": "Doc 2: ALPHA beta"}', "alpha beta"),
    ('{"results": "Doc 3: ALPHA beta"}', "alpha beta"),
    ('Doc 19: ALPHA \\"beta\\" Alpha!', "alpha beta alpha"),
    ('{"result": ["Alpha", "Beta"]}', "alpha beta"),
    ("中文 Doc 2: 123", "123"),
])
def test_document_parser_matches_historical_keys(raw, expected):
    assert engine.canonical_document_key(raw) == expected


def test_structured_input_requires_explicit_adapter():
    with pytest.raises(ValueError, match="adapt structured"):
        engine.canonical_document_key(["alpha"])


def test_idf_unique_bundles_repeated_tf_and_oov_exact_oracle():
    kernel = Kernel.fit(["Doc 1: apple apple pear", "Doc 2: apple apple pear", "pear plum", None])
    assert kernel.M == 2
    assert dict(kernel.df) == {"apple": 1, "pear": 2, "plum": 1}
    apple = (1+math.log(2)) * (1+math.log(3/2))
    oov = 1+math.log(3)
    norm = math.sqrt(apple*apple + oov*oov)
    vector = kernel.vector("apple apple unseen")
    assert vector["apple"] == pytest.approx(apple/norm, abs=1e-15)
    assert vector["unseen"] == pytest.approx(oov/norm, abs=1e-15)
    assert "unseen" not in kernel.df
    with pytest.raises(TypeError):
        kernel.df["new"] = 2


def test_kernel_empty_equal_and_stop_only_priority():
    kernel = Kernel.fit(["alpha beta", "beta gamma"])
    assert kernel("", "") == 0
    assert kernel("", "alpha") == 0
    assert kernel("the and", "the and") == 1
    assert kernel("the and", "the or") == 0
    assert kernel("alpha beta", "alpha beta") == 1
    left, right = kernel.vector("alpha beta"), kernel.vector("beta gamma")
    assert kernel("alpha beta", "beta gamma") == pytest.approx((left["beta"]*right["beta"])**4, abs=1e-15)


def test_kernel_cache_bounded_and_asset_roundtrip(tmp_path):
    kernel = Kernel.fit(["alpha beta", "beta gamma"], cache_size=2,
                        metadata={"fit_step_start": 51, "fit_step_end": 81})
    path = tmp_path / "idf.json"
    digest = kernel.export(path)
    loaded = Kernel.load(path, expected_sha256=digest, cache_size=2)
    assert loaded.idf_sha256 == digest
    assert loaded.to_asset() == kernel.to_asset()
    for key in ("a1 alpha", "b2 beta", "c3 gamma", "d4 delta"):
        loaded(key, "alpha beta")
    assert loaded.cache_info == {"vectors": 2, "pairs": 2, "limit": 2}
    with pytest.raises(FileExistsError):
        kernel.export(path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        Kernel.load(path, expected_sha256="0"*64)
    damaged = kernel.to_asset()
    damaged["stop_words"] = ["the"]
    with pytest.raises(ValueError, match="stop_words"):
        Kernel.from_asset(damaged)
    damaged = kernel.to_asset()
    damaged["normalization_version"] = "sklearn-default"
    with pytest.raises(ValueError, match="normalization_version"):
        Kernel.from_asset(damaged)


@pytest.mark.parametrize("kwargs", [
    {"target": "shaped_reward"}, {"similarity": "embedding"}, {"expected_rollout_n": 1},
    {"expected_rollout_n": 2.0}, {"depth_window": -1}, {"strict": False},
    {"parent_alpha": 0}, {"question_parent_beta": -1}, {"similarity_power": float("nan")},
    {"cross_depth_decay": 1.1}, {"verifier_same_route": False},
    {"verifier_same_depth_same_reason": True}, {"auxiliary_mode": "silent_drop"},
    {"application": "maybe"}, {"typo": 2},
])
def test_configuration_fail_closed(kwargs):
    with pytest.raises(ValueError):
        Config.from_mapping(dict(mode="boundary_loto", **kwargs))


def test_legacy_default_runtime_guards_and_immutable_config():
    assert Config.from_mapping().mode == "legacy_loto"
    config = Config.from_mapping({"mode": "boundary_loto"})
    config.validate_runtime(1, 1, "off")
    with pytest.raises(ValueError, match="discounted peer"):
        config.validate_runtime(.95, 1, "off")
    with pytest.raises(ValueError, match="G1/SMDP"):
        config.validate_runtime(1, 1, "active")
    with pytest.raises(AttributeError):
        config.mode = "legacy_loto"
    events, chains = batch(trajectory("a", ["ANSWER"]), trajectory("b", ["ANSWER"], 1))
    with pytest.raises(ValueError, match="frozen IDF"):
        engine.estimate_boundary_values(events, chains, Config(mode="boundary_loto", expected_rollout_n=2))


def test_shared_boundaries_mc_and_telescoping_for_unequal_chains():
    events, chains = batch(trajectory("a", ["alpha", "beta", "ANSWER"], 1),
                           trajectory("b", ["beta", "ANSWER"], 0),
                           trajectory("c", ["ANSWER"], 1))
    result = estimate(events, chains)
    for tid, keys in chains.items():
        for current, following in zip(keys, keys[1:]):
            assert result.post[current] == result.pre[following]
        assert result.post[keys[-1]] == 0
        deltas = [events[key]["env_reward"]+result.post[key]-result.pre[key] for key in keys]
        assert sum(deltas) == pytest.approx(result.returns[tid]-result.pre[keys[0]], abs=1e-12)
        mc = gae(events, keys, result, 1, 1)
        for key in keys:
            assert mc[key] == pytest.approx(result.returns[tid]-result.pre[key], abs=1e-12)


def test_own_whole_trajectory_reward_exclusion_and_input_permutation():
    events, chains = batch(trajectory("a", ["alpha", "alpha", "ANSWER"], 0),
                           trajectory("b", ["beta", "ANSWER"], 0),
                           trajectory("c", ["gamma", "ANSWER"], 1))
    original = copy.deepcopy((events, chains))
    result = estimate(events, chains)
    assert (events, chains) == original
    changed = copy.deepcopy(events)
    changed[chains["a"][-1]]["env_reward"] = 1
    other = estimate(changed, chains)
    for key in chains["a"]:
        assert result.pre[key] == other.pre[key]
        assert result.post[key] == other.post[key]
        assert all(peer["traj_uid"] != "a" for peer in result.details[key]["matched_peers"])
    assert gae(events, chains["a"], result) != gae(changed, chains["a"], other)
    shuffled = estimate(dict(reversed(list(events.items()))), {key: list(reversed(value)) for key, value in reversed(list(chains.items()))})
    assert shuffled == result


def test_parent_zero_one_seven_peers_and_no_document_support():
    own = trajectory("t0", ["own0", "own1", "own2", "ANSWER"], 1)
    peers = [trajectory(f"t{k}", ["peer0", "peer1", "ANSWER"] if k == 1 else ["ANSWER"],
                        1 if k in (2, 3, 4) else 0) for k in range(1, 8)]
    events, chains = batch(own, *peers)
    result = estimate(events, chains)
    d0, d1, d2 = (result.details[f"t0:{e}"] for e in (0, 3, 5))
    assert d0["n_same"] == 7
    assert d0["parent"] == pytest.approx(3/7)
    assert d1["n_same"] == 1
    assert d1["mass"] == 0
    assert d1["parent"] == pytest.approx(2/7)
    assert result.post["t0:3"] == pytest.approx(2/7)
    assert d2["n_same"] == 0
    assert d2["parent"] == pytest.approx(3/7)
    assert result.post["t0:5"] == pytest.approx(3/7)


def test_search_adjacent_window_max_per_peer_and_chronological_ties():
    events, chains = batch(trajectory("a", ["x", "shared", "z", "ANSWER"], 0),
                           trajectory("b", ["shared", "shared", "shared", "ANSWER"], 1),
                           trajectory("c", ["different", "ANSWER"], 0))
    result = estimate(events, chains)
    detail = result.details["a:3"]
    assert detail["mass"] == 1
    assert detail["positive_peer_count"] == 1
    assert detail["matched_peers"][0]["event_uid"] == "b:3"
    assert detail["ess"] == 1
    tied = estimate(events, chains, cross_depth_decay=1)
    assert tied.details["a:3"]["matched_peers"][0]["event_uid"] == "b:1"
    # Only a peer's adjacent S0 matches S1; a peer's S0 cannot match S2.
    events["b:3"]["tool_observation"] = "other"
    events["b:5"]["tool_observation"] = "other"
    events["a:5"]["tool_observation"] = "shared"
    result = estimate(events, chains)
    assert result.details["a:3"]["mass"] == .5
    assert result.details["a:3"]["cross_round_peer_count"] == 1
    assert result.details["a:5"]["mass"] == 0


def test_verifier_same_round_route_condition_not_parent_condition():
    events, chains = batch(trajectory("a", ["shared", "ANSWER"], 1),
                           trajectory("b", ["shared", "other", "ANSWER"], 0))
    result = estimate(events, chains, depth_window=0)
    detail = result.details["a:2"]
    assert detail["n_same"] == 1  # Coarse parent includes differently routed V1.
    assert detail["mass"] == 0
    assert result.pre["a:2"] == result.post["a:1"]


def test_verifier_same_round_reasons_not_split_cross_round_reasons_split():
    events, chains = batch(trajectory("a", ["shared", "ANSWER"], 0, reasons=["n", "yes"]),
                           trajectory("b", ["shared", "ANSWER"], 1, reasons=["n", "forced"]))
    result = estimate(events, chains)
    assert result.details["a:2"]["mass"] == 1
    events, chains = batch(trajectory("a", ["shared", "shared", "ANSWER"], 0,
                                     reasons=["n", "n", "yes"]),
                           trajectory("b", ["shared", "ANSWER"], 1, reasons=["n", "forced"]))
    result = estimate(events, chains)
    assert result.details["a:4"]["mass"] == 0
    events["b:2"]["route_reason"] = "yes"
    result = estimate(events, chains)
    assert result.details["a:4"]["mass"] == .5


def test_v0_no_docs_exception_cannot_match_later_verifier():
    events, chains = batch(trajectory("a", [None, "ANSWER"], 0),
                           trajectory("b", [None, None, "ANSWER"], 1))
    result = estimate(events, chains)
    # V0 only gets peer V0's one unit, never the later no-document V1.
    assert result.details["a:0"]["mass"] == 1
    assert result.details["a:0"]["matched_peers"][0]["event_uid"] == "b:0"
    assert result.details["a:1"]["mass"] == 0  # Ordinary empty Search has no shortcut.
    assert result.details["a:2"]["mass"] == 0  # Later empty V also has no shortcut.


def test_current_observation_time_and_empty_search_clear_previous():
    events, chains = batch(trajectory("a", ["old", None, "ANSWER", "ANSWER"], 1),
                           trajectory("b", ["old", "unrelated", "ANSWER"], 0),
                           trajectory("c", ["fresh", "ANSWER"], 1))
    result = estimate(events, chains)
    assert result.details["a:2"]["document_key_sha256"] == result.details["a:1"]["document_key_sha256"]
    assert result.details["a:4"]["document_key_sha256"] == result.details["a:3"]["document_key_sha256"]
    assert result.details["a:4"]["document_key_sha256"] != result.details["a:1"]["document_key_sha256"]
    assert result.post["a:5"] == result.pre["a:5"]  # Nonterminal Answer carry.
    assert result.details["a:6"]["document_key_sha256"] == result.details["a:4"]["document_key_sha256"]
    changed = copy.deepcopy(events)
    changed["a:3"]["tool_observation"] = "fresh"
    revised = estimate(changed, chains)
    for key in ("a:0", "a:1", "a:2"):
        assert result.pre[key] == revised.pre[key]
        assert result.post[key] == revised.post[key]
    assert result.pre["a:3"] == revised.pre["a:3"]
    assert result.post["a:3"] != revised.post["a:3"]


def test_nonterminal_answer_preserves_last_search_not_answer_text():
    events, chains = batch(trajectory("a", ["evidence", "ANSWER", "ANSWER"], 1),
                           trajectory("b", ["evidence", "ANSWER"], 0))
    result = estimate(events, chains)
    assert result.details["a:4"]["document_key_sha256"] == result.details["a:1"]["document_key_sha256"]
    assert result.post["a:3"] == result.pre["a:3"]
    assert result.post["a:5"] == 0


@pytest.mark.parametrize("reward", [0, 1])
def test_all_same_results_have_zero_task_gae(reward):
    events, chains = batch(*(trajectory(f"t{i}", [f"doc{i}", "ANSWER"], reward) for i in range(8)))
    result = estimate(events, chains)
    for keys in chains.values():
        assert all(abs(value) < 1e-12 for value in gae(events, keys, result, .8, .8).values())


@pytest.mark.parametrize(("env_lam", "internal_lam", "expected"), [
    (.95, 1, [.95, .95, 1, 1]), (.8, .8, [.512, .64, .8, 1]), (0, 0, [0, 0, 0, 1]),
])
def test_single_success_has_zero_own_value_but_positive_gae(env_lam, internal_lam, expected):
    events, chains = batch(*(trajectory(f"t{i}", ["shared", "ANSWER"], int(i == 0)) for i in range(8)))
    result = estimate(events, chains)
    keys = chains["t0"]
    assert all(result.pre[key] == result.post[key] == 0 for key in keys)
    raw = gae(events, keys, result, env_lam, internal_lam)
    assert [raw[key] for key in keys] == pytest.approx(expected, abs=1e-12)


def test_group_isolation_and_reference_empty_parent_carry():
    a = trajectory("a", ["one", "two", "ANSWER"], 1)
    b = trajectory("b", ["one", "ANSWER"], 0)
    events, chains = batch(a, b)
    reference = estimate(events, chains, depth_window=0, question_parent_beta=0,
                         empty_parent_fallback="carry")
    assert reference.post["a:3"] == reference.pre["a:3"]
    extra = trajectory("x", ["one", "ANSWER"], 1, group="other")
    extra2 = trajectory("y", ["one", "ANSWER"], 1, group="other")
    joined_events, joined_chains = batch(a, b, extra, extra2)
    joined = estimate(joined_events, joined_chains, expected_rollout_n=2)
    separate = estimate(events, chains)
    for key in events:
        assert joined.pre[key] == separate.pre[key]
        assert joined.post[key] == separate.post[key]


@pytest.mark.parametrize(("key", "field", "value", "message"), [
    ("a:0", "env_reward", 1, "non-owner"),
    ("a:1", "env_reward", .1, "nonterminal reward"),
    ("a:3", "env_reward", .5, "binary terminal"),
    ("a:3", "env_done", False, "true terminal"),
    ("a:1", "env_done", True, "after done"),
    ("a:0", "route_target", "Answer Agent", "executed owner"),
    ("a:0", "route_reason", "", "route_reason"),
    ("a:2", "event_index", 9, "incomplete event_index"),
    ("a:2", "role_event_index", 9, "incomplete role"),
    ("a:2", "env_step_index", 3, "contiguous chronological"),
    ("a:0", "task_type", "AppWorld", "task_type=Search"),
    ("a:0", "is_env_action", True, "ownership mismatch"),
])
def test_schema_and_reward_failures_are_explicit(key, field, value, message):
    events, chains = batch(trajectory("a", ["alpha", "ANSWER"], 1), trajectory("b", ["beta", "ANSWER"], 0))
    if field == "env_done" and value is False:
        # Isolate missing terminal metadata from the separate nonterminal-reward guard.
        events[key]["env_reward"] = 0.0
    events[key][field] = value
    with pytest.raises(ValueError, match=message):
        estimate(events, chains)


def test_missing_observation_not_treated_as_empty_and_incomplete_groups_fail():
    events, chains = batch(trajectory("a", ["alpha", "ANSWER"], 1), trajectory("b", ["beta", "ANSWER"], 0))
    del events["a:1"]["tool_observation"]
    with pytest.raises(ValueError, match="missing tool_observation"):
        estimate(events, chains)
    events["a:1"]["tool_observation"] = None
    estimate(events, chains)
    with pytest.raises(ValueError, match="expected_rollout_n=8"):
        estimate(events, chains, expected_rollout_n=8)
    broken = copy.deepcopy(chains)
    broken["a"].append("a:0")
    with pytest.raises(ValueError, match="duplicate event UID"):
        estimate(events, broken)
    del broken["a"]
    with pytest.raises(ValueError, match="orphan"):
        estimate(events, broken, expected_rollout_n=2)


def test_full_precision_eight_event_gae_oracle():
    boundaries = [.42857142857142855, .4285714285714286, .8164124352210786,
                  .8164886055150277, .7777129084222318, .7777503486091735,
                  .7142845887114423, .7142845887114423, 0.0]
    expected = [.3296061373150809, .41200767164385105, .030208331242751255,
                .03766520118600279, .09555112284849826, .11939210332694578,
                .22857232903084618, .2857154112885577]
    rows = trajectory("a", ["first", "second", "third", "ANSWER"], 1)
    events = {row["event_uid"]: row for row in rows}
    chain = list(events)
    result = engine.BoundaryValueResult(dict(zip(chain, boundaries[:-1])),
                                       dict(zip(chain, boundaries[1:])), {}, {"a": 1}, "oracle")
    raw = gae(events, chain, result, .8, .8)
    assert [raw[key] for key in chain] == pytest.approx(expected, abs=1e-12)
    # Independent Search and second-Verifier weighted arithmetic from the guide.
    assert (1.5+2/7)/(1.5000039395161595+1) == pytest.approx(boundaries[6], abs=1e-15)
    assert (2.114288263668777+3/7)/(2.1143847875699517+1) == pytest.approx(boundaries[3], abs=1e-15)
