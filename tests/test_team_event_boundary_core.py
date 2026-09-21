"""Execute the actual core function on CPU without Ray/DataProto imports.

AST extraction changes no function bodies. These are isolated numerical tests,
not a substitute for the server's full trainer/K5/distributed integration tests.
"""
import ast
import copy
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from team_event_trace_adapter import load_engine
from replay_team_event_boundary_value import gae_reference, normalize_reference

engine = load_engine()


def load_core():
    path = TOOLS.parent / "verl/trainer/ppo/core_algos.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_event_scalar", "_event_bool", "_event_state_key", "compute_team_event_gae_advantage"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    # The Math engine is a pure-CPU module too; inject it beside the Search
    # engine so the extracted function body sees the same import surface.
    sys.path.insert(0, str(TOOLS.parent))
    from verl.trainer.ppo.team_event_math_value import MathValueConfig, estimate_math_values
    scope = dict(globals(), BoundaryValueConfig=engine.BoundaryValueConfig,
                 estimate_boundary_values=engine.estimate_boundary_values,
                 MathValueConfig=MathValueConfig, estimate_math_values=estimate_math_values)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["compute_team_event_gae_advantage"]


core = load_core()


def fixture_records(rewards=(1, 0, 1, 0, 0, 1, 0, 0), varied=False):
    rows = []
    for t, reward in enumerate(rewards):
        # Variable depth checks per-trajectory rather than per-row normalization.
        actions = (["alpha", "beta", "ANSWER"] if varied and t % 2 else
                   ["alpha" if t % 3 else "gamma", "ANSWER"])
        chain = []
        for step, action in enumerate(actions):
            owner = "Answer Agent" if action == "ANSWER" else "Search Agent"
            for p, role in enumerate(("Verifier Agent", owner)):
                index = 2 * step + p
                chain.append(dict(event_uid=f"t{t}:{index}", uid="question", traj_uid=f"t{t}",
                                  event_index=index, event_count=2*len(actions), agent_id=role,
                                  event_type="verifier" if p == 0 else "final_answer" if action == "ANSWER" else "search_query",
                                  env_step_index=step, is_env_action=bool(p), env_action_owner=owner,
                                  env_done=bool(p and step == len(actions)-1),
                                  env_reward=float(reward) if p and step == len(actions)-1 else 0.,
                                  is_action_valid=True, route_target=owner if p == 0 else None,
                                  route_reason="verifier_yes" if owner == "Answer Agent" else "verifier_no",
                                  tool_observation=action if p and action != "ANSWER" else None))
        for role in {r["agent_id"] for r in chain}:
            cohort = [r for r in chain if r["agent_id"] == role]
            for i, record in enumerate(cohort):
                record.update(role_event_index=i, role_event_count=len(cohort))
        rows.extend(chain)
    return rows


@pytest.fixture
def value_config(tmp_path):
    asset = tmp_path / "idf.json"
    engine.FrozenTfidfKernel.fit(["alpha beta", "beta gamma", "gamma delta"]).export(asset)
    return dict(mode="boundary_loto", idf_path=str(asset), auxiliary_mode="env_only")


def run(rows, value_config=None, normalize=False, kl=None, **overrides):
    n = len(rows)
    mask = torch.tensor([[0., 1., 1.]] * n, dtype=torch.float64)
    # Simulate a reward manager broadcasting the episode score on all rows.
    scores = torch.full((n, 3), 10., dtype=torch.float64)
    rewards = scores.clone()
    if kl is not None:
        rewards += torch.tensor(kl, dtype=torch.float64)
    array = lambda field: np.array([r[field] for r in rows], dtype=object)
    kwargs = {key: array(key) for key in (
        "uid", "event_uid", "event_index", "event_count", "role_event_index", "role_event_count",
        "agent_id", "event_type", "env_step_index", "is_env_action", "env_action_owner",
        "env_reward", "env_done", "is_action_valid", "route_target", "route_reason", "tool_observation")}
    kwargs.update(traj_index=array("traj_uid"), state_chats=np.array([str(r["env_step_index"]) for r in rows], dtype=object),
                  token_level_rewards=rewards, token_level_scores=scores, response_mask=mask,
                  value_config=value_config, normalize_advantages=normalize, lam=.95)
    kwargs.update(overrides)
    meta = {}
    a, ret, diagnostics = core(**kwargs, diagnostics_meta=meta)
    return a, ret, diagnostics, meta


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_active_core_matches_independent_gae_and_normalization(value_config, varied, normalize):
    rows = fixture_records(varied=varied)
    a, ret, d, meta = run(rows, value_config, normalize, lam=.8, internal_lam=.8)
    events = {r["event_uid"]: r for r in rows}
    trajectories = defaultdict(list)
    for r in rows:
        trajectories[r["traj_uid"]].append(r["event_uid"])
    result = engine.estimate_boundary_values(events, trajectories, value_config)
    raw, _, _, _, error = gae_reference(events, trajectories, result.pre, result.post,
                                       lambda_env=.8, lambda_internal=.8)
    normalized, _ = normalize_reference(events, raw, enabled=normalize)
    assert error < 1e-12
    expected = torch.tensor([normalized[r["event_uid"]] for r in rows], dtype=torch.float64)
    assert torch.allclose(a[:, 1], expected, atol=1e-12, rtol=0)
    assert torch.equal(a[:, 1], a[:, 2]) and torch.count_nonzero(a[:, 0]) == 0
    assert torch.equal(d["event_raw_advantages"], d["event_boundary_raw_advantages"])
    assert torch.equal(d["event_values"], d["event_boundary_value_pre"])
    assert torch.allclose(ret[:, 1], d["event_raw_advantages"] + d["event_values"], atol=1e-12)
    assert len(meta["details_by_event"]) == len(rows) and len(meta["idf_sha256"]) == 64


def test_shadow_keeps_legacy_actor_and_returns_exact(value_config):
    rows = fixture_records(varied=True)
    rows[1]["is_action_valid"] = False
    opts = dict(normalize=True, invalid_action_penalty_coef=.1, kl=[[.7, -.02, -.03]]*len(rows))
    legacy = run(rows, **opts)
    shadow = run(rows, dict(value_config, application="shadow", auxiliary_mode="keep_existing_local"), **opts)
    for i in (0, 1):
        assert torch.equal(legacy[i], shadow[i])
    for key, old in legacy[2].items():
        assert torch.equal(old, shadow[2][key])
    assert not torch.equal(shadow[2]["event_advantages"], shadow[2]["event_boundary_advantages"])


def test_duplicate_and_row_order_invariance(value_config):
    rows = fixture_records(varied=True)
    base = run(rows, value_config, True)
    copied = rows + [copy.deepcopy(rows[i]) for i in (1, 1, 7, 9, 11, 13, 13)]
    np.random.default_rng(7).shuffle(copied)
    revised = run(copied, value_config, True)
    indices = {r["event_uid"]: i for i, r in enumerate(rows)}
    for j, row in enumerate(copied):
        i = indices[row["event_uid"]]
        assert torch.allclose(base[0][i], revised[0][j], atol=1e-12, rtol=0)
        for key in base[2]:
            assert torch.allclose(base[2][key][i], revised[2][key][j], atol=1e-12, rtol=0)
    copied[-1]["tool_observation"] = "conflicting document"
    with pytest.raises(ValueError, match="Copied rows disagree"):
        run(copied + [rows[indices[copied[-1]["event_uid"]]]], value_config)


def test_aux_is_event_local_value_environment_only_and_masked(value_config):
    rows = fixture_records()
    pure = run(rows, value_config)
    rows[1]["is_action_valid"] = False
    kl = [[123., 0., 0.] for _ in rows]  # Prompt/padding is not reward shaping.
    kl[2][1] = -.2
    _, _, d, _ = run(rows, dict(value_config, auxiliary_mode="keep_existing_local"),
                      kl=kl, invalid_action_penalty_coef=.1)
    assert torch.equal(d["event_boundary_value_pre"], pure[2]["event_boundary_value_pre"])
    assert torch.equal(d["event_boundary_value_post"], pure[2]["event_boundary_value_post"])
    assert torch.allclose(d["event_boundary_aux_raw_advantages"][:4],
                          torch.tensor([-.29, -.29, -.2, 0.], dtype=torch.float64), atol=1e-12)
    assert torch.count_nonzero(d["event_boundary_aux_raw_advantages"][4:]) == 0
    assert d["event_boundary_reward_kl"].sum().item() == pytest.approx(-.2)
    assert d["event_boundary_reward_invalid"].sum().item() == pytest.approx(-.1)
    assert torch.allclose(d["event_raw_advantages"], d["event_boundary_task_raw_advantages"] + d["event_boundary_aux_raw_advantages"])


@pytest.mark.parametrize("rewards", [(0,)*8, (1,)*8])
def test_equal_outcomes_have_zero_pure_task_signal(value_config, rewards):
    _, _, d, _ = run(fixture_records(rewards, varied=True), value_config, True)
    assert torch.allclose(d["event_raw_advantages"], torch.zeros_like(d["event_raw_advantages"]), atol=1e-12)
    assert torch.count_nonzero(d["event_advantages"]) == 0


def test_unique_success_not_erased(value_config):
    _, _, d, _ = run(fixture_records((1, 0, 0, 0, 0, 0, 0, 0)), value_config)
    assert torch.equal(d["event_values"][:4], torch.zeros(4, dtype=torch.float64))
    assert torch.allclose(d["event_raw_advantages"][:4], torch.tensor([.95, .95, 1., 1.], dtype=torch.float64))


@pytest.mark.parametrize("kwargs,match", [
    ({"gamma": .99}, "gamma"), ({"internal_gamma": .99}, "gamma"),
    ({"agent_local_mode": "blend"}, "agent_local"),
    ({"invalid_action_penalty_coef": .01}, "env_only"),
    ({"tool_observation": None}, "tool_observation"),
    ({"route_target": None}, "route_target"),
])
def test_runtime_guards_fail_closed(value_config, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(fixture_records(), value_config, **kwargs)


def test_env_only_rejects_actual_reward_kl(value_config):
    rows = fixture_records()
    with pytest.raises(ValueError, match="env_only"):
        run(rows, value_config, kl=[[0., -.01, 0.]]*len(rows))


def test_legacy_default_analytic_chain_and_no_asset():
    a, _, d, meta = run(fixture_records((1, 0)), normalize=False)
    expected = torch.tensor([.95, .95, 1., 1., -.95, -.95, -1., -1.], dtype=torch.float64)
    assert torch.allclose(a[:, 1], expected, atol=1e-12)
    assert not meta and not any(k.startswith("event_boundary_") for k in d)
