"""Run real zxj Math LOTO/GAE and hybrid credit helpers without Ray workers.

Only DataProto transport is replaced. The fixture adapter, numerical estimator,
Top16 ranking, temporal gates, and final advantage multiplication are production
functions executed unchanged with real CPU tensors.
"""
import ast
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[3]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _definitions(path, namespace, names=None):
    parsed = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (names is None or node.name in names)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, path, "exec"), namespace)


@pytest.fixture(scope="module")
def runtime():
    math_value = _module("hybrid_integration_math", "verl/trainer/ppo/team_event_math_value.py")
    boundary = _module("hybrid_integration_boundary", "verl/trainer/ppo/team_event_boundary_value.py")
    entropy = _module("hybrid_integration_entropy", "verl/utils/entropy_credit.py")
    sparse = _module("hybrid_integration_sparse", "verl/utils/sparse_entropy_credit.py")
    observability = _module("hybrid_integration_observability", "verl/utils/hybrid_observability.py")
    ns = {"torch": torch, "np": np, "math": math, "re": re, "Counter": Counter,
          "defaultdict": defaultdict, "MathValueConfig": math_value.MathValueConfig,
          "BoundaryValueConfig": boundary.BoundaryValueConfig,
          "estimate_math_values": math_value.estimate_math_values,
          "estimate_boundary_values": boundary.estimate_boundary_values,
          "compute_entropy_credit_multipliers": entropy.compute_entropy_credit_multipliers,
          "first_unique_action_indices": entropy.first_unique_action_indices,
          "prepare_sparse_entropy_credit": sparse.prepare_sparse_entropy_credit,
          "finalize_sparse_entropy_credit": sparse.finalize_sparse_entropy_credit,
          "compute_hybrid_observability": observability.compute_hybrid_observability,
          "json": json, "FIXTURES": ROOT / "tests/fixtures/math_loto",
          "MATH_CFG": {"mode": "math_answer_loto"}}
    _definitions("verl/trainer/ppo/core_algos.py", ns,
                 {"_event_scalar", "_event_bool", "_event_state_key", "compute_team_event_gae_advantage"})
    _definitions("tests/test_team_event_math_gae.py", ns,
                 {"_opaque", "_fixture_group", "_math_records", "_math_run"})
    _definitions("verl/trainer/ppo/hybrid_credit.py", ns)
    _definitions("verl/trainer/ppo/hybrid_training.py", ns, {"HybridTrainingMixin"})
    return ns


class Batch:
    def __init__(self, tensors, metadata):
        self.batch = tensors
        self.non_tensor_batch = metadata

    def __len__(self):
        return len(self.batch["responses"])

    def take(self, indices):
        return Batch({key: tensor[indices].clone() for key, tensor in self.batch.items()},
                     {key: values[indices].copy() for key, values in self.non_tensor_batch.items()})


def data_for(runtime, fixture="one_quarter_correction"):
    group = runtime["_fixture_group"](fixture)
    records = runtime["_math_records"](group)
    advantages, returns, diagnostics, _ = runtime["_math_run"](
        records, normalize=True, width=4, pad_from=3)
    size = len(records)
    mask = torch.ones(size, 4)
    mask[:, -1] = 0
    tensors = {"responses": torch.zeros(size, 4, dtype=torch.long),
               "attention_mask": torch.cat((torch.ones(size, 2), mask), -1),
               "response_mask": mask, "advantages": advantages, "returns": returns,
               "token_level_rewards": torch.zeros(size, 4),
               "token_level_scores": torch.zeros(size, 4),
               "sample_weight": torch.ones(size), **diagnostics}
    mapping = {"event_uid": "event_uid", "event_index": "index", "role_event_index": "role_index",
               "traj_uid": "traj", "uid": "uid", "agent_id": "agent"}
    meta = {key: np.asarray([record[column] for record in records]) for key, column in mapping.items()}
    meta["value_action_index"] = meta["event_index"].copy()
    success = {trajectory["traj_uid"]: trajectory["R"] for trajectory in group}
    meta["pass"] = np.asarray([success[record["traj"]] for record in records])
    meta["is_action_valid"] = np.ones(size, dtype=bool)
    trajectory_index = {trajectory["traj_uid"]: index for index, trajectory in enumerate(group)}
    previous = np.linspace(.2, .8, len(group))
    previous[0] = .95
    current = previous + .01
    current[0] = .1
    entropies = []
    for record in records:
        index = trajectory_index[record["traj"]]
        turn = record["role_index"]
        entropies.append((previous[index] if turn == 0 else current[index] + .005 * (turn - 1))
                         if record["agent"] == "Solver Agent" else .3 + .02 * index + .01 * turn)
    meta["top16_entropy_mean"] = np.asarray(entropies)
    meta["top16_entropy"] = np.asarray(
        [{"mean": value, "coverage": 1., "effective_support": 8.} for value in entropies], dtype=object)
    return Batch(tensors, meta), records


def prepare(runtime, data, *, action_scale=.2, eta=.04):
    config = {"action_scale": action_scale, "sparse": {"enable": True, "eta": eta}}
    runtime["prepare_hybrid_action_metadata"](data)
    runtime["prepare_entropy_credit"](data, config)
    runtime["prepare_sparse_entropy_credit_for_batch"](data, config["sparse"])
    return config


def finish(runtime, data, config):
    _, metrics = runtime["finalize_sparse_entropy_credit_for_advantages"](data, config["sparse"])
    runtime["apply_entropy_credit_to_advantages"](data)
    return metrics


def test_disabled_driver_skips_credit_and_preserves_exact_zxj_output(runtime):
    data, _ = data_for(runtime)
    expected = {key: tensor.clone() for key, tensor in data.batch.items()}
    # Disabled modulation requires no Top16 or semantic fields.
    data.non_tensor_batch = {}
    driver = runtime["HybridTrainingMixin"]()
    driver.hybrid_enabled = False
    driver.entropy_credit_enabled = False
    driver.entropy_controller = None
    assert driver._prepare_hybrid_rollout(data) == {}
    assert driver._apply_hybrid_after_advantage(data) == {}
    assert data.non_tensor_batch == {}
    for key, tensor in expected.items():
        torch.testing.assert_close(data.batch[key], tensor, rtol=0, atol=0)


def test_driver_preserves_base_advantage_audit_and_uses_one_final_coefficient(runtime):
    class Config(dict):
        __getattr__ = dict.__getitem__

    data, _ = data_for(runtime)
    before = data.batch["advantages"].clone()
    driver = runtime["HybridTrainingMixin"]()
    driver.hybrid_enabled = True
    driver.entropy_credit_enabled = True
    driver.entropy_credit_config = Config(action_scale=.2, sparse=Config(enable=True, eta=.04))
    driver.entropy_controller = None
    driver.semantic_scorer = None
    driver._prepare_hybrid_rollout(data)
    driver._apply_hybrid_after_advantage(data)
    coefficients = torch.tensor(data.non_tensor_batch["entropy_credit_final_multiplier"], dtype=torch.float32)
    torch.testing.assert_close(data.batch["hybrid_base_advantages"], before, rtol=0, atol=0)
    torch.testing.assert_close(data.batch["advantages"], before * coefficients[:, None], rtol=0, atol=0)


def test_neutral_modulation_matches_exact_zxj_math_output(runtime):
    data, _ = data_for(runtime)
    expected = {key: tensor.clone() for key, tensor in data.batch.items()}
    config = prepare(runtime, data, action_scale=0., eta=0.)
    finish(runtime, data, config)
    np.testing.assert_array_equal(data.non_tensor_batch["entropy_credit_final_multiplier"], np.ones(len(data)))
    for key, tensor in expected.items():
        torch.testing.assert_close(data.batch[key], tensor, rtol=0, atol=0)


def test_two_stages_only_scale_actor_advantages_and_preserve_all_zxj_targets(runtime):
    data, _ = data_for(runtime)
    expected = {key: tensor.clone() for key, tensor in data.batch.items()}
    config = prepare(runtime, data)
    finish(runtime, data, config)
    coefficients = torch.tensor(data.non_tensor_batch["entropy_credit_final_multiplier"], dtype=torch.float32)
    assert torch.all((coefficients >= .8) & (coefficients <= 1.2))
    assert torch.any(coefficients != 1.)
    torch.testing.assert_close(data.batch["advantages"], expected["advantages"] * coefficients[:, None])
    torch.testing.assert_close(torch.sign(data.batch["advantages"]), torch.sign(expected["advantages"]))
    for key, tensor in expected.items():
        if key != "advantages":
            torch.testing.assert_close(data.batch[key], tensor, rtol=0, atol=0)
    assert not any("recover" in key or "policy_action_weight" == key for key in data.batch)


@pytest.mark.parametrize("fixture", ["all_success", "all_failure"])
def test_zero_variance_groups_are_not_recovered(runtime, fixture):
    data, _ = data_for(runtime, fixture)
    assert not torch.any(data.batch["advantages"])
    config = prepare(runtime, data)
    finish(runtime, data, config)
    assert not torch.any(data.batch["advantages"])
    assert not np.any(data.non_tensor_batch["pure_entropy_selected_gate"])


def test_padding_copies_inherit_frozen_coefficients_and_keep_event_weights(runtime):
    unique, _ = data_for(runtime)
    config = prepare(runtime, unique)
    indices = np.asarray([*reversed(range(len(unique))), 0, 0, 3, 7, 7])
    padded = unique.take(indices)
    multiplicities = np.bincount(indices, minlength=len(unique))
    padded.batch["sample_weight"] = torch.tensor(1. / multiplicities[indices], dtype=torch.float32)
    expected_weights = padded.batch["sample_weight"].clone()
    expected_returns = padded.batch["returns"].clone()
    finish(runtime, unique, config)
    metrics = finish(runtime, padded, config)
    np.testing.assert_allclose(padded.non_tensor_batch["entropy_credit_final_multiplier"],
                               unique.non_tensor_batch["entropy_credit_final_multiplier"][indices])
    torch.testing.assert_close(padded.batch["advantages"], unique.batch["advantages"][indices])
    torch.testing.assert_close(padded.batch["sample_weight"], expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(padded.batch["returns"], expected_returns, rtol=0, atol=0)
    assert metrics["pure_entropy/final_padding_copies"] == 5


@pytest.mark.parametrize("pair_advantages", [(-.5, .2), (.2, -.5), (-.5, -.2), (0., .2)])
def test_success_flag_cannot_activate_stage_two_for_incompatible_advantage_signs(runtime, pair_advantages):
    data, records = data_for(runtime)
    config = prepare(runtime, data)
    candidates = np.flatnonzero(data.non_tensor_batch["pure_entropy_correction_gate"] > 0)
    assert len(candidates) == 1
    current = int(candidates[0])
    record = records[current]
    prior = next(i for i, item in enumerate(records)
                 if item["traj"] == record["traj"] and item["agent"] == record["agent"]
                 and item["role_index"] == record["role_index"] - 1)
    assert data.non_tensor_batch["pass"][current] == 1
    # Inject the estimator/credit interface's allowed signed scalar values.
    # A successful terminal label must not replace the actual advantage sign.
    for row, value in zip((prior, current), pair_advantages):
        data.batch["advantages"][row] = value * data.batch["response_mask"][row]
    before = data.batch["advantages"].clone()
    first_stage = data.non_tensor_batch["entropy_credit_action_multiplier"].copy()
    metrics = finish(runtime, data, config)
    assert metrics["pure_entropy/sign_mismatch_edges"] == 1
    assert metrics["pure_entropy/active_edges"] == 0
    assert not np.any(data.non_tensor_batch["pure_entropy_selected_gate"])
    np.testing.assert_array_equal(data.non_tensor_batch["entropy_credit_final_multiplier"], first_stage)
    torch.testing.assert_close(torch.sign(data.batch["advantages"]), torch.sign(before))


def test_role_index_alias_and_unique_event_identity_are_checked(runtime):
    data, _ = data_for(runtime)
    runtime["prepare_hybrid_action_metadata"](data)
    np.testing.assert_array_equal(data.non_tensor_batch["role_turn_index"], data.non_tensor_batch["role_event_index"])
    solver = data.non_tensor_batch["agent_id"] == "Solver Agent"
    assert np.any(data.non_tensor_batch["event_index"][solver] != data.non_tensor_batch["role_turn_index"][solver])
    bad_alias = data.take(np.arange(len(data)))
    bad_alias.non_tensor_batch["role_turn_index"][2] += 1
    with pytest.raises(ValueError, match="role_turn_index"):
        runtime["prepare_hybrid_action_metadata"](bad_alias)
    duplicate = data.take(np.asarray([*range(len(data)), 0]))
    with pytest.raises(ValueError, match="unique|repeat|duplicate"):
        runtime["prepare_hybrid_action_metadata"](duplicate)


def test_global_semantic_action_index_must_equal_zxj_event_index(runtime):
    data, _ = data_for(runtime)
    data.non_tensor_batch["value_action_index"][2] = 999
    with pytest.raises(ValueError, match="value_action_index|event_index"):
        runtime["prepare_hybrid_action_metadata"](data)
