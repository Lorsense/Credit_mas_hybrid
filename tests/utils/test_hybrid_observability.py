import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("hybrid_observability_test", Path(__file__).resolve().parents[2] / "verl/utils/hybrid_observability.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def batch():
    base = np.array([[1, 1, np.nan], [-1, -1, np.nan], [0, 0, np.nan], [0, 0, np.nan]], dtype=float)
    data = {"advantages": base * np.array([1.2, .8, 1.2, .8])[:, None], "hybrid_base_advantages": base,
            "response_mask": np.array([[1, 1, 0]] * 4),
            "entropy_control_weight": np.array([.5, 0, .8, 0]),
            "entropy_control_cap": np.ones(4) * 1.5, "entropy_control_valid": np.array([1, 1, 1, 0], bool)}
    meta = {"event_uid": np.array(["a", "b", "c", "d"]), "uid": np.array(["q", "q", "z", "z"]),
            "agent_id": np.array(["Solver Agent"] * 4), "value_control_eligible": np.array([1, 1, 1, 0], bool),
            "value_credit_available": np.array([1, 0, 1, 1], bool), "value_credit_delta": np.array([-.2, 0, -.1, .5]),
            "pass": np.array([1, 0, 1, 1]), "pure_entropy_correction_gate": np.array([.2, 0, 0, 0]),
            "pure_entropy_regression_gate": np.array([0, .3, 0, 0]), "pure_entropy_selected_gate": np.array([.2, 0, 0, 0]),
            "entropy_credit_action_multiplier": np.array([1.2, .8, 1.2, .8]),
            "entropy_credit_final_multiplier": np.array([1.18, .8, 1.2, .8])}
    return SimpleNamespace(batch=data, non_tensor_batch=meta)


def test_event_advantage_scaling_zero_groups_and_semantic_zero_signal():
    result = module.compute_hybrid_observability(batch())
    p = "hybrid/solver/"
    assert result[p + "advantage/base_mean"] == 0
    assert result[p + "advantage/final_mean"] == pytest.approx(.1)
    assert result[p + "advantage/base_std"] == pytest.approx(np.sqrt(.5))
    assert result[p + "advantage/base_abs_mean"] == .5
    assert result[p + "advantage/scaling_changed_fraction"] == .5
    assert result[p + "advantage/sign_flip_fraction"] == 0
    assert result[p + "zero_groups/count"] == 1
    assert result[p + "zero_groups/preserved_fraction"] == 1
    assert result[p + "stage2/candidate_skip_fraction"] == .5
    assert result[p + "semantic/eligible_fraction"] == .75
    assert result[p + "semantic/available_given_eligible_fraction"] == pytest.approx(2 / 3)
    assert result[p + "semantic/base_zero_gated_fraction"] == .5
    assert result[p + "semantic/gated_cap_mean"] == 1.5
    assert all(isinstance(value, float) and np.isfinite(value) for value in result.values())


def test_padding_copies_and_row_reordering_do_not_change_any_statistic():
    original = batch()
    copied = copy.deepcopy(original)
    indices = [3, 1, 0, 0, 0, 2, 3]
    copied.batch = {key: value[indices] for key, value in copied.batch.items()}
    copied.non_tensor_batch = {key: value[indices] for key, value in copied.non_tensor_batch.items()}
    assert module.compute_hybrid_observability(original) == module.compute_hybrid_observability(copied)


def test_conflicting_copies_cannot_silently_bias_metrics():
    data = batch()
    data.non_tensor_batch["event_uid"][1] = "a"
    with pytest.raises(ValueError, match="conflicting event_uid"):
        module.compute_hybrid_observability(data)


def test_real_response_nan_fails_but_masked_nan_is_ignored():
    data = batch()
    assert module.compute_hybrid_observability(data)
    data.batch["advantages"][0, 0] = np.nan
    with pytest.raises(ValueError, match="real response token"):
        module.compute_hybrid_observability(data)


def test_sign_flip_and_accidental_zero_recovery_are_visible():
    data = batch()
    data.batch["advantages"][0, :2] *= -1
    data.batch["advantages"][2, :2] = .3
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/solver/advantage/sign_flip_fraction"] == .25
    assert result["hybrid/solver/advantage/base_zero_became_nonzero_count"] == 1
    assert result["hybrid/solver/zero_groups/final_nonzero_count"] == 1


def test_disabled_optional_modules_and_empty_response_rows():
    data = batch()
    data.non_tensor_batch = {key: value for key, value in data.non_tensor_batch.items() if key in ("event_uid", "uid", "agent_id")}
    del data.batch["hybrid_base_advantages"]
    data.batch["response_mask"][0] = 0
    data.batch["advantages"][0] = np.nan
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/solver/empty_response_events"] == 1
    assert result["hybrid/solver/advantage/scaling_changed_fraction"] == 0
    assert not any("semantic/" in key or "stage2/" in key for key in result)


def test_roles_are_summarized_independently_and_input_is_unchanged():
    data = batch()
    data.non_tensor_batch["agent_id"] = np.array(["Solver Agent", "Solver Agent", "Verifier Agent", "Verifier Agent"])
    saved = copy.deepcopy(data)
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/verifier/advantage/base_zero_fraction"] == 1
    assert result["hybrid/solver/advantage/base_zero_fraction"] == 0
    for key, value in data.batch.items():
        np.testing.assert_array_equal(value, saved.batch[key])


def test_cpu_tensor_and_bfloat16_inputs_are_detached():
    torch = pytest.importorskip("torch")
    data = batch()
    data.batch = {key: torch.as_tensor(value).to(torch.bfloat16) for key, value in data.batch.items()}
    data.batch["advantages"].requires_grad_(True)
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/solver/advantage/base_zero_fraction"] == .5
    assert data.batch["advantages"].grad is None


def test_empty_scope_reports_counts_without_fabricating_statistics():
    data = batch()
    data.batch["response_mask"][:] = 0
    data.batch["advantages"][:] = np.nan
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/solver/empty_response_events"] == 4
    assert result["hybrid/solver/zero_groups/count"] == 0
    assert not any("mean" in key or "std" in key or "fraction" in key for key in result)


def test_no_predictions_reports_coverage_without_inventing_delta_statistics():
    data = batch()
    data.non_tensor_batch["value_credit_available"][:] = False
    data.batch["entropy_control_weight"][:] = 0
    result = module.compute_hybrid_observability(data)
    assert result["hybrid/solver/semantic/available_events"] == 0
    assert result["hybrid/solver/semantic/available_given_eligible_fraction"] == 0
    assert "hybrid/solver/semantic/delta_std" not in result
    assert "hybrid/solver/semantic/gated_cap_mean" not in result
