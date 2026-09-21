import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "entropy_credit.py"
_SPEC = importlib.util.spec_from_file_location("drmas_entropy_credit", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_action_topk_entropy = _MODULE.compute_action_topk_entropy
compute_entropy_credit_multipliers = _MODULE.compute_entropy_credit_multipliers
first_unique_action_indices = _MODULE.first_unique_action_indices


def test_top16_entropy_uniform_concentrated_and_translation_invariant():
    uniform = compute_action_topk_entropy([[0.0] * 16], top_k=16)
    concentrated = compute_action_topk_entropy([[0.0] + [-100.0] * 15], top_k=16)
    values = np.linspace(-8.0, -0.5, 16).tolist()
    original = compute_action_topk_entropy([values], top_k=16)
    translated = compute_action_topk_entropy([[value + 123.0 for value in values]], top_k=16)

    assert uniform["mean"] == 1.0
    assert uniform["effective_support"] == 16.0
    assert concentrated["mean"] < 1e-12
    assert abs(concentrated["effective_support"] - 1.0) < 1e-12
    assert abs(original["mean"] - translated["mean"]) < 1e-12


def test_top16_entropy_sglang_format_mask_and_missing_position():
    complete = [(value, token_id, None) for token_id, value in enumerate(np.linspace(-4.0, -0.1, 16))]
    incomplete = [(0.0, token_id, None) for token_id in range(15)]
    stats = compute_action_topk_entropy(
        [complete, incomplete, complete],
        response_mask=[True, True, False],
        top_k=16,
    )

    assert stats["num_tokens"] == 1
    assert stats["valid_response_tokens"] == 2
    assert stats["coverage"] == 0.5
    assert compute_action_topk_entropy([incomplete], top_k=16) is None


def test_action_credit_has_opposite_success_and_failure_directions():
    result = compute_entropy_credit_multipliers(
        prompt_group_ids=["p"] * 4,
        trajectory_ids=["success-low", "success-high", "failure-low", "failure-high"],
        agent_ids=["Solver Agent"] * 4,
        role_turn_indices=[0] * 4,
        terminal_success=[True, True, False, False],
        action_entropies=[0.1, 0.9, 0.1, 0.9],
        action_scale=0.2,
        trajectory_scale=0.0,
    )

    np.testing.assert_allclose(result["action"], [0.8, 1.2, 1.2, 0.8])


def test_tied_and_singleton_action_groups_are_neutral():
    result = compute_entropy_credit_multipliers(
        prompt_group_ids=["p", "p", "p"],
        trajectory_ids=["success-a", "success-b", "failure-only"],
        agent_ids=["Verifier Agent"] * 3,
        role_turn_indices=[0, 0, 0],
        terminal_success=[True, True, False],
        action_entropies=[0.5, 0.5, 0.7],
    )

    np.testing.assert_allclose(result["action"], np.ones(3))


def test_trajectory_credit_uses_explicit_same_role_turn_order_only():
    # Rows are deliberately shuffled.  Each role and trajectory must remain
    # independent, and every first role action must stay neutral.
    fields = {
        "prompt_group_ids": np.asarray(["p"] * 6, dtype=object),
        "trajectory_ids": np.asarray(["t1", "t1", "t2", "t1", "t2", "t1"], dtype=object),
        "agent_ids": np.asarray(
            ["Solver Agent", "Verifier Agent", "Solver Agent", "Solver Agent", "Solver Agent", "Verifier Agent"],
            dtype=object,
        ),
        "role_turn_indices": np.asarray([1, 0, 1, 0, 0, 1]),
        "terminal_success": np.asarray([True] * 6),
        "action_entropies": np.asarray([0.4, 0.8, 0.3, 0.2, 0.9, 0.4]),
    }
    result = compute_entropy_credit_multipliers(
        **fields,
        action_scale=0.0,
        trajectory_scale=0.2,
        trajectory_deadzone=0.0,
    )

    np.testing.assert_allclose(result["trajectory"], [1.2, 1.0, 0.8, 1.0, 1.0, 0.8])

    permutation = np.asarray([5, 2, 4, 0, 3, 1])
    reordered = compute_entropy_credit_multipliers(
        **{key: value[permutation] for key, value in fields.items()},
        action_scale=0.0,
        trajectory_scale=0.2,
        trajectory_deadzone=0.0,
    )
    np.testing.assert_allclose(result["trajectory"], reordered["trajectory"][np.argsort(permutation)])


def test_success_failure_entropy_change_quadrants_and_deadzone():
    result = compute_entropy_credit_multipliers(
        prompt_group_ids=["p"] * 8,
        trajectory_ids=[
            "success-rise", "success-rise",
            "success-fall", "success-fall",
            "failure-rise", "failure-rise",
            "failure-fall", "failure-fall",
        ],
        agent_ids=["Solver Agent"] * 8,
        role_turn_indices=[0, 1] * 4,
        terminal_success=[True, True, True, True, False, False, False, False],
        action_entropies=[0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1],
        action_scale=0.0,
        trajectory_scale=0.2,
        trajectory_deadzone=0.05,
    )

    np.testing.assert_allclose(
        result["trajectory"],
        [1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 1.0, 1.2],
    )

    deadzone = compute_entropy_credit_multipliers(
        prompt_group_ids=["p", "p"],
        trajectory_ids=["t", "t"],
        agent_ids=["Search Agent", "Search Agent"],
        role_turn_indices=[0, 1],
        terminal_success=[True, True],
        action_entropies=[0.50, 0.52],
        action_scale=0.0,
        trajectory_scale=0.2,
        trajectory_deadzone=0.05,
    )
    np.testing.assert_allclose(deadzone["trajectory"], [1.0, 1.0])


def test_invalid_or_missing_entropy_is_neutral_and_breaks_local_transition():
    result = compute_entropy_credit_multipliers(
        prompt_group_ids=["p"] * 3,
        trajectory_ids=["t"] * 3,
        agent_ids=["Verifier Agent"] * 3,
        role_turn_indices=[0, 1, 2],
        terminal_success=[True] * 3,
        action_entropies=[0.1, 0.5, 0.9],
        action_valid=[True, False, True],
        action_scale=0.2,
        trajectory_scale=0.2,
        trajectory_deadzone=0.0,
    )

    assert result["action"][1] == 1.0
    assert result["trajectory"][1] == 1.0
    # Turn 2 compares only with the immediately preceding invalid turn; it must
    # not skip backward to turn 0.
    assert result["trajectory"][2] == 1.0


def test_third_clip_limits_product_of_two_individually_clipped_factors():
    result = compute_entropy_credit_multipliers(
        prompt_group_ids=["rise", "rise", "fall", "fall"],
        trajectory_ids=["r", "r", "f", "f"],
        agent_ids=["Solver Agent"] * 4,
        role_turn_indices=[0, 1, 0, 1],
        terminal_success=[True] * 4,
        action_entropies=[0.1, 0.9, 0.9, 0.1],
        action_scale=0.2,
        trajectory_scale=0.2,
        trajectory_deadzone=0.0,
        multiplier_min=0.8,
        multiplier_max=1.2,
        final_multiplier_min=0.8,
        final_multiplier_max=1.2,
    )

    assert result["action"][1] == 1.2
    assert result["trajectory"][1] == 1.2
    assert result["final"][1] == 1.2
    assert result["action"][3] == 0.8
    assert result["trajectory"][3] == 0.8
    assert result["final"][3] == 0.8


def test_trajectory_export_deduplicates_adjust_batch_copies_only():
    keep = first_unique_action_indices(
        trajectory_ids=["t1", "t1", "t2", "t1", "t2", "t1", "t1"],
        agent_ids=["Solver Agent", "Verifier Agent", "Solver Agent", "Solver Agent", "Solver Agent", "Verifier Agent", "Solver Agent"],
        role_turn_indices=[0, 0, 0, 1, 0, 0, 0],
    )

    # Turn 1 is a distinct action; rows 4-6 are copies of rows 2, 1, and 0.
    np.testing.assert_array_equal(keep, [0, 1, 2, 3])


# Hybrid obtains role_turn_index from the audited zxj role_event_index.
# Rollout identity is covered in test_hybrid_rollout_metadata and
# test_hybrid_credit_integration, replacing mix BaseOrchestra counters.
