import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("semantic_metadata_test", Path(__file__).resolve().parents[2] / "verl/utils/semantic_credit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metadata():
    return {"uid": ["q"] * 3, "traj_uid": ["t"] * 3,
            "value_question": ["entire original question"] * 3,
            "value_max_solver_turns": [2] * 3,
            "value_action_index": [0, 1, 2], "role_turn_index": [0, 0, 1],
            "agent_id": ["Solver Agent", "Verifier Agent", "Solver Agent"],
            "value_action_text": ["first", "<verify>reject</verify>", "final"],
            "is_action_valid": [True] * 3, "pass": [1] * 3}


def test_records_need_no_entropy_and_deduplicate_action_identity():
    meta = metadata()
    for values in meta.values():
        values.append(values[-1])
    records, metrics = module.build_trajectory_records(meta, 2)
    assert len(records) == 1 and len(records[0]["actions"]) == 3
    assert metrics["value_credit/padding_rows"] == 1
    assert all("entropy_mean" not in action for action in records[0]["actions"])
    assert records[0]["question"] == "entire original question"


def test_terminal_semantic_prediction_remains_nonzero_but_control_is_closed():
    meta = metadata()
    module.attach_value_predictions(meta, {"t": [{"sem": p} for p in [.7, .6, .5, .9]]}, True)
    np.testing.assert_allclose(meta["value_credit_delta"], [-.1, -.1, .4])
    assert meta["value_sem_after"][-1] == .9
    np.testing.assert_array_equal(meta["value_control_eligible"], [True, True, False])
    assert meta["value_credit_available"].all()
    assert "value_entropy_delta" not in meta


def test_initial_calibration_eligibility_does_not_require_ready_predictions():
    meta = metadata()
    module.attach_value_predictions(meta, {}, False)
    np.testing.assert_array_equal(meta["value_control_eligible"], [True, True, False])
    assert not meta["value_credit_available"].any()


def test_truncation_and_invalid_actions_only_mask_own_control():
    meta = metadata()
    meta["value_action_truncated"] = [True, False, False]
    module.attach_value_predictions(meta, {"t": [{"sem": .5}] * 4}, True)
    np.testing.assert_array_equal(meta["value_control_eligible"], [False, True, False])
    records, _ = module.build_trajectory_records(metadata(), 2)
    assert records[0]["actions"][0]["text"] == "first"


@pytest.mark.parametrize("change", ["gap", "budget", "early_approve", "conflict"])
def test_incomplete_or_conflicting_histories_are_rejected(change):
    meta = metadata()
    if change == "gap":
        meta["value_action_index"][-1] = 4
    elif change == "budget":
        meta["value_max_solver_turns"][-1] = 3
    elif change == "early_approve":
        meta["value_action_text"][1] = "<verify>approve</verify>"
    else:
        for values in meta.values():
            values.append(values[-1])
        meta["value_action_text"][-1] = "conflicting copy"
    records, metrics = module.build_trajectory_records(meta, 2)
    assert not records and metrics["value_credit/skipped_trajectories"] == 1


def test_bad_optional_entropy_metadata_cannot_block_semantic_records():
    a, b = metadata(), metadata()
    b["top16_entropy_mean"] = [float("nan"), "malformed", -500]
    b["top16_entropy"] = [None, {"coverage": "broken"}, 2]
    assert module.build_trajectory_records(a, 2) == module.build_trajectory_records(b, 2)


def test_offline_cli_accepts_no_entropy_and_rejects_partial(tmp_path):
    import json
    cli_spec = importlib.util.spec_from_file_location("semantic_pretrain_test", Path(__file__).resolve().parents[2] / "examples/drmas_trainer/pretrain_semantic_value.py")
    cli = importlib.util.module_from_spec(cli_spec)
    cli_spec.loader.exec_module(cli)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"non_tensor_batch": metadata()}))
    records, provenance = cli.load_records([str(path)])
    assert len(records) == 1 and provenance["unique_questions"] == 1
    assert cli.main(["--input", str(path), "--output", str(tmp_path / "unused.pt"), "--validate-only"]) == 0
    bad = metadata()
    for values in bad.values():
        values.pop()
    path.write_text(json.dumps({"non_tensor_batch": bad}))
    with pytest.raises(ValueError, match="incomplete"):
        cli.load_records([str(path)])
