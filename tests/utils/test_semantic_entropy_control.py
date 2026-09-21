import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("semantic_controller_test", Path(__file__).resolve().parents[2] / "verl/utils/semantic_entropy_control.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metadata():
    return {"agent_id": np.array(["solver"] * 4), "traj_uid": np.array(["a", "b", "c", "d"]),
            "role_turn_index": np.array([0] * 4), "is_action_valid": np.ones(4, bool),
            "pure_entropy_response_tokens": np.array([3] * 4),
            "value_credit_delta": np.array([-.2] * 4), "value_credit_available": np.ones(4, bool),
            "value_control_eligible": np.ones(4, bool)}


def controller(**kwargs):
    return module.SemanticEntropyController({"enabled": True, "min_calibration_actions": 2,
                                             "ramp_steps": 1, "min_group_actions": 2,
                                             "cap_margin": .1, **kwargs})


def test_initial_caps_calibrate_before_semantic_head_qualifies():
    c, meta = controller(), metadata()
    meta["value_credit_available"][:] = False
    first, _ = c.prepare(meta, np.ones(4), step=1, ready=False, reliability={"solver": 0})
    assert not first["entropy_control_weight"].any()
    assert c.groups[("solver", 0)]["cap"] == pytest.approx(1.1)
    meta["value_credit_available"][:] = True
    output, metrics = c.prepare(meta, np.ones(4) * 2, step=2, ready=True, reliability={"solver": 1})
    assert np.all(output["entropy_control_weight"] == 1)
    assert metrics["entropy_control/valid_unique_actions"] == 4


def test_semantic_delta_alone_drives_gate_no_entropy_branch_permission():
    c, meta = controller(), metadata()
    c.prepare(meta, np.ones(4), 1, True, {"solver": 1})
    meta["value_credit_delta"] = np.array([-.2, -.07, .2, -.01])
    output, _ = c.prepare(meta, np.ones(4) * 2, 2, True, {"solver": 1})
    np.testing.assert_allclose(output["entropy_control_weight"], [1, .5, 0, 0], atol=1e-6)


def test_terminal_actions_never_calibrate_or_control():
    c, meta = controller(), metadata()
    meta["value_control_eligible"][:] = False
    output, metrics = c.prepare(meta, np.ones(4) * 5, 1, True, {"solver": 1})
    assert not output["entropy_control_valid"].any()
    assert not output["entropy_control_weight"].any()
    assert metrics["entropy_control/calibrated_groups"] == 0


def test_duplicate_statistics_count_once_but_actor_copies_get_same_gate():
    c, meta = controller(), metadata()
    for key, value in list(meta.items()):
        meta[key] = np.concatenate([value, value[:1]])
    c.prepare(meta, np.ones(5), 1, False, {"solver": 0})
    assert c.groups[("solver", 0)]["count"] == 4
    output, metrics = c.prepare(meta, np.ones(5) * 2, 2, True, {"solver": 1})
    assert output["entropy_control_valid"].all()
    assert output["entropy_control_weight"][0] == output["entropy_control_weight"][-1] == 1
    assert metrics["entropy_control/valid_unique_actions"] == 4


def test_conflicting_duplicate_action_is_rejected():
    c, meta = controller(), metadata()
    meta["traj_uid"][1] = "a"
    meta["value_credit_delta"][1] = -.9
    with pytest.raises(ValueError, match="conflicting"):
        c.prepare(meta, np.ones(4), 1, True, {"solver": 1})


def test_ramp_resets_when_role_loses_semantic_qualification():
    c, meta = controller(ramp_steps=2), metadata()
    c.prepare(meta, np.ones(4), 1, False, {"solver": 0})
    output, _ = c.prepare(meta, np.ones(4) * 2, 2, True, {"solver": 1})
    np.testing.assert_allclose(output["entropy_control_weight"], .5)
    output, _ = c.prepare(meta, np.ones(4) * 2, 3, True, {"solver": 0})
    assert not output["entropy_control_weight"].any()
    output, _ = c.prepare(meta, np.ones(4) * 2, 4, True, {"solver": 1})
    np.testing.assert_allclose(output["entropy_control_weight"], .5)


def test_checkpoint_restores_caps_ramps_and_requires_config_and_schema():
    c, meta = controller(), metadata()
    c.prepare(meta, np.ones(4), 1, True, {"solver": 1})
    state = c.state_dict()
    restored = controller()
    restored.load_state_dict(state)
    a, am = c.prepare(meta, np.ones(4) * 2, 2, True, {"solver": 1})
    b, bm = restored.prepare(meta, np.ones(4) * 2, 2, True, {"solver": 1})
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert am == bm
    with pytest.raises(ValueError, match="config mismatch"):
        controller(ramp_steps=3).load_state_dict(state)
    bad = copy.deepcopy(state)
    bad["schema"] = "old-mix"
    with pytest.raises(ValueError, match="version"):
        controller().load_state_dict(bad)


def test_uncalibrated_late_group_cannot_redefine_cap():
    c, meta = controller(), metadata()
    c.prepare(meta, np.ones(4), 1, False, {"solver": 0})
    meta["role_turn_index"][:] = 1
    output, _ = c.prepare(meta, np.ones(4) * 10, 2, True, {"solver": 1})
    assert not output["entropy_control_weight"].any()
    assert c.groups[("solver", 1)]["cap"] is None


def test_real_autograd_hinge_detaches_gate_and_caps_and_masks_padding():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[1., 0., -1.], [0., 1., -1.]], [[0., 0., 0.], [0., 0., 0.]]], requires_grad=True)
    probs = logits.softmax(-1)
    entropy = -(probs * logits.log_softmax(-1)).sum(-1)
    weights = torch.tensor([1., 1.], requires_grad=True)
    caps = torch.tensor([.1, .1], requires_grad=True)
    loss, metrics = module.action_entropy_hinge_loss(entropy, torch.tensor([[1, 1], [0, 0]]), weights, caps)
    loss.backward()
    assert loss > 0 and metrics["valid_actions"] == 1
    assert logits.grad[0].abs().sum() > 0 and logits.grad[1].abs().sum() == 0
    assert weights.grad is None and caps.grad is None


def test_fractional_sample_weights_produce_correct_microbatch_gradient():
    torch = pytest.importorskip("torch")
    entropy = torch.tensor([[2., 4.]], requires_grad=True)
    sample_weight = torch.tensor([.25])
    loss, metrics = module.action_entropy_hinge_loss(entropy, torch.ones_like(entropy), torch.ones(1), torch.ones(1), sample_weight=sample_weight)
    assert loss == 2 and metrics["valid_actions"] == .25
    # Actor scales micro loss by weighted micro count / full mini denominator.
    (loss * metrics["valid_actions"] / 2).backward()
    torch.testing.assert_close(entropy.grad, torch.full_like(entropy, .0625))


def test_zero_valid_entropy_is_differentiable_zero_even_with_nan_padding():
    torch = pytest.importorskip("torch")
    entropy = torch.tensor([[float("nan"), float("nan")]], requires_grad=True)
    loss, _ = module.action_entropy_hinge_loss(entropy, torch.zeros_like(entropy), torch.ones(1), torch.zeros(1))
    assert loss == 0
    loss.backward()
    assert torch.equal(entropy.grad, torch.zeros_like(entropy))
