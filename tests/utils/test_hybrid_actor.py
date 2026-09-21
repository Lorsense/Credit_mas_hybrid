"""Execute actual Actor methods on CPU without importing Ray/FSDP backends.

Only batching and the PPO primitive are substituted; the production update
method and production entropy loss execute unchanged with real torch autograd.
"""

import ast
import importlib.util
import itertools
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest


torch = pytest.importorskip("torch")
_ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location("actor_test_entropy_control", _ROOT / "verl/utils/semantic_entropy_control.py")
_CONTROL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTROL)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Batch(dict):
    def __len__(self):
        return next(iter(self.values())).shape[0]

    def split(self, size):
        return [Batch({key: value[start:start + size] for key, value in self.items()})
                for start in range(0, len(self), size)]

    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class Proto:
    def __init__(self, batch, meta_info=None, non_tensor_batch=None):
        self.batch = batch
        self.meta_info = {} if meta_info is None else meta_info
        self.non_tensor_batch = {} if non_tensor_batch is None else non_tensor_batch

    def select(self, batch_keys=None, **kwargs):
        return Proto(Batch({key: self.batch[key] for key in batch_keys}), self.meta_info, self.non_tensor_batch)

    def __len__(self):
        return len(self.batch)


def append_to_dict(target, values):
    for key, value in values.items():
        target.setdefault(key, []).append(value)


def actor_method(name, **overrides):
    tree = ast.parse((_ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    environment = {
        "torch": torch, "np": np, "DataProto": Proto, "itertools": itertools,
        "Tuple": Tuple,
        "get_torch_device": lambda: SimpleNamespace(current_device=lambda: torch.device("cpu")),
        "action_entropy_hinge_loss": _CONTROL.action_entropy_hinge_loss,
        "append_to_dict": append_to_dict,
        "get_reverse_idx": lambda indices: sorted(range(len(indices)), key=indices.__getitem__),
    }
    environment.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "isolated_actor_method", "exec"), environment)
    return environment[name]


def test_dynamic_log_prob_restores_entropy_and_log_prob_to_same_action_order():
    original = Batch({key: torch.arange(3).reshape(3, 1) for key in
                      ("responses", "input_ids", "attention_mask", "position_ids")})

    def repack(batch, max_token_len):
        assert max_token_len == 100
        return [Batch({key: value[[2, 0]] for key, value in batch.items()}),
                Batch({key: value[[1]] for key, value in batch.items()})], [[2, 0], [1]]

    method = actor_method("compute_log_prob", rearrange_micro_batches=repack)
    actor = SimpleNamespace(actor_module=SimpleNamespace(eval=lambda: None), ulysses_sequence_parallel_size=1)
    actor._forward_micro_batch = lambda batch, **kwargs: (batch["responses"] + 100., batch["responses"] + 10.)
    data = Proto(original, {"micro_batch_size": 2, "temperature": 1., "use_dynamic_bsz": True, "max_token_len": 100})
    log_probs, entropies = method(actor, data, calculate_entropy=True)
    assert log_probs[:, 0].tolist() == [10., 11., 12.]
    assert entropies[:, 0].tolist() == [100., 101., 102.]


def test_dense_forward_uses_full_vocabulary_entropy_and_safe_shared_backward():
    calls = []
    logits = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).repeat(1, 5, 1) / 10.)

    def logprobs(scores, responses, inplace_backward=True):
        calls.append(inplace_backward)
        return scores.log_softmax(-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)

    entropy_fn = lambda scores: -(scores.softmax(-1) * scores.log_softmax(-1)).sum(-1)
    actor = SimpleNamespace(
        actor_module=lambda **kwargs: SimpleNamespace(logits=logits.clone()),
        use_remove_padding=False, use_fused_kernels=False, device_name="cpu")
    micro_batch = {key: torch.ones(1, 5, dtype=torch.long) for key in ("input_ids", "attention_mask", "position_ids")}
    micro_batch["responses"] = torch.zeros(1, 2, dtype=torch.long)
    entropy, log_probs = actor_method(
        "_forward_micro_batch", logprobs_from_logits=logprobs,
        verl_F=SimpleNamespace(entropy_from_logits=entropy_fn),
    )(actor, micro_batch, temperature=1., calculate_entropy=True)
    assert calls == [False]
    assert entropy.shape == (1, 2)
    torch.testing.assert_close(entropy, entropy_fn(logits[:, 2:4]))
    assert entropy.mean() > 1  # Full entropy in nats, not normalized Top-16.
    (entropy.mean() + log_probs.mean()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def run_update(enabled, *, gate=1., advantage=0., micro_size=1, valid=(True, True)):
    logits = torch.nn.Parameter(torch.tensor([2., 0., -1.]))
    model = torch.nn.ParameterList([logits])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    forward_requests = []
    ppo_advantages = []

    def forward(micro_batch, temperature, calculate_entropy):
        forward_requests.append(calculate_entropy)
        n = len(micro_batch["responses"])
        log_prob = logits.log_softmax(-1)[0].expand(n, 2)
        entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum().expand(n, 2) if calculate_entropy else None
        return entropy, log_prob

    def policy_loss(**kwargs):
        ppo_advantages.append(kwargs["advantages"].clone())
        loss = -(kwargs["log_prob"] * kwargs["advantages"] * kwargs["response_mask"]).mean()
        return loss, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

    actor = SimpleNamespace(
        actor_module=model, actor_optimizer=optimizer, _forward_micro_batch=forward,
        config=Config(use_kl_loss=False, use_adaptive_ppo_mini_batch_size=False,
                      ppo_mini_batch_size=2, ppo_micro_batch_size_per_gpu=micro_size,
                      ppo_epochs=1, use_dynamic_bsz=False, clip_ratio=.2, clip_ratio_low=None,
                      clip_ratio_high=None, entropy_coeff=0., loss_agg_mode="token-mean",
                      entropy_control={"enabled": enabled, "loss_coef": .1}),
    )

    def optimizer_step():
        norm = logits.grad.norm().detach()
        optimizer.step()
        return norm

    actor._optimizer_step = optimizer_step
    batch = Batch({key: torch.ones(2, 2) for key in
                   ("responses", "input_ids", "attention_mask", "position_ids", "old_log_probs")})
    batch["advantages"] = torch.full((2, 2), advantage)
    if enabled:
        batch.update(entropy_control_weight=torch.full((2,), gate),
                     entropy_control_cap=torch.zeros(2),
                     entropy_control_valid=torch.tensor(valid))
    initial = logits.detach().clone()
    result = actor_method("update_policy", compute_policy_loss=policy_loss)(
        actor, Proto(batch, {"temperature": 1.}, {"wg_id": ["solver", "solver"]}))
    return initial, logits.detach().clone(), result, forward_requests, ppo_advantages


def test_actor_zero_advantage_still_allows_independent_entropy_gradient():
    initial, final, metrics, requests, advantages = run_update(True)
    assert requests == [True, True]
    assert not torch.equal(initial, final)
    assert all(not item.any() for item in advantages)
    old_h = -(initial.softmax(-1) * initial.log_softmax(-1)).sum()
    new_h = -(final.softmax(-1) * final.log_softmax(-1)).sum()
    assert new_h < old_h
    assert metrics["actor/solver/entropy_control/hinge_loss"][0] > 0


def test_actor_disabled_path_needs_no_control_fields_or_entropy_forward():
    initial, final, metrics, requests, _ = run_update(False)
    assert torch.equal(initial, final)
    assert requests == [False, False]
    assert not any("entropy_control" in key for key in metrics)


def test_actor_zero_gate_preserves_original_ppo_update():
    _, original, _, _, _ = run_update(False, advantage=.7)
    _, controlled, _, _, _ = run_update(True, gate=0., advantage=.7)
    torch.testing.assert_close(original, controlled, rtol=0, atol=0)


def test_actor_entropy_term_is_invariant_to_microbatch_partition_with_invalid_padding():
    _, small, _, _, _ = run_update(True, micro_size=1, valid=(True, False))
    _, large, _, _, _ = run_update(True, micro_size=2, valid=(True, False))
    torch.testing.assert_close(small, large, rtol=0, atol=0)


def duplicate_update(indices, micro_size, enabled=True, advantage=0.):
    """Run the real update method with UID-coherent padded event copies."""
    logits = torch.nn.Parameter(torch.tensor([[2., 0., -1.], [.2, .1, -.1]]))
    model = torch.nn.ParameterList([logits])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    index = torch.tensor(indices)
    rows = len(indices)
    copies = torch.bincount(index).float()
    sample_weight = 1. / copies[index]

    def forward(micro_batch, temperature, calculate_entropy):
        selected = logits[micro_batch["input_ids"][:, 0].long()]
        log_prob = selected.log_softmax(-1)[:, 0, None].expand(-1, 2)
        entropy = (-(selected.softmax(-1) * selected.log_softmax(-1)).sum(-1)[:, None].expand(-1, 2)
                   if calculate_entropy else None)
        return entropy, log_prob

    def normalizer(loss_mask, sample_weight, loss_agg_mode):
        counts = loss_mask.sum(-1) if loss_agg_mode == "token-mean" else loss_mask.bool().any(-1)
        return (counts * sample_weight).sum()

    def policy_loss(**kwargs):
        # Zero PPO advantages isolate the added entropy objective. Nonzero
        # advantages exercise the preservation of zxj's weighted event mean.
        per_event = -(kwargs["log_prob"] * kwargs["advantages"]).mean(-1)
        loss = (per_event * kwargs["sample_weight"]).sum()
        loss = loss / kwargs["sample_weight_normalizer"] * kwargs["sample_weight_scale"]
        return loss, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

    actor = SimpleNamespace(
        actor_module=model, actor_optimizer=optimizer, _forward_micro_batch=forward,
        config=Config(use_kl_loss=False, use_adaptive_ppo_mini_batch_size=False,
                      ppo_mini_batch_size=rows, ppo_micro_batch_size_per_gpu=micro_size,
                      ppo_epochs=1, ppo_mini_update_num=1, use_dynamic_bsz=False,
                      clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None,
                      entropy_coeff=0., loss_agg_mode="seq-mean-token-mean",
                      entropy_control={"enabled": enabled, "loss_coef": .1}),
    )

    def optimizer_step():
        norm = logits.grad.norm().detach()
        optimizer.step()
        return norm

    actor._optimizer_step = optimizer_step
    batch = Batch({key: torch.ones(rows, 2) for key in
                   ("responses", "attention_mask", "position_ids", "old_log_probs")})
    batch.update(input_ids=index[:, None].expand(-1, 2),
                 advantages=torch.full((rows, 2), advantage), sample_weight=sample_weight,
                 optimizer_mini_batch_id=torch.zeros(rows, dtype=torch.long))
    if enabled:
        batch.update(entropy_control_weight=torch.tensor([.8, .3])[index],
                     entropy_control_cap=torch.tensor([.1, .6])[index],
                     entropy_control_valid=torch.ones(rows, dtype=torch.bool))
    data = Proto(batch, {"temperature": 1., "requires_duplicate_aware_weighting": True,
                         "duplicate_aware_optimizer_mini_batch_count": 1,
                         "duplicate_aware_optimizer_schedule": "uid_coherent_v1"},
                 {"wg_id": np.array(["solver"] * rows),
                  "event_uid": np.array([f"event-{i}" for i in indices])})
    metrics = actor_method("update_policy", compute_policy_loss=policy_loss,
                           compute_sample_weight_normalizer=normalizer)(actor, data)
    return logits.detach(), metrics


def test_duplicate_weighting_and_fractional_microbatches_preserve_unique_event_gradient():
    original, original_metrics = duplicate_update([0, 1], 2)
    # Every microbatch contains weight 1/3 or 1/2: clamping its denominator to
    # one would incorrectly weaken the added regularizer.
    repeated, repeated_metrics = duplicate_update([0, 0, 0, 1, 1], 1)
    torch.testing.assert_close(original, repeated, rtol=1e-6, atol=1e-7)
    assert original_metrics["actor/solver/entropy_control/valid_actions"] == [2.]
    assert repeated_metrics["actor/solver/entropy_control/valid_actions"] == pytest.approx([2.])
    assert repeated_metrics["actor/solver/entropy_control/hinge_loss"] == pytest.approx(
        original_metrics["actor/solver/entropy_control/hinge_loss"])


def test_duplicate_aware_ppo_is_unchanged_when_control_disabled():
    original, _ = duplicate_update([0, 1], 2, enabled=False, advantage=.7)
    repeated, metrics = duplicate_update([0, 0, 0, 1, 1], 1, enabled=False, advantage=.7)
    torch.testing.assert_close(original, repeated, rtol=1e-6, atol=1e-7)
    assert not any("entropy_control" in name for name in metrics)


def test_distributed_weighted_control_matches_full_action_mean():
    entropy = torch.tensor([[1.8, 2.2], [.8, 1.0]], requires_grad=True)
    reference, _ = _CONTROL.action_entropy_hinge_loss(
        entropy, torch.ones_like(entropy), torch.tensor([.8, .3]), torch.tensor([.1, .6]))
    reference_grad = torch.autograd.grad(reference, entropy)[0]
    total = entropy.sum() * 0.
    # DP rank 0 holds 2 copies of event 0, rank 1 one copy of each event.
    for rank_indices in ([0, 0], [0, 1]):
        for i in rank_indices:
            weight = torch.tensor([1./3 if i == 0 else 1.])
            loss, metrics = _CONTROL.action_entropy_hinge_loss(
                entropy[[i]], torch.ones_like(entropy[[i]]),
                torch.tensor([.8 if i == 0 else .3]), torch.tensor([.1 if i == 0 else .6]),
                sample_weight=weight)
            # Actor uses global weighted count 2, compensates world size 2;
            # FSDP then averages the two rank gradients.
            total = total + loss * metrics["valid_actions"] / 2 * 2 / 2
    torch.testing.assert_close(torch.autograd.grad(total, entropy)[0], reference_grad)
