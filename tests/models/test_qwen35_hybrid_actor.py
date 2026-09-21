"""Run the production Actor forward and both hybrid losses on real HF models."""
import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.semantic_entropy_control import action_entropy_hinge_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor

spec = importlib.util.spec_from_file_location("qwen35_tiny_fixture", Path(__file__).with_name("test_qwen3_5_native.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_actor_bf16_ppo_and_independent_entropy_update(moe, device):
    model = fixture.tiny_model(moe).to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    actor = DataParallelPPOActor(OmegaConf.create({
        "use_remove_padding": False, "use_fused_kernels": False,
        "ulysses_sequence_parallel_size": 1, "use_torch_compile": False,
    }), model, optimizer)
    actor.device_name = device
    ids = torch.tensor([[7, 7, 3, 4, 5, 6], [7, 3, 8, 9, 2, 7]], device=device)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]], device=device)
    response_mask = mask[:, -2:]
    batch = {"input_ids": ids, "attention_mask": mask,
             "position_ids": (mask.cumsum(-1) - 1).clamp_min(0), "responses": ids[:, -2:]}
    with torch.no_grad():
        _, old_log_probs = actor._forward_micro_batch(batch, temperature=1., calculate_entropy=False)
    entropy, log_probs = actor._forward_micro_batch(batch, temperature=1., calculate_entropy=True)
    assert entropy.shape == log_probs.shape == (2, 2)
    assert torch.isfinite(entropy).all() and torch.isfinite(log_probs).all()
    # Existing zxj advantages and detached mix scaling enter only the PPO term.
    base_advantage = torch.tensor([[1., 1.], [-1., 0.]], device=device)
    multipliers = torch.tensor([[1.2], [.8]], device=device)
    pg_loss, *_ = compute_policy_loss(old_log_probs, log_probs, base_advantage * multipliers,
                                     response_mask, cliprange=.2, loss_agg_mode="seq-mean-token-mean")
    entropy_loss, _ = action_entropy_hinge_loss(entropy, response_mask,
                                               torch.ones(2, device=device), torch.zeros(2, device=device))
    entropy_gradient = torch.autograd.grad(entropy_loss, model.lm_head.weight, retain_graph=True)[0]
    assert torch.isfinite(entropy_gradient).all() and entropy_gradient.abs().sum() > 0
    before = model.lm_head.weight.detach().clone()
    (pg_loss + .0025 * entropy_loss).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
    assert all(p.grad is None for p in model.model.visual.parameters())
