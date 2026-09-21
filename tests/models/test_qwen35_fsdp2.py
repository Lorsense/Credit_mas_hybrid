"""Single-GPU production FSDP2 path, including native gradient checkpointing.

Capability probes may skip platforms whose distributed build lacks CUDA
collectives. No model/load/forward/backward exception is converted to a skip.
"""
from datetime import timedelta
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.checkpoint.fsdp_checkpoint_manager import _gather_full_model_state
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2, fsdp2_load_full_state_dict
from verl.utils.semantic_entropy_control import action_entropy_hinge_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor

spec = importlib.util.spec_from_file_location("qwen35_fsdp_tiny", Path(__file__).with_name("test_qwen3_5_native.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.fixture
def cuda_collectives():
    if not torch.cuda.is_available():
        pytest.skip("single-GPU FSDP2 integration requires CUDA")
    if not dist.is_available() or not (dist.is_nccl_available() or dist.is_gloo_available()):
        pytest.skip("this PyTorch build has neither NCCL nor Gloo")
    if dist.is_initialized():
        pytest.fail("FSDP2 integration needs an isolated process group")
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    try:
        try:
            # TCPStore without libuv supports Windows builds that lack libuv.
            store = dist.TCPStore("127.0.0.1", 0, 1, True, timeout=timedelta(seconds=20), use_libuv=False)
            dist.init_process_group(backend, store=store, rank=0, world_size=1, timeout=timedelta(seconds=20))
            probe = torch.ones(1, device="cuda")
            dist.broadcast(probe, src=0)
            dist.all_reduce(probe)
            dist.all_gather_into_tensor(torch.empty_like(probe), probe)
            dist.reduce_scatter_tensor(torch.empty_like(probe), probe)
            torch.cuda.synchronize()
        except (RuntimeError, NotImplementedError) as error:
            # This block contains only backend initialization/collective probes,
            # so a skip cannot conceal a Qwen/FSDP/model implementation error.
            pytest.skip(f"{backend} CUDA collective capability unavailable: {error}")
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("moe", [False, True])
def test_fsdp2_gradcheckpoint_actor_update_and_full_hf_export(moe, cuda_collectives, tmp_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    model = fixture.tiny_model(moe)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    config = model.config
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                          cast_forward_inputs=True),
        "offload_policy": None,
        "reshard_after_forward": True,
    }
    apply_fsdp2(model, fsdp_kwargs, OmegaConf.create({"wrap_policy": {}}))
    fsdp2_load_full_state_dict(model, original, mesh)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    actor_config = OmegaConf.create({"use_remove_padding": False, "use_fused_kernels": False,
                                     "ulysses_sequence_parallel_size": 1, "use_torch_compile": False})
    actor = DataParallelPPOActor(actor_config, model, optimizer)
    actor.device_name = "cuda"
    ids = torch.tensor([[7, 7, 3, 4, 5, 6], [7, 3, 8, 9, 2, 7]], device="cuda")
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]], device="cuda")
    batch = {"input_ids": ids, "attention_mask": mask,
             "position_ids": (mask.cumsum(-1) - 1).clamp_min(0), "responses": ids[:, -2:]}
    with torch.no_grad():
        _, old_log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
    entropy, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=True)
    assert torch.isfinite(entropy).all() and torch.isfinite(log_probs).all()
    advantages = torch.tensor([[1.2, 1.2], [-.8, 0]], device="cuda")
    ppo_loss, *_ = compute_policy_loss(old_log_probs, log_probs, advantages, mask[:, -2:],
                                      cliprange=.2, loss_agg_mode="seq-mean-token-mean")
    entropy_loss, _ = action_entropy_hinge_loss(entropy, mask[:, -2:], torch.ones(2, device="cuda"),
                                               torch.zeros(2, device="cuda"))
    loss = ppo_loss + .0025 * entropy_loss
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g.to_local() if isinstance(g, DTensor) else g).all() for g in gradients)
    assert all(p.grad is None and not p.requires_grad for p in model.model.visual.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Exercise the production FSDP2 export helper: never serialize DTensors as
    # an HF checkpoint, even on a one-rank mesh.
    exported = _gather_full_model_state(model)
    assert exported and all(isinstance(value, torch.Tensor) and not isinstance(value, DTensor)
                            and value.device.type == "cpu" for value in exported.values())
    assert not torch.equal(exported["lm_head.weight"], original["lm_head.weight"])
    unsharded = fixture.adapter.select_auto_model_class(config).from_config(config)
    unsharded.load_state_dict(exported, strict=True)
    unsharded.save_pretrained(tmp_path)
    reloaded, info = fixture.adapter.select_auto_model_class(config).from_pretrained(
        tmp_path, config=config, dtype=torch.float32, output_loading_info=True)
    fixture.adapter.validate_qwen3_5_loading_info(info)
    fixture.adapter.prepare_qwen3_5_for_text_training(reloaded)
    for name, value in reloaded.state_dict().items():
        torch.testing.assert_close(value, exported[name], rtol=0, atol=0)
    model.eval()
    reloaded = reloaded.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(input_ids=ids, attention_mask=mask, position_ids=batch["position_ids"], use_cache=False).logits
        expected = reloaded(input_ids=ids, attention_mask=mask, position_ids=batch["position_ids"], use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
