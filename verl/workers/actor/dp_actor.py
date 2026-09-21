# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_sample_weight_normalizer, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.semantic_entropy_control import action_entropy_hinge_loss
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

# Padded Qwen3.5 training uses native HF attention and does not need FlashAttention.
# Load the packing dependency only when that execution path is requested.
def _load_padding_ops():
    global index_first_axis, pad_input, rearrange, unpad_input
    if is_cuda_available:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    elif is_npu_available:
        from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
    else:
        raise RuntimeError("remove-padding requires a supported FlashAttention backend")


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if self.use_remove_padding:
            _load_padding_ops()
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    # Entropy and log-probability share logits in backward.
                    # The CE kernel must not overwrite the entropy input.
                    log_probs = logprobs_from_logits(
                        logits, micro_batch["responses"], inplace_backward=not calculate_entropy
                    )
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        wg_id = data.non_tensor_batch['wg_id'][0]

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        requires_duplicate_aware = bool(data.meta_info.get("requires_duplicate_aware_weighting", False))
        duplicate_aware = "sample_weight" in data.batch.keys()
        if requires_duplicate_aware and not duplicate_aware:
            raise ValueError(
                "This actor update requires duplicate-aware event weighting, but sample_weight is missing"
            )
        expected_optimizer_minis = 1
        if duplicate_aware:
            expected_optimizer_minis = int(
                data.meta_info.get("duplicate_aware_optimizer_mini_batch_count", -1)
            )
            configured_optimizer_minis = int(self.config.ppo_mini_update_num)
            if expected_optimizer_minis != configured_optimizer_minis or expected_optimizer_minis < 1:
                raise ValueError(
                    "Duplicate-aware optimizer schedule count must match ppo_mini_update_num: "
                    f"schedule={expected_optimizer_minis}, configured={configured_optimizer_minis}"
                )
            if expected_optimizer_minis > 1 and int(self.config.ppo_epochs) != 1:
                raise ValueError(
                    "UID-coherent multi-mini actor updates require ppo_epochs=1 so the "
                    "optimizer executes exactly the scheduled number of steps"
                )
            if data.meta_info.get("duplicate_aware_optimizer_schedule") != "uid_coherent_v1":
                raise ValueError("Duplicate-aware actor update requires a uid_coherent_v1 schedule")
            if "optimizer_mini_batch_id" not in data.batch:
                raise ValueError("Duplicate-aware actor update is missing optimizer_mini_batch_id")
        if duplicate_aware:
            select_keys.append("sample_weight")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        entropy_control_config = self.config.get("entropy_control", {})
        entropy_control_enabled = bool(entropy_control_config.get("enabled", False))
        entropy_control_coef = (
            float(entropy_control_config.get("loss_coef", 0.0025)) if entropy_control_enabled else 0.0
        )
        if entropy_control_enabled:
            if not 0 <= entropy_control_coef < float("inf"):
                raise ValueError("actor entropy_control.loss_coef must be finite and nonnegative")
            control_keys = ["entropy_control_weight", "entropy_control_cap", "entropy_control_valid"]
            missing = [key for key in control_keys if key not in data.batch]
            if missing:
                raise ValueError(f"enabled entropy control is missing frozen rollout tensors: {missing}")
            select_keys.extend(control_keys)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if self.config.use_adaptive_ppo_mini_batch_size:
            self.config.ppo_mini_batch_size = data.meta_info.get(f"{wg_id}/ppo_mini_batch_size", self.config.ppo_mini_batch_size)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        if duplicate_aware:
            if len(dataloader) != expected_optimizer_minis:
                raise ValueError(
                    "UID-coherent schedule produced the wrong number of local optimizer mini-batches: "
                    f"expected={expected_optimizer_minis}, actual={len(dataloader)}"
                )
            local_mini_size = int(self.config.ppo_mini_batch_size)
            if len(data) != expected_optimizer_minis * local_mini_size:
                raise ValueError(
                    "UID-coherent local layout must contain equal optimizer mini-batches: "
                    f"rows={len(data)}, count={expected_optimizer_minis}, size={local_mini_size}"
                )
            mini_ids = data.batch["optimizer_mini_batch_id"].detach().cpu().numpy()
            event_uids = np.asarray(data.non_tensor_batch.get("event_uid"), dtype=object)
            if event_uids.shape != (len(data),):
                raise ValueError("UID-coherent local layout requires one event_uid per row")
            uid_to_mini = {}
            for mini_index in range(expected_optimizer_minis):
                start = mini_index * local_mini_size
                stop = (mini_index + 1) * local_mini_size
                if not np.all(mini_ids[start:stop] == mini_index):
                    raise ValueError(
                        f"DP rank local rows are not contiguous for optimizer mini {mini_index}"
                    )
                for uid_value in event_uids[start:stop].astype(str):
                    previous = uid_to_mini.setdefault(uid_value, mini_index)
                    if previous != mini_index:
                        raise ValueError(
                            f"event_uid={uid_value!r} crosses local optimizer mini-batches"
                        )

        metrics = {}
        if duplicate_aware:
            append_to_dict(
                metrics,
                {
                    f"actor/{wg_id}/optimizer_mini_batch_count": expected_optimizer_minis,
                    f"actor/{wg_id}/optimizer_unique_events": data.meta_info.get(
                        f"{wg_id}/optimizer_unique_event_count", float("nan")
                    ),
                    f"actor/{wg_id}/optimizer_scheduled_rows": data.meta_info.get(
                        f"{wg_id}/optimizer_scheduled_row_count", len(data)
                    ),
                },
            )
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                mini_batch_tensors = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                sample_weight_normalizer = None
                sample_weight_scale = 1.0
                sample_weight_metric_normalizer = None
                if duplicate_aware:
                    mini_response_length = mini_batch_tensors["responses"].size(1)
                    if multi_turn:
                        mini_response_mask = mini_batch_tensors["loss_mask"][:, -mini_response_length:]
                    else:
                        mini_response_mask = mini_batch_tensors["attention_mask"][:, -mini_response_length:]
                    sample_weight_normalizer = compute_sample_weight_normalizer(
                        loss_mask=mini_response_mask,
                        sample_weight=mini_batch_tensors["sample_weight"],
                        loss_agg_mode=self.config.loss_agg_mode,
                    ).detach()
                    sample_weight_metric_normalizer = compute_sample_weight_normalizer(
                        loss_mask=mini_response_mask,
                        sample_weight=mini_batch_tensors["sample_weight"],
                        loss_agg_mode="token-mean",
                    ).detach()
                    weight_stats = torch.stack(
                        (
                            mini_batch_tensors["sample_weight"].float().sum(),
                            mini_batch_tensors["sample_weight"].new_tensor(
                                mini_batch_tensors["sample_weight"].numel(), dtype=torch.float32
                            ),
                        )
                    )
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        sample_weight_normalizer = sample_weight_normalizer.clone()
                        sample_weight_metric_normalizer = sample_weight_metric_normalizer.clone()
                        torch.distributed.all_reduce(sample_weight_normalizer, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(
                            sample_weight_metric_normalizer, op=torch.distributed.ReduceOp.SUM
                        )
                        torch.distributed.all_reduce(weight_stats, op=torch.distributed.ReduceOp.SUM)
                        sample_weight_scale = float(torch.distributed.get_world_size())
                    if not torch.isfinite(sample_weight_normalizer) or sample_weight_normalizer <= 0:
                        raise ValueError(
                            "Global duplicate-aware sample-weight normalizer must be finite and positive, "
                            f"got {sample_weight_normalizer}"
                        )
                    effective_event_count = weight_stats[0]
                    adjusted_row_count = weight_stats[1]
                    if not torch.isfinite(weight_stats).all() or effective_event_count <= 0:
                        raise ValueError(f"Invalid global duplicate-aware weight statistics: {weight_stats}")
                    append_to_dict(
                        metrics,
                        {
                            f"actor/{wg_id}/effective_unique_events": effective_event_count.detach().item(),
                            f"actor/{wg_id}/adjusted_event_rows": adjusted_row_count.detach().item(),
                            f"actor/{wg_id}/event_duplication_factor": (
                                adjusted_row_count / effective_event_count
                            ).detach().item(),
                            f"actor/{wg_id}/sample_weight_normalizer": sample_weight_normalizer.detach().item(),
                        },
                    )
                if entropy_control_enabled:
                    mini_response_length = mini_batch_tensors["responses"].shape[-1]
                    mini_mask_key = "loss_mask" if multi_turn else "attention_mask"
                    mini_response_mask = mini_batch_tensors[mini_mask_key][:, -mini_response_length:]
                    mini_control_valid = (
                        mini_batch_tensors["entropy_control_valid"].bool()
                        & mini_response_mask.bool().any(-1)
                    )
                    mini_control_weights = (
                        mini_batch_tensors["sample_weight"].detach().float()
                        if duplicate_aware else torch.ones_like(mini_control_valid, dtype=torch.float32)
                    )
                    mini_control_count = (mini_control_valid.float() * mini_control_weights).sum().to(
                        device=get_torch_device().current_device(), dtype=torch.float32
                    )
                    control_dp_scale = 1.0
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(mini_control_count, op=torch.distributed.ReduceOp.SUM)
                        control_dp_scale = float(torch.distributed.get_world_size())
                    if not torch.isfinite(mini_control_count) or mini_control_count < 0:
                        raise ValueError("Entropy control unique-action count must be finite and nonnegative")
                    mini_control_denominator = torch.where(
                        mini_control_count > 0, mini_control_count, torch.ones_like(mini_control_count)
                    )
                    control_metric_sums = {}
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                duplicate_metric_sums = {} if duplicate_aware else None

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    sample_weight = data["sample_weight"] if duplicate_aware else None

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0 or (entropy_control_enabled and entropy_control_coef > 0):
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        sample_weight=sample_weight,
                        sample_weight_normalizer=sample_weight_normalizer,
                        sample_weight_scale=sample_weight_scale,
                        sample_weight_metric_normalizer=sample_weight_metric_normalizer,
                        # Return this micro-batch's contribution to the global
                        # diagnostic. We explicitly sum and all-reduce below;
                        # actor meta_info is not reduced across DP ranks.
                        sample_weight_metric_scale=1.0,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            sample_weight=sample_weight,
                            sample_weight_normalizer=sample_weight_normalizer,
                            sample_weight_scale=sample_weight_scale,
                        )

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        if duplicate_aware:
                            metric_value = entropy_loss.detach() / sample_weight_scale
                            duplicate_metric_sums["entropy_regularizer"] = (
                                duplicate_metric_sums.get("entropy_regularizer", torch.zeros_like(metric_value))
                                + metric_value
                            )
                        else:
                            append_to_dict(
                                metrics,
                                {f"actor/{wg_id}/entropy_regularizer": entropy_loss.detach().item()},
                            )
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            sample_weight=sample_weight,
                            sample_weight_normalizer=sample_weight_normalizer,
                            sample_weight_scale=sample_weight_scale,
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        if duplicate_aware:
                            metric_value = kl_loss.detach() / sample_weight_scale
                            duplicate_metric_sums["kl_loss"] = (
                                duplicate_metric_sums.get("kl_loss", torch.zeros_like(metric_value))
                                + metric_value
                            )
                        else:
                            append_to_dict(
                                metrics,
                                {
                                    f"actor/{wg_id}/kl_loss": kl_loss.detach().item(),
                                    f"actor/{wg_id}/kl_coef": self.config.kl_loss_coef,
                                },
                            )

                    if duplicate_aware:
                        # Each micro-batch already contributes a weighted
                        # numerator divided by the full optimizer mini-batch
                        # (and global DP) denominator. Summing backward calls
                        # is therefore the correctly normalized mini-batch
                        # objective; applying gradient_accumulation scaling a
                        # second time would underweight it.
                        loss = policy_loss
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    if entropy_control_enabled and entropy_control_coef > 0:
                        control_loss, control_metrics = action_entropy_hinge_loss(
                            token_entropy=entropy,
                            response_mask=response_mask,
                            weights=data["entropy_control_weight"],
                            caps=data["entropy_control_cap"],
                            action_valid=data["entropy_control_valid"],
                            sample_weight=sample_weight,
                        )
                        # The PPO objective is already normalized separately.
                        # Sum weighted action numerators over microbatches and
                        # divide once by the global optimizer-mini denominator;
                        # compensate for FSDP's rank-gradient averaging.
                        control_fraction = control_metrics["valid_actions"] / mini_control_denominator
                        loss = loss + entropy_control_coef * control_loss * control_fraction * control_dp_scale
                        for metric_name, metric_value in control_metrics.items():
                            if metric_name == "valid_actions":
                                continue
                            contribution = metric_value.detach() * control_fraction
                            control_metric_sums[metric_name] = (
                                control_metric_sums.get(metric_name, torch.zeros_like(contribution)) + contribution
                            )
                    loss.backward()

                    if duplicate_aware:
                        objective_contributions = {
                            "pg_loss": pg_loss.detach() / sample_weight_scale,
                            "policy_loss": policy_loss.detach() / sample_weight_scale,
                            "pg_clipfrac": pg_clipfrac.detach(),
                            "ppo_kl": ppo_kl.detach(),
                            "pg_clipfrac_lower": pg_clipfrac_lower.detach(),
                        }
                        for metric_name, metric_value in objective_contributions.items():
                            duplicate_metric_sums[metric_name] = (
                                duplicate_metric_sums.get(metric_name, torch.zeros_like(metric_value))
                                + metric_value
                            )
                    else:
                        append_to_dict(
                            metrics,
                            {
                                f"actor/{wg_id}/pg_loss": pg_loss.detach().item(),
                                f"actor/{wg_id}/pg_clipfrac": pg_clipfrac.detach().item(),
                                f"actor/{wg_id}/ppo_kl": ppo_kl.detach().item(),
                                f"actor/{wg_id}/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            },
                        )

                if duplicate_aware:
                    metric_names = tuple(sorted(duplicate_metric_sums))
                    global_metric_values = torch.stack(
                        [duplicate_metric_sums[name].to(dtype=torch.float32) for name in metric_names]
                    )
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_metric_values, op=torch.distributed.ReduceOp.SUM)
                    append_to_dict(
                        metrics,
                        {
                            f"actor/{wg_id}/{name}": global_metric_values[index].item()
                            for index, name in enumerate(metric_names)
                        },
                    )
                    if self.config.use_kl_loss:
                        append_to_dict(metrics, {f"actor/{wg_id}/kl_coef": self.config.kl_loss_coef})

                if entropy_control_enabled:
                    control_metric_names = tuple(sorted(control_metric_sums))
                    if control_metric_names:
                        control_metric_values = torch.stack([
                            control_metric_sums[name].to(dtype=torch.float32) for name in control_metric_names
                        ])
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(control_metric_values, op=torch.distributed.ReduceOp.SUM)
                        append_to_dict(metrics, {
                            f"actor/{wg_id}/entropy_control/{name}": control_metric_values[index].item()
                            for index, name in enumerate(control_metric_names)
                        })
                    append_to_dict(metrics, {
                        f"actor/{wg_id}/entropy_control/valid_actions": mini_control_count.item(),
                        f"actor/{wg_id}/entropy_control/loss_coef": entropy_control_coef,
                    })

                grad_norm = self._optimizer_step()
                data = {f"actor/{wg_id}/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
