# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
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

import math
import random
import zlib
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import ListConfig
from PIL import Image

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
import verl.utils.torch_functional as verl_F

from agent_system.event_trace import event_metadata_enabled

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def preprocess_fn(
    item: int,
    gen_batch: DataProto,
    obs: Dict,
    config,
    tokenizer,
    processor = None,
):
    """
    Process a single observation sample, organizing environment observations (text and/or images) 
    into a format processable by the model.
    
    Parameters:
        item (int): Sample index in the batch
        gen_batch (DataProto): Batch data containing original prompts
        obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
    
    Returns:
        dict: Contains processed input data such as input_ids, attention_mask, etc.
    """

    raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
    data_source = gen_batch.non_tensor_batch['data_source'][item]
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    # Get observation components
    obs_texts = obs.get('text', None)
    obs_images = obs.get('image', None)
    obs_anchors = obs.get('anchor', None)
    obs_text = obs_texts[item] if obs_texts is not None else None
    obs_image = obs_images[item] if obs_images is not None else None
    obs_anchor = obs_anchors[item] if obs_anchors is not None else None
    is_multi_modal = obs_image is not None

    _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

    # Build chat structure
    # obs_content = raw_prompt[0]['content']
    # if '<image>' in obs_content: 
    #     obs_content = obs_content.replace('<image>', '')

    # Build chat structure
    system_prompt = "You are a helpful and harmless assistant."
    for message in raw_prompt:
        if message['role'] == 'system':
            system_prompt = message['content']
        # if message['role'] == 'user':
        #     context_from_dataset = message['content']

    # if len(context_from_dataset) > 0 and obs_text is not None:
    #     obs_content = obs_text.replace('{placeholder_of_dataset_context}', context_from_dataset)
    # else:
    #     print(f"Warning: No text observation found!")

    obs_content = obs_text
    chat = np.array([
        {"content": system_prompt, "role": "system"},
        {"content": obs_content, "role": "user",}
        ])
    # Apply chat template
    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs
    )
    
    # Initialize return dict
    row_dict = {}
    
    # Process multimodal data
    if is_multi_modal:
        # Replace image placeholder with vision tokens
        raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
        image_inputs = processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
        if image_grid_thw is not None:
            merge_length = processor.image_processor.merge_size**2
            index = 0
            while '<image>' in prompt_with_chat_template:
                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index += 1

            prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                            processor.image_token)

    else:
        raw_prompt = prompt_with_chat_template
    
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                        tokenizer=tokenizer,
                                                                        max_length=config.data.max_prompt_length,
                                                                        pad_token_id=tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation=config.data.truncation,)
    
    

    if is_multi_modal:
        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = [
            get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            ]  # (1, 3, seq_len)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)
    

    raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    untruncated_prompt_token_count = len(raw_prompt_ids)
    prompt_was_truncated = untruncated_prompt_token_count > config.data.max_prompt_length
    if prompt_was_truncated:
        if config.data.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-config.data.max_prompt_length :]
        elif config.data.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: config.data.max_prompt_length]
        elif config.data.truncation == "middle":
            left_half = config.data.max_prompt_length // 2
            right_half = config.data.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif config.data.truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {config.data.max_prompt_length}.")

    # Build final output dict
    row_dict.update({
        'input_ids': input_ids[0],
        'attention_mask': attention_mask[0],
        'position_ids': position_ids[0],
        'raw_prompt_ids': raw_prompt_ids,
        'anchor_obs': _obs_anchor,
        'index': item,
        'data_source': data_source
    })

    if config.data.get('return_raw_chat', False):
        row_dict['raw_prompt'] = chat.tolist()

    if event_metadata_enabled(config):
        # This is the exact agent-specific state before the current action.  It
        # is kept separate from raw_prompt because BaseAgent pops raw_prompt
        # before generation.  Never mutate this object when adding hindsight.
        row_dict['hcapo_state_chat'] = deepcopy(chat.tolist())
        row_dict['prompt_was_truncated'] = prompt_was_truncated
        row_dict['untruncated_prompt_token_count'] = untruncated_prompt_token_count
    
    return row_dict


def preprocess_batch(
    gen_batch: DataProto, 
    obs: Dict, 
    config,
    tokenizer,
    processor=None,
) -> DataProto:
    """
    Process a batch of observation samples, converting environment observations into model-processable format.
    
    Parameters:
        gen_batch (DataProto): Batch data containing original prompts
        obs (Dict): Environment observation dictionary
            - 'text' (None or List[str]): Text observation data
            - 'image' (np.ndarray or torch.Tensor): Image observation data
            - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
    
    Returns:
        DataProto: Contains processed batch data with preserved metadata
    """
    batch_size = len(gen_batch.batch['input_ids'])
    processed_samples = []
    
    # Process each sample in parallel
    for item in range(batch_size):
        # Extract per-sample observations
        processed = preprocess_fn(
            item=item,
            gen_batch=gen_batch,
            obs=obs,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
        )
        processed_samples.append(processed)
    
    # Aggregate batch data
    batch = collate_fn(processed_samples)

    if event_metadata_enabled(config):
        # np.array(list_of_equal_length_chats, dtype=object) otherwise becomes
        # a 2-D array.  Force one opaque chat object per rollout row so all
        # DataProto select/repeat operations keep event alignment.
        state_chats = np.empty(batch_size, dtype=object)
        for item, processed in enumerate(processed_samples):
            state_chats[item] = deepcopy(processed['hcapo_state_chat'])
        batch['hcapo_state_chat'] = state_chats
    
    # Create DataProto with preserved metadata
    new_batch = DataProto.from_single_dict(
        data=batch,
        meta_info=gen_batch.meta_info
    )

    return new_batch


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto, wg_id: str, mode="copy") -> DataProto:
    """Make a worker-group batch divisible without changing event weighting.

    ``mode="copy"`` is required by the distributed rollout/actor stack, but
    copied Search events must not become additional PPO samples.  For
    ``team_event_gae`` we therefore attach an inverse-multiplicity
    ``sample_weight`` tensor after padding: all rows sharing one ``event_uid``
    have weights that sum to one.  The actor consumes this tensor with a
    mini-batch/global-DP normalizer.

    Other advantage estimators keep the historical unweighted behavior.
    """
    use_adaptive_bs = config.actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size
    ppo_mini_update_num = config.actor_rollout_ref.actor.ppo_mini_update_num

    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes

    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size * world_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size

    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        adjusted_batch = data
    else:
        if mode == "delete":
            # Generate indices to remove, rather than indices to keep
            remove_indices = np.random.choice(bs, remainder, replace=False)
            # Sort remove_indices to maintain stability when deleting
            remove_indices = np.sort(remove_indices)
            
            # Create a boolean mask for elements to keep
            keep_mask = np.ones(bs, dtype=bool)
            keep_mask[remove_indices] = False

            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            # Apply the mask to keep elements in their original order
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            del data
        elif mode == "copy":
            to_add = size_divisor - remainder
            # If to_add > bs, we need to copy multiple times
            dup_protos = []
            remaining = to_add
            while remaining > 0:
                if remaining >= bs:
                    # Copy the entire batch
                    dup_protos.append(data)
                    remaining -= bs
                    print(f"Copy the entire batch, remaining: {remaining}")
                else:
                    # Copy a subset
                    dup_indices = np.random.choice(bs, remaining, replace=False)
                    dup_protos.append(data.select_idxs(dup_indices))
                    remaining = 0
            
            adjusted_batch = DataProto.concat([data] + dup_protos)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    algorithm_config = getattr(config, "algorithm", None)
    adv_estimator = None if algorithm_config is None else algorithm_config.get("adv_estimator", None)
    if str(adv_estimator).lower() == "team_event_gae":
        if "event_uid" not in adjusted_batch.non_tensor_batch:
            raise ValueError("team_event_gae adjust_batch requires event_uid metadata")

        event_uids = np.asarray(adjusted_batch.non_tensor_batch["event_uid"], dtype=object)
        if event_uids.shape != (len(adjusted_batch),):
            raise ValueError(
                "event_uid must contain exactly one scalar per adjusted row, "
                f"got shape={event_uids.shape}, rows={len(adjusted_batch)}"
            )
        if any(uid is None or str(uid) == "" for uid in event_uids):
            raise ValueError("team_event_gae adjust_batch received an empty event_uid")

        # np.unique sorts the IDs, but inverse_indices maps every row back to
        # its UID cohort, so input order and subsequent balance_batch reorders
        # do not affect the weights.
        unique_event_uids, inverse_indices, multiplicities = np.unique(
            event_uids.astype(str),
            return_inverse=True,
            return_counts=True,
        )
        row_multiplicities = multiplicities[inverse_indices]
        sample_weights = np.reciprocal(row_multiplicities.astype(np.float64)).astype(np.float32)
        if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
            raise ValueError("team_event_gae produced invalid duplicate-aware sample weights")

        adjusted_batch.batch["sample_weight"] = torch.as_tensor(
            sample_weights,
            dtype=torch.float32,
            device=adjusted_batch.batch.device,
        )
        adjusted_batch.non_tensor_batch["sample_dup_count"] = row_multiplicities.astype(np.int64)
        adjusted_batch.non_tensor_batch["is_adjustment_copy"] = np.arange(len(adjusted_batch)) >= bs
        adjusted_batch.meta_info[f"{wg_id}/unique_event_count"] = int(len(unique_event_uids))
        adjusted_batch.meta_info[f"{wg_id}/adjusted_row_count"] = int(len(adjusted_batch))
        sample_weight_sum = float(sample_weights.sum(dtype=np.float64))
        adjusted_batch.meta_info[f"{wg_id}/sample_weight_sum"] = sample_weight_sum

        expected_weight_sum = float(len(unique_event_uids))
        if not math.isclose(
            adjusted_batch.meta_info[f"{wg_id}/sample_weight_sum"],
            expected_weight_sum,
            # 1 / multiplicity is stored as float32 for the actor.  Cohorts
            # such as 1/6 therefore accumulate a tiny representation error;
            # use a scale-aware check instead of rejecting valid large
            # adjusted batches (for example 43 unique rows padded to 256).
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Duplicate-aware sample weights must sum to the number of unique events: "
                f"sum={sample_weight_sum}, "
                f"unique={len(unique_event_uids)}"
            )

    if use_adaptive_bs:
        adjusted_bs = len(adjusted_batch)
        ulysses_sequence_parallel_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        assert config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) == config.critic.get("ulysses_sequence_parallel_size", 1)
        # assert adjusted_bs % ppo_mini_update_num == 0, f"Adjusted batch size {adjusted_bs} is not divisible by (update_num*node_num//ulysses_sequence_parallel_size) {ppo_mini_update_num*world_size//ulysses_sequence_parallel_size}."
        adjusted_batch.meta_info[f"{wg_id}/ppo_mini_batch_size"] = -(-adjusted_bs // (ppo_mini_update_num*world_size//ulysses_sequence_parallel_size)) # ceil division
        assert adjusted_batch.meta_info[f"{wg_id}/ppo_mini_batch_size"] > 0, "ppo_mini_batch_size must be greater than 0."

    return adjusted_batch


def prepare_team_event_optimizer_batch(
    config,
    data: DataProto,
    *,
    wg_id: str,
    seed: int = 0,
    actor_world_size: int | None = None,
) -> DataProto:
    """Build UID-coherent optimizer mini-batches for Team Event GAE.

    ``adjust_batch`` padding is needed before log-prob/reward/advantage
    computation, but its copied rows are not suitable for several sequential
    optimizer steps: copies of one event could otherwise land in different
    steps.  This function collapses the adjusted batch to one representative
    per ``event_uid``, partitions unique events into disjoint logical optimizer
    mini-batches, and pads *inside each logical mini only*.  Inverse
    multiplicity weights therefore sum to one inside the sole optimizer step
    that owns an event.

    The returned global layout is rank-major then mini-major.  After the
    standard DP dispatch chunks it by rank, every worker observes the same
    ordered number of logical mini-batches and can call ``optimizer.step`` in
    lockstep.  ``ppo_mini_update_num=1`` keeps the existing adjusted rows and
    numerical path, apart from adding schedule metadata used by runtime
    validation.
    """

    if "event_uid" not in data.non_tensor_batch:
        raise ValueError("Team-event optimizer scheduling requires event_uid metadata")
    if "sample_weight" not in data.batch:
        raise ValueError("Team-event optimizer scheduling requires sample_weight")
    if "multi_modal_inputs" in data.non_tensor_batch:
        raise ValueError("UID-coherent Team Event optimizer scheduling currently supports text batches only")

    if "responses" not in data.batch or "attention_mask" not in data.batch:
        raise ValueError(
            "Team-event optimizer scheduling requires responses and attention_mask "
            "for driver-side valid-token validation"
        )
    response_length = int(data.batch["responses"].shape[-1])
    multi_turn_config = config.actor_rollout_ref.rollout.get("multi_turn", {})
    multi_turn_get = getattr(multi_turn_config, "get", None)
    is_multi_turn = bool(
        multi_turn_get("enable", False)
        if callable(multi_turn_get)
        else getattr(multi_turn_config, "enable", False)
    )
    action_mask_key = "loss_mask" if is_multi_turn else "attention_mask"
    if action_mask_key not in data.batch:
        raise ValueError(
            f"Team-event optimizer scheduling requires {action_mask_key} for every event row"
        )
    action_mask = data.batch[action_mask_key][:, -response_length:]
    valid_token_counts = action_mask.reshape(len(data), -1).sum(dim=-1)
    invalid_token_rows = torch.nonzero(valid_token_counts <= 0, as_tuple=False).flatten()
    if invalid_token_rows.numel() > 0:
        examples = invalid_token_rows[:8].detach().cpu().tolist()
        raise ValueError(
            "Every Team Event actor row must contain at least one valid response token before "
            f"distributed collectives; invalid row indices={examples}"
        )

    event_uids = np.asarray(data.non_tensor_batch["event_uid"], dtype=object)
    weights = data.batch["sample_weight"].detach().float().cpu().numpy()
    if weights.ndim == 2 and weights.shape[-1] == 1:
        weights = weights[:, 0]
    if event_uids.shape != (len(data),) or weights.shape != (len(data),):
        raise ValueError(
            "Misaligned team-event optimizer inputs: "
            f"event_uid={event_uids.shape}, sample_weight={weights.shape}, rows={len(data)}"
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Team-event optimizer sample weights must be finite and positive")

    uid_strings = event_uids.astype(str)
    unique_uids, inverse = np.unique(uid_strings, return_inverse=True)
    uid_mass = np.bincount(inverse, weights=weights.astype(np.float64), minlength=len(unique_uids))
    if not np.allclose(uid_mass, np.ones_like(uid_mass), rtol=1e-6, atol=1e-6):
        raise ValueError("Input rows must sum to sample_weight=1 for every event_uid")

    actor_config = config.actor_rollout_ref.actor
    num_optimizer_minis = int(actor_config.ppo_mini_update_num)
    if num_optimizer_minis < 1:
        raise ValueError(
            "ppo_mini_update_num must be a positive integer for Team Event GAE, "
            f"got {num_optimizer_minis}"
        )
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    if actor_world_size is not None and int(actor_world_size) != world_size:
        raise ValueError(
            "Configured world size does not match the actor worker group: "
            f"configured={world_size}, actor={actor_world_size}"
        )
    configured_micro_batch_size = actor_config.get("ppo_micro_batch_size_per_gpu", None)
    if configured_micro_batch_size is None:
        raise ValueError(
            "UID-coherent multi-mini scheduling requires an explicit "
            "actor.ppo_micro_batch_size_per_gpu"
        )
    micro_batch_size = int(configured_micro_batch_size)
    if world_size < 1 or micro_batch_size < 1:
        raise ValueError(
            "Team-event optimizer scheduling requires positive world/micro batch sizes, "
            f"got world_size={world_size}, micro_batch_size={micro_batch_size}"
        )
    if num_optimizer_minis > 1:
        if int(actor_config.get("ppo_epochs", 1)) != 1:
            raise ValueError(
                "UID-coherent ppo_mini_update_num>1 currently requires actor.ppo_epochs=1 "
                "so K denotes exactly K optimizer steps per rollout update"
            )
        sequence_parallel_size = int(actor_config.get("ulysses_sequence_parallel_size", 1))
        if sequence_parallel_size != 1:
            raise ValueError(
                "UID-coherent multi-mini scheduling currently requires "
                "ulysses_sequence_parallel_size=1; physical-rank packing is not valid for SP groups"
            )
        if bool(actor_config.get("use_dynamic_bsz", False)):
            raise ValueError(
                "UID-coherent multi-mini scheduling currently requires actor.use_dynamic_bsz=False "
                "so every DP rank executes the same number of micro-batch backward calls"
            )

    def attach_schedule_meta(batch: DataProto, *, local_mini_size: int, unique_count: int) -> DataProto:
        batch.meta_info = deepcopy(batch.meta_info)
        batch.meta_info[f"{wg_id}/ppo_mini_batch_size"] = int(local_mini_size)
        batch.meta_info[f"{wg_id}/optimizer_mini_batch_count"] = int(num_optimizer_minis)
        batch.meta_info[f"{wg_id}/optimizer_unique_event_count"] = int(unique_count)
        batch.meta_info[f"{wg_id}/optimizer_scheduled_row_count"] = int(len(batch))
        batch.meta_info[f"{wg_id}/global_token_num"] = (
            torch.sum(batch.batch["attention_mask"], dim=-1).detach().cpu().tolist()
        )
        batch.meta_info["duplicate_aware_optimizer_mini_batch_count"] = int(num_optimizer_minis)
        batch.meta_info["duplicate_aware_optimizer_schedule"] = "uid_coherent_v1"
        return batch

    if num_optimizer_minis == 1:
        if len(data) % world_size != 0:
            raise ValueError(
                f"Adjusted Team Event batch size {len(data)} is not divisible by world_size={world_size}"
            )
        local_mini_size = len(data) // world_size
        if local_mini_size % micro_batch_size != 0:
            raise ValueError(
                "Local Team Event optimizer mini-batch must be divisible by the actor micro-batch: "
                f"local={local_mini_size}, micro={micro_batch_size}"
            )
        data.batch["optimizer_mini_batch_id"] = torch.zeros(
            len(data), dtype=torch.int64, device=data.batch.device
        )
        return attach_schedule_meta(data, local_mini_size=local_mini_size, unique_count=len(unique_uids))

    if len(unique_uids) < num_optimizer_minis:
        raise ValueError(
            "Cannot create non-empty UID-coherent optimizer mini-batches: "
            f"unique_events={len(unique_uids)}, ppo_mini_update_num={num_optimizer_minis}"
        )

    # Pick a canonical representative and order unique rows by UID so ordinary
    # balance/reorder operations before this function cannot change the logical
    # optimizer partition.
    copy_flags = np.asarray(data.non_tensor_batch.get("is_adjustment_copy"), dtype=object)
    if copy_flags.shape != (len(data),):
        raise ValueError(
            "UID-coherent optimizer scheduling requires is_adjustment_copy for every adjusted row"
        )
    first_index_by_uid: Dict[str, int] = {}
    for uid_value in unique_uids:
        real_rows = np.flatnonzero(np.logical_and(uid_strings == uid_value, ~copy_flags.astype(bool)))
        if len(real_rows) != 1:
            raise ValueError(
                "Every event_uid must have exactly one non-adjustment canonical row before optimizer "
                f"repacking; event_uid={uid_value!r}, canonical_rows={len(real_rows)}"
            )
        first_index_by_uid[uid_value] = int(real_rows[0])
    first_indices = [first_index_by_uid[uid_value] for uid_value in unique_uids]
    unique_data = data.select_idxs(np.asarray(first_indices, dtype=np.int64))
    unique_count = len(unique_data)

    stable_seed = (int(seed) + int(zlib.crc32(str(wg_id).encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(stable_seed)
    permutation = rng.permutation(unique_count)
    logical_groups = [permutation[index::num_optimizer_minis] for index in range(num_optimizer_minis)]
    if any(len(group) == 0 for group in logical_groups):
        raise ValueError("UID-coherent optimizer scheduling produced an empty logical mini-batch")

    global_micro_divisor = world_size * micro_batch_size
    target_global_rows = int(
        math.ceil(max(len(group) for group in logical_groups) / global_micro_divisor)
        * global_micro_divisor
    )
    local_mini_size = target_global_rows // world_size
    logical_minis: List[DataProto] = []

    for mini_id, group in enumerate(logical_groups):
        base_indices = [int(index) for index in group]
        selection = list(base_indices)
        if len(selection) < target_global_rows:
            shuffled_base = list(base_indices)
            rng.shuffle(shuffled_base)
            cursor = 0
            while len(selection) < target_global_rows:
                selection.append(shuffled_base[cursor % len(shuffled_base)])
                cursor += 1

        mini = unique_data.select_idxs(np.asarray(selection, dtype=np.int64))
        mini_uids = np.asarray(mini.non_tensor_batch["event_uid"], dtype=object).astype(str)
        _, mini_inverse, mini_counts = np.unique(
            mini_uids,
            return_inverse=True,
            return_counts=True,
        )
        row_counts = mini_counts[mini_inverse]
        mini.batch["sample_weight"] = torch.as_tensor(
            np.reciprocal(row_counts.astype(np.float64)).astype(np.float32),
            dtype=torch.float32,
            device=mini.batch.device,
        )
        mini.batch["optimizer_mini_batch_id"] = torch.full(
            (len(mini),), mini_id, dtype=torch.int64, device=mini.batch.device
        )
        mini.non_tensor_batch["sample_dup_count"] = row_counts.astype(np.int64)
        mini.non_tensor_batch["is_adjustment_copy"] = np.arange(len(mini)) >= len(base_indices)

        # Balance token work independently inside each logical optimizer mini;
        # the resulting contiguous partitions are one slice per DP rank.
        sequence_lengths = (
            mini.batch["attention_mask"].view(len(mini), -1).sum(-1).detach().cpu().tolist()
        )
        rank_partitions = get_seqlen_balanced_partitions(
            sequence_lengths,
            k_partitions=world_size,
            equal_size=True,
        )
        mini.reorder(torch.tensor([row for partition in rank_partitions for row in partition]))
        logical_minis.append(mini)

    # DP dispatch chunks contiguous rank blocks.  Store every rank's mini 0,
    # mini 1, ... pieces together so its local DataLoader performs the same
    # optimizer-step sequence as every other rank.
    rank_major_pieces: List[DataProto] = []
    for rank in range(world_size):
        start = rank * local_mini_size
        stop = (rank + 1) * local_mini_size
        for mini in logical_minis:
            rank_major_pieces.append(mini.slice(start, stop))
    scheduled = DataProto.concat(rank_major_pieces)
    scheduled = attach_schedule_meta(
        scheduled,
        local_mini_size=local_mini_size,
        unique_count=unique_count,
    )

    scheduled_uids = np.asarray(scheduled.non_tensor_batch["event_uid"], dtype=object).astype(str)
    scheduled_ids = scheduled.batch["optimizer_mini_batch_id"].detach().cpu().numpy()
    scheduled_weights = scheduled.batch["sample_weight"].detach().float().cpu().numpy()
    uid_to_mini: Dict[str, int] = {}
    for uid_value, mini_id in zip(scheduled_uids, scheduled_ids):
        previous = uid_to_mini.setdefault(uid_value, int(mini_id))
        if previous != int(mini_id):
            raise ValueError(f"event_uid={uid_value!r} was split across optimizer mini-batches")
    scheduled_unique, scheduled_inverse = np.unique(scheduled_uids, return_inverse=True)
    scheduled_mass = np.bincount(
        scheduled_inverse,
        weights=scheduled_weights.astype(np.float64),
        minlength=len(scheduled_unique),
    )
    if set(scheduled_unique) != set(unique_uids) or not np.allclose(
        scheduled_mass, np.ones_like(scheduled_mass), rtol=1e-6, atol=1e-6
    ):
        raise ValueError("UID-coherent optimizer scheduling changed unique-event identity or weight mass")

    return scheduled

def split_batch_by_wg_ids(data: DataProto, unique_wg_ids: List[str], update_agent_ids: List[str] = None) -> Dict[str, DataProto]:
    """
    Split a DataProto batch into multiple batches based on unique model IDs.
    
    Parameters:
        data (DataProto): Input batch containing agent IDs in non_tensor_batch['agent_id']
        unique_wg_ids (list): List of unique workgroup IDs to split the batch by.
        update_agent_ids (List[str], optional): List of agent IDs that will be trained. If provided, all agents will be included in the output.
    Returns:
        Dict[str, DataProto]: Dictionary mapping agent IDs to their respective DataProto batches
    """
    wg_ids = data.non_tensor_batch.get('wg_id', None)
    agent_ids = data.non_tensor_batch.get('agent_id', None)
    if wg_ids is None:
        raise ValueError("DataProto does not contain 'wg_id' in non_tensor_batch.")

    split_batches = {}
    update_agent_ids = None if update_agent_ids is None else np.array(update_agent_ids, dtype=object)
    active_mask = np.ones_like(agent_ids, dtype=bool) if update_agent_ids is None \
                else np.isin(agent_ids, update_agent_ids)
        
    for _id in unique_wg_ids:
        indices = np.flatnonzero(np.logical_and((wg_ids == _id), active_mask))
        if len(indices) > 0:
            split_batches[_id] = data.select_idxs(indices)
    
    return split_batches

def combine_batches(split_batches: Dict[str, DataProto]) -> DataProto:
    """
    Combine multiple DataProto batches into a single batch.
    
    Parameters:
        split_batches (Dict[str, DataProto]): Dictionary mapping agent IDs to their respective DataProto batches
    
    Returns:
        DataProto: Combined batch containing all data from the input split batches
    """
    combined_batch = None
    combined_meta_info = {}
    
    for _id, batch in split_batches.items():
        # DataProto.concat historically keeps meta_info only from its first
        # input. Preserve every WG-scoped key (adaptive mini-batch sizes,
        # duplicate diagnostics, token counts) explicitly; common keys retain
        # the first WG's value, matching the original concat contract.
        for key, value in batch.meta_info.items():
            if key not in combined_meta_info or key.startswith(f"{_id}/"):
                combined_meta_info[key] = value
        if combined_batch is None:
            combined_batch = batch
        else:
            combined_batch = DataProto.concat([combined_batch, batch])

    if combined_batch is not None:
        combined_batch.meta_info = combined_meta_info
    return combined_batch

def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
