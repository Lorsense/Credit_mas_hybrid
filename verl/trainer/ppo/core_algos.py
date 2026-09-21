# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict, Counter
import math

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any, Optional

from verl.trainer.ppo.team_event_boundary_value import (
    BoundaryValueConfig,
    estimate_boundary_values,
)
from verl.trainer.ppo.team_event_math_value import (
    MathValueConfig,
    estimate_math_values,
)


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    env_step = batch.non_tensor_batch['env_step'].astype(np.int32)
    # returns_by_traj_ = {}
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        traj_env_step = env_step[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        first_of_group = np.r_[True, traj_env_step[1:] != traj_env_step[:-1]]
        step_starts = np.flatnonzero(first_of_group)
        step_ends = np.r_[step_starts[1:], len(traj_env_step)]

        step_rewards = traj_rewards[step_starts].astype(np.float32)

        # Calculate returns
        step_returns = np.zeros_like(step_rewards, dtype=np.float32)
        running_return = 0.0
        
        # Calculate returns from the end to the start
        for k in reversed(range(len(step_rewards))):
            running_return = step_rewards[k] + gamma * running_return
            step_returns[k] = running_return
        
        traj_returns = np.zeros_like(traj_rewards, dtype=np.float32)
        for sr, s, e in zip(step_returns, step_starts, step_ends):
            traj_returns[s:e] = sr
            
        # Store the results
        # returns_by_traj_[uid] = traj_returns
        returns_by_traj[uid] = (traj_indices, traj_returns)
    
    # Recombine the returns into the original batch order
    # all_returns_ = np.zeros_like(rewards)
    # for i, uid in enumerate(traj_uids):
    #     traj_indices = np.where(traj_uids == uid)[0]
    #     idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
    #     all_returns_[i] = returns_by_traj_[uid][idx_in_traj]

    all_returns = np.zeros_like(rewards, dtype=np.float32)
    for uid in unique_traj_uids:
        traj_indices, traj_returns = returns_by_traj[uid]
        all_returns[traj_indices] = traj_returns

    # assert (all_returns==all_returns_).all()
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   group_by_agent_id: bool = False
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, group_by_agent_id)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        group_by_agent_id: bool = False,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard trajectory-level GRPO), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages



def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def _event_scalar(value: Any, *, default: Optional[float] = None) -> float:
    """Convert one collated event scalar to float without hiding bad shapes."""

    if value is None:
        if default is None:
            raise ValueError("Expected an event scalar, got None")
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape {value.shape}")
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _event_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape {value.shape}")
        return bool(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return bool(value.item())
    return bool(value)


def _event_state_key(value: Any) -> Any:
    """Build an exact, hashable key for a nested pre-action chat state."""

    if isinstance(value, torch.Tensor):
        return _event_state_key(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return tuple(_event_state_key(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _event_state_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_event_state_key(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def compute_team_event_gae_advantage(
    token_level_rewards: torch.Tensor,
    token_level_scores: torch.Tensor,
    response_mask: torch.Tensor,
    uid: np.ndarray,
    traj_index: np.ndarray,
    event_uid: np.ndarray,
    event_index: np.ndarray,
    event_count: np.ndarray,
    role_event_index: np.ndarray,
    role_event_count: np.ndarray,
    agent_id: np.ndarray,
    event_type: np.ndarray,
    env_step_index: np.ndarray,
    is_env_action: np.ndarray,
    env_action_owner: np.ndarray,
    env_reward: np.ndarray,
    env_done: np.ndarray,
    state_chats: np.ndarray,
    is_action_valid: np.ndarray,
    gamma: float = 1.0,
    lam: float = 1.0,
    internal_gamma: float = 1.0,
    internal_lam: float = 1.0,
    use_invalid_action_penalty: bool = True,
    invalid_action_penalty_coef: float = 0.0,
    include_kl_shaping: bool = True,
    normalize_advantages: bool = True,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6,
    agent_local_mode: str = "off",
    agent_local_mix_alpha: float = 0.0,
    agent_local_lam: float = 0.95,
    agent_local_internal_lam: float = 1.0,
    value_config: Any = None,
    route_target: Optional[np.ndarray] = None,
    route_reason: Optional[np.ndarray] = None,
    tool_observation: Optional[np.ndarray] = None,
    math_value_config: Any = None,
    executed_action_text: Optional[np.ndarray] = None,
    diagnostics_meta: Optional[Dict[str, Any]] = None,
):
    """Compute critic-free team TD/GAE over the global cross-agent event chain.

    Every unique ``event_uid`` is evaluated exactly once even when
    ``adjust_batch`` has copied/reordered rows.  The scalar event advantage is
    then mapped back to every matching row and broadcast over that response's
    valid tokens; the trainer's existing WG split therefore sends it only to
    the policy that produced the event.

    Values are leave-one-trajectory-out means of discounted Monte-Carlo
    returns.  Exact pre-action state matches are preferred.  A singleton exact
    state falls back to the same (task, role, event type, environment depth),
    then (task, role, event type), always excluding the current trajectory.
    This avoids the self-inclusion cancellation of a singleton grouped value.

    The transition after a Verifier event is internal (defaults gamma=lambda=1).
    The transition after Search/Answer consumes an environment step and uses
    ``gamma``/``lam``.  A terminal environment action never bootstraps.

    ``agent_local_mode`` optionally adds a same-agent semi-Markov GAE.  Other
    agents' events are treated as environment dynamics between two decisions
    of the same producer: rewards are accumulated across the whole segment and
    gamma/lambda factors are multiplied across every crossed global edge.
    ``shadow`` computes diagnostics without changing the actor advantage;
    ``blend`` convexly mixes raw team/local advantages before the single
    existing normalization pass.  The default ``off`` path is the v3.1.0
    Team Event GAE behavior.

    ``value_config.mode=boundary_loto`` constructs post-action environment
    values from frozen document kernels and route-compatible LOTO peers, then
    shares each post value with the next event's pre value. ``application``
    may be ``shadow``: boundary diagnostics are computed but legacy actor
    advantages/returns remain unchanged. This is independent of G1 shadow.

    ``math_value_config.mode=math_answer_loto`` is the Math-orchestra analogue:
    post values are answer-conditioned leave-one-trajectory-out expectations of
    the binary terminal env success (same-round exact answer match, then
    cross-round decay), anchored by the question-mean shrinkage parent.  The
    dispatch happens before the Search ``BoundaryValueConfig`` is resolved and
    the two value engines are mutually exclusive.
    """

    math_config = MathValueConfig.from_mapping(math_value_config)
    math_enabled = math_config.mode == "math_answer_loto"
    if math_enabled:
        math_config.validate_runtime(
            gamma=gamma, internal_gamma=internal_gamma, agent_local_mode=agent_local_mode
        )
        if (
            math_config.auxiliary_mode == "env_only"
            and use_invalid_action_penalty
            and float(invalid_action_penalty_coef) != 0.0
        ):
            raise ValueError("math_answer_loto env_only requires invalid action penalty disabled or zero")
    boundary_config = BoundaryValueConfig.from_mapping(value_config)
    boundary_enabled = boundary_config.mode == "boundary_loto"
    boundary_active = boundary_enabled and boundary_config.application == "active"
    if boundary_enabled and math_enabled:
        raise ValueError("math_answer_loto and Search boundary_loto are mutually exclusive value engines")
    if boundary_enabled:
        boundary_config.validate_runtime(
            gamma=gamma, internal_gamma=internal_gamma, agent_local_mode=agent_local_mode
        )
        if (
            boundary_config.auxiliary_mode == "env_only"
            and use_invalid_action_penalty
            and float(invalid_action_penalty_coef) != 0.0
        ):
            raise ValueError("boundary_loto env_only requires invalid action penalty disabled or zero")

    if token_level_rewards.shape != token_level_scores.shape:
        raise ValueError(
            "token_level_rewards and token_level_scores must have the same shape: "
            f"{tuple(token_level_rewards.shape)} != {tuple(token_level_scores.shape)}"
        )
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError(
            "response_mask and token rewards must have the same shape: "
            f"{tuple(response_mask.shape)} != {tuple(token_level_rewards.shape)}"
        )
    bsz = int(response_mask.shape[0])
    fields = {
        "uid": uid,
        "traj_uid": traj_index,
        "event_uid": event_uid,
        "event_index": event_index,
        "event_count": event_count,
        "role_event_index": role_event_index,
        "role_event_count": role_event_count,
        "agent_id": agent_id,
        "event_type": event_type,
        "env_step_index": env_step_index,
        "is_env_action": is_env_action,
        "env_action_owner": env_action_owner,
        "env_reward": env_reward,
        "env_done": env_done,
        "hcapo_state_chat": state_chats,
        "is_action_valid": is_action_valid,
    }
    if boundary_enabled:
        for name, values in (
            ("route_target", route_target),
            ("route_reason", route_reason),
            ("tool_observation", tool_observation),
        ):
            if values is None:
                raise ValueError(f"boundary_loto requires the online {name} field")
            fields[name] = values
    if math_enabled:
        if executed_action_text is None:
            raise ValueError("math_answer_loto requires the online executed_action_text field")
        fields["executed_action_text"] = executed_action_text
    for name, values in fields.items():
        if len(values) != bsz:
            raise ValueError(f"{name} length {len(values)} != batch size {bsz}")
    if not (0.0 <= float(gamma) <= 1.0 and 0.0 <= float(lam) <= 1.0):
        raise ValueError(f"gamma and lam must be in [0, 1], got {gamma}, {lam}")
    if not (0.0 <= float(internal_gamma) <= 1.0 and 0.0 <= float(internal_lam) <= 1.0):
        raise ValueError(
            "internal_gamma and internal_lam must be in [0, 1], got "
            f"{internal_gamma}, {internal_lam}"
        )
    local_mode = str(agent_local_mode).strip().lower()
    if local_mode not in {"off", "shadow", "blend"}:
        raise ValueError(
            "agent_local_mode must be one of {'off', 'shadow', 'blend'}, "
            f"got {agent_local_mode!r}"
        )
    if not 0.0 <= float(agent_local_mix_alpha) <= 1.0:
        raise ValueError(
            "agent_local_mix_alpha must be in [0, 1], "
            f"got {agent_local_mix_alpha}"
        )
    if not 0.0 <= float(agent_local_lam) <= 1.0:
        raise ValueError(f"agent_local_lam must be in [0, 1], got {agent_local_lam}")
    if not 0.0 <= float(agent_local_internal_lam) <= 1.0:
        raise ValueError(
            "agent_local_internal_lam must be in [0, 1], "
            f"got {agent_local_internal_lam}"
        )

    # KL is event-local shaping.  The outcome reward manager duplicates the
    # episode score on every row, so that score is deliberately not used here.
    # The true environment reward comes only from actor-owned env transitions.
    kl_shaping = (token_level_rewards - token_level_scores).sum(dim=-1).detach().cpu()
    boundary_kl_shaping = None
    if boundary_enabled:
        boundary_kl_shaping = (
            (token_level_rewards - token_level_scores) * response_mask
        ).sum(dim=-1).detach().cpu()

    def boundary_text(value, name):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"boundary_loto {name} must contain one scalar per row")
            value = value.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, str):
            raise ValueError(f"boundary_loto {name} must be text or None, got {type(value).__name__}")
        return value

    rows_by_event: Dict[str, List[int]] = defaultdict(list)
    events: Dict[str, Dict[str, Any]] = {}
    for row in range(bsz):
        event_key = str(event_uid[row])
        rows_by_event[event_key].append(row)
        owner_value = env_action_owner[row]
        if isinstance(owner_value, torch.Tensor):
            if owner_value.numel() != 1:
                raise ValueError(f"Expected scalar env_action_owner, got shape {tuple(owner_value.shape)}")
            owner_value = owner_value.detach().cpu().item()
        elif isinstance(owner_value, np.ndarray):
            if owner_value.size != 1:
                raise ValueError(f"Expected scalar env_action_owner, got shape {owner_value.shape}")
            owner_value = owner_value.reshape(-1)[0]
        elif isinstance(owner_value, np.generic):
            owner_value = owner_value.item()
        current = {
            "event_uid": event_key,
            "uid": str(uid[row]),
            "traj_uid": str(traj_index[row]),
            "event_index": int(_event_scalar(event_index[row])),
            "event_count": int(_event_scalar(event_count[row])),
            "role_event_index": int(_event_scalar(role_event_index[row])),
            "role_event_count": int(_event_scalar(role_event_count[row])),
            "agent_id": str(agent_id[row]),
            "event_type": str(event_type[row]),
            "env_step_index": int(_event_scalar(env_step_index[row])),
            "is_env_action": _event_bool(is_env_action[row]),
            "env_action_owner": None if owner_value is None else str(owner_value),
            "env_done": _event_bool(env_done[row]),
            "state_key": _event_state_key(state_chats[row]),
            "is_action_valid": _event_bool(is_action_valid[row], default=True),
            "env_reward": _event_scalar(env_reward[row], default=0.0),
            "kl_shaping": float(kl_shaping[row]) if include_kl_shaping else 0.0,
        }
        if boundary_enabled:
            current.update({
                "route_target": boundary_text(route_target[row], "route_target"),
                "route_reason": boundary_text(route_reason[row], "route_reason"),
                "tool_observation": boundary_text(tool_observation[row], "tool_observation"),
                "boundary_kl_shaping": float(boundary_kl_shaping[row]) if include_kl_shaping else 0.0,
            })
            if boundary_active:
                current["kl_shaping"] = current["boundary_kl_shaping"]
            if not all(math.isfinite(current[name]) for name in (
                "env_reward", "kl_shaping", "boundary_kl_shaping"
            )):
                raise ValueError(f"Non-finite boundary_loto reward for event_uid={event_key}")
            if boundary_config.auxiliary_mode == "env_only" and abs(current["boundary_kl_shaping"]) > epsilon:
                raise ValueError("boundary_loto env_only cannot consume nonzero reward-KL shaping")
        if math_enabled:
            text_value = executed_action_text[row]
            if isinstance(text_value, np.ndarray):
                if text_value.size != 1:
                    raise ValueError(f"math_answer_loto executed_action_text must be one scalar per row ({event_key})")
                text_value = text_value.reshape(-1)[0]
            if isinstance(text_value, np.generic):
                text_value = text_value.item()
            if not isinstance(text_value, str):
                raise ValueError(
                    f"math_answer_loto executed_action_text must be text for event_uid={event_key}"
                )
            current["executed_action_text"] = text_value
            current["math_kl_shaping"] = float(kl_shaping[row])
            if not math.isfinite(current["math_kl_shaping"]) or not math.isfinite(current["env_reward"]):
                raise ValueError(f"Non-finite math_answer_loto reward for event_uid={event_key}")
            if math_config.auxiliary_mode == "env_only" and abs(current["math_kl_shaping"]) > epsilon:
                raise ValueError("math_answer_loto env_only cannot consume nonzero reward-KL shaping")
        previous = events.get(event_key)
        if previous is not None:
            identity_keys = (
                "uid",
                "traj_uid",
                "event_index",
                "event_count",
                "role_event_index",
                "role_event_count",
                "agent_id",
                "event_type",
                "env_step_index",
                "is_env_action",
                "env_action_owner",
                "env_done",
                "state_key",
                "is_action_valid",
            )
            if boundary_enabled:
                identity_keys += ("route_target", "route_reason", "tool_observation")
            if math_enabled:
                identity_keys += ("executed_action_text",)
            if any(previous[key] != current[key] for key in identity_keys):
                raise ValueError(f"Copied rows disagree on metadata for event_uid={event_key}")
            if abs(previous["env_reward"] - current["env_reward"]) > epsilon:
                raise ValueError(f"Copied rows disagree on env_reward for event_uid={event_key}")
            if abs(previous["kl_shaping"] - current["kl_shaping"]) > epsilon:
                raise ValueError(f"Copied rows disagree on KL shaping for event_uid={event_key}")
            if boundary_enabled and abs(previous["boundary_kl_shaping"] - current["boundary_kl_shaping"]) > epsilon:
                raise ValueError(f"Copied rows disagree on boundary KL shaping for event_uid={event_key}")
            continue

        is_declared_owner = current["env_action_owner"] == current["agent_id"]
        if current["is_env_action"] != is_declared_owner:
            raise ValueError(
                f"Event ownership mismatch for {event_key}: agent={current['agent_id']}, "
                f"env_action_owner={current['env_action_owner']}, "
                f"is_env_action={current['is_env_action']}"
            )
        if not current["is_env_action"] and abs(current["env_reward"]) > epsilon:
            raise ValueError(
                f"Non-owner event {event_key} carries env_reward={current['env_reward']}; "
                "this would duplicate team reward"
            )
        invalid_penalty = 0.0
        if use_invalid_action_penalty and not current["is_action_valid"]:
            invalid_penalty = -float(invalid_action_penalty_coef)
        if boundary_enabled:
            current["invalid_reward"] = invalid_penalty
            current["boundary_reward"] = (
                current["env_reward"] + invalid_penalty + current["boundary_kl_shaping"]
            )
            if not math.isfinite(current["boundary_reward"]):
                raise ValueError(f"Non-finite boundary_loto used reward for event_uid={event_key}")
        if math_enabled:
            current["math_invalid_reward"] = invalid_penalty
            current["math_used_reward"] = (
                current["env_reward"] + invalid_penalty + current["math_kl_shaping"]
            )
            if not math.isfinite(current["math_used_reward"]):
                raise ValueError(f"Non-finite math_answer_loto used reward for event_uid={event_key}")
        current["reward"] = current["env_reward"] + invalid_penalty + current["kl_shaping"]
        events[event_key] = current

    trajectories: Dict[str, List[str]] = defaultdict(list)
    for key, event in events.items():
        trajectories[event["traj_uid"]].append(key)
    for trajectory, keys in trajectories.items():
        keys.sort(key=lambda key: events[key]["event_index"])
        indices = [events[key]["event_index"] for key in keys]
        expected_count = events[keys[0]]["event_count"]
        if any(events[key]["event_count"] != expected_count for key in keys):
            raise ValueError(f"Trajectory {trajectory} has inconsistent event_count metadata")
        if indices != list(range(expected_count)):
            raise ValueError(
                f"Trajectory {trajectory} is incomplete after batch adjustment: "
                f"event indices={indices}, expected={list(range(expected_count))}"
            )
        events_by_step: Dict[int, List[str]] = defaultdict(list)
        for key in keys:
            events_by_step[events[key]["env_step_index"]].append(key)
        for step, step_keys in events_by_step.items():
            owners = [key for key in step_keys if events[key]["is_env_action"]]
            if len(owners) != 1:
                raise ValueError(
                    f"Trajectory {trajectory} env step {step} has {len(owners)} action owners; expected 1"
                )
            if owners[0] != step_keys[-1]:
                raise ValueError(
                    f"Trajectory {trajectory} env step {step} does not end with its action owner"
                )
        if any(events[key]["env_done"] for key in keys[:-1]):
            raise ValueError(f"Trajectory {trajectory} contains events after env_done")
        if not events[keys[-1]]["env_done"]:
            raise ValueError(
                f"Trajectory {trajectory} does not end in an env_done event; "
                "team_event_gae refuses to bootstrap across a truncated chain"
            )
        role_keys: Dict[str, List[str]] = defaultdict(list)
        for key in keys:
            role_keys[events[key]["agent_id"]].append(key)
        for role, current_role_keys in role_keys.items():
            role_indices = [events[key]["role_event_index"] for key in current_role_keys]
            expected_role_count = events[current_role_keys[0]]["role_event_count"]
            if any(events[key]["role_event_count"] != expected_role_count for key in current_role_keys):
                raise ValueError(
                    f"Trajectory {trajectory}/{role} has inconsistent role_event_count metadata"
                )
            if role_indices != list(range(expected_role_count)):
                raise ValueError(
                    f"Trajectory {trajectory}/{role} has invalid role-event indices "
                    f"{role_indices}; expected {list(range(expected_role_count))}"
                )

    transition_gamma: Dict[str, float] = {}
    transition_lam: Dict[str, float] = {}
    monte_carlo_return: Dict[str, float] = {}
    for keys in trajectories.values():
        running_return = 0.0
        for key in reversed(keys):
            event = events[key]
            if event["env_done"]:
                edge_gamma = 0.0
                edge_lam = 0.0
            elif event["is_env_action"]:
                edge_gamma = float(gamma)
                edge_lam = float(lam)
            else:
                edge_gamma = float(internal_gamma)
                edge_lam = float(internal_lam)
            transition_gamma[key] = edge_gamma
            transition_lam[key] = edge_lam
            running_return = event["reward"] + edge_gamma * running_return
            monte_carlo_return[key] = running_return

    exact_groups: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    depth_groups: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    role_groups: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for key, event in events.items():
        trajectory = event["traj_uid"]
        exact_key = (event["uid"], event["agent_id"], event["event_type"], event["state_key"])
        depth_key = (
            event["uid"],
            event["agent_id"],
            event["event_type"],
            event["env_step_index"],
        )
        role_key = (event["uid"], event["agent_id"], event["event_type"])
        exact_groups[exact_key][trajectory].append(monte_carlo_return[key])
        depth_groups[depth_key][trajectory].append(monte_carlo_return[key])
        role_groups[role_key][trajectory].append(monte_carlo_return[key])

    def loto_value(group: Dict[str, List[float]], own_trajectory: str) -> Optional[float]:
        peer_values = [
            float(np.mean(values))
            for trajectory, values in group.items()
            if trajectory != own_trajectory
        ]
        return float(np.mean(peer_values)) if peer_values else None

    values: Dict[str, float] = {}
    for key, event in events.items():
        exact_key = (event["uid"], event["agent_id"], event["event_type"], event["state_key"])
        depth_key = (
            event["uid"],
            event["agent_id"],
            event["event_type"],
            event["env_step_index"],
        )
        role_key = (event["uid"], event["agent_id"], event["event_type"])
        estimate = loto_value(exact_groups[exact_key], event["traj_uid"])
        if estimate is None:
            estimate = loto_value(depth_groups[depth_key], event["traj_uid"])
        if estimate is None:
            estimate = loto_value(role_groups[role_key], event["traj_uid"])
        values[key] = 0.0 if estimate is None else estimate

    boundary_result = None
    boundary_task_raw: Dict[str, float] = {}
    boundary_train_raw: Dict[str, float] = {}
    boundary_delta_env: Dict[str, float] = {}
    boundary_delta_used: Dict[str, float] = {}
    if boundary_enabled:
        boundary_result = estimate_boundary_values(events, trajectories, boundary_config)
        for keys in trajectories.values():
            next_task = next_train = 0.0
            for key in reversed(keys):
                event = events[key]
                bootstrap = (
                    transition_gamma[key] * boundary_result.post[key] - boundary_result.pre[key]
                )
                task_delta = event["env_reward"] + bootstrap
                train_delta = event["boundary_reward"] + bootstrap
                coefficient = transition_gamma[key] * transition_lam[key]
                next_task = task_delta + coefficient * next_task
                next_train = train_delta + coefficient * next_train
                boundary_delta_env[key] = task_delta
                boundary_delta_used[key] = train_delta
                boundary_task_raw[key] = next_task
                boundary_train_raw[key] = next_train
        if boundary_active:
            # The existing team GAE uses next pre as next value. The new
            # estimator enforces post[e] == pre[e+1], so no actor-side logic
            # or producer routing needs to change.
            values = dict(boundary_result.pre)

    math_result = None
    math_task_raw: Dict[str, float] = {}
    math_train_raw: Dict[str, float] = {}
    math_delta_env: Dict[str, float] = {}
    math_delta_used: Dict[str, float] = {}
    if math_enabled:
        math_result = estimate_math_values(events, trajectories, math_config)
        for keys in trajectories.values():
            next_task = next_train = 0.0
            for key in reversed(keys):
                event = events[key]
                bootstrap = (
                    transition_gamma[key] * math_result.post[key] - math_result.pre[key]
                )
                task_delta = event["env_reward"] + bootstrap
                train_delta = event["math_used_reward"] + bootstrap
                coefficient = transition_gamma[key] * transition_lam[key]
                next_task = task_delta + coefficient * next_task
                next_train = train_delta + coefficient * next_train
                math_delta_env[key] = task_delta
                math_delta_used[key] = train_delta
                math_task_raw[key] = next_task
                math_train_raw[key] = next_train
        # Same identity as boundary_loto: post[e] == pre[e+1] and the terminal
        # post is zero, so the general team GAE recursion below computes these
        # exact advantages with values replaced by the math pre estimates.
        values = dict(math_result.pre)

    team_raw_advantages: Dict[str, float] = {}
    team_lambda_returns: Dict[str, float] = {}
    for keys in trajectories.values():
        next_advantage = 0.0
        for position in reversed(range(len(keys))):
            key = keys[position]
            next_value = values[keys[position + 1]] if position + 1 < len(keys) else 0.0
            delta = events[key]["reward"] + transition_gamma[key] * next_value - values[key]
            advantage = delta + transition_gamma[key] * transition_lam[key] * next_advantage
            team_raw_advantages[key] = advantage
            team_lambda_returns[key] = advantage + values[key]
            next_advantage = advantage

    local_raw_advantages: Dict[str, float] = {}
    local_lambda_returns: Dict[str, float] = {}
    local_deltas: Dict[str, float] = {}
    local_segment_rewards: Dict[str, float] = {}
    local_segment_gammas: Dict[str, float] = {}
    local_segment_lambdas: Dict[str, float] = {}
    local_span_events: Dict[str, float] = {}
    local_span_env_steps: Dict[str, float] = {}
    if local_mode != "off":
        for keys in trajectories.values():
            positions_by_agent: Dict[str, List[int]] = defaultdict(list)
            for position, key in enumerate(keys):
                positions_by_agent[events[key]["agent_id"]].append(position)

            for positions in positions_by_agent.values():
                next_local_advantage = 0.0
                for local_position in reversed(range(len(positions))):
                    start = positions[local_position]
                    stop = positions[local_position + 1] if local_position + 1 < len(positions) else len(keys)
                    key = keys[start]

                    segment_reward = 0.0
                    segment_gamma = 1.0
                    segment_lambda = 1.0
                    env_steps = 0
                    for global_position in range(start, stop):
                        segment_key = keys[global_position]
                        segment_event = events[segment_key]
                        segment_reward += segment_gamma * segment_event["reward"]
                        if segment_event["is_env_action"]:
                            env_steps += 1
                        edge_gamma = transition_gamma[segment_key]
                        if segment_event["env_done"]:
                            edge_local_lambda = 0.0
                        elif segment_event["is_env_action"]:
                            edge_local_lambda = float(agent_local_lam)
                        else:
                            edge_local_lambda = float(agent_local_internal_lam)
                        segment_gamma *= edge_gamma
                        segment_lambda *= edge_local_lambda

                    next_value = values[keys[stop]] if stop < len(keys) else 0.0
                    delta = segment_reward + segment_gamma * next_value - values[key]
                    advantage = (
                        delta
                        + segment_gamma * segment_lambda * next_local_advantage
                    )
                    local_raw_advantages[key] = advantage
                    local_lambda_returns[key] = advantage + values[key]
                    local_deltas[key] = delta
                    local_segment_rewards[key] = segment_reward
                    local_segment_gammas[key] = segment_gamma
                    local_segment_lambdas[key] = segment_lambda
                    local_span_events[key] = float(stop - start)
                    local_span_env_steps[key] = float(env_steps)
                    next_local_advantage = advantage

    def normalize_event_advantages(raw_values: Dict[str, float]) -> Dict[str, float]:
        normalized = dict(raw_values)
        if not normalize_advantages:
            return normalized
        normalization_groups: Dict[Any, List[str]] = defaultdict(list)
        for key, event in events.items():
            normalization_groups[(event["uid"], event["agent_id"])].append(key)
        for keys in normalization_groups.values():
            distinct_trajectories = {events[key]["traj_uid"] for key in keys}
            # With no peer trajectory, keep the valid V=0 policy-gradient
            # signal instead of silently erasing a rare role/event cohort.
            if len(distinct_trajectories) < 2:
                continue
            counts = Counter(events[key]["traj_uid"] for key in keys)
            weights = np.array(
                [1.0 / counts[events[key]["traj_uid"]] for key in keys],
                dtype=np.float64,
            )
            samples = np.array([raw_values[key] for key in keys], dtype=np.float64)
            mean = float(np.sum(weights * samples) / np.sum(weights))
            centered = samples - mean
            if norm_adv_by_std:
                variance = float(np.sum(weights * centered * centered) / np.sum(weights))
                scale = float(np.sqrt(max(variance, 0.0)))
                if not np.isfinite(scale) or scale <= epsilon:
                    centered[:] = 0.0
                else:
                    centered /= scale + epsilon
            for key, advantage in zip(keys, centered):
                normalized[key] = float(advantage)
        return normalized

    team_normalized_advantages = normalize_event_advantages(team_raw_advantages)
    local_normalized_advantages = (
        normalize_event_advantages(local_raw_advantages) if local_mode != "off" else {}
    )

    if local_mode == "blend" and float(agent_local_mix_alpha) > 0.0:
        alpha = float(agent_local_mix_alpha)
        raw_advantages = {
            key: (1.0 - alpha) * team_raw_advantages[key] + alpha * local_raw_advantages[key]
            for key in events
        }
        normalized_advantages = normalize_event_advantages(raw_advantages)
        lambda_returns = {key: raw_advantages[key] + values[key] for key in events}
    else:
        # Keep the exact v3.1.0 dictionaries on off/shadow/alpha=0 paths.
        raw_advantages = dict(team_raw_advantages)
        normalized_advantages = dict(team_normalized_advantages)
        lambda_returns = dict(team_lambda_returns)

    device = response_mask.device
    dtype = token_level_rewards.dtype
    def event_map_to_rows(mapping: Dict[str, float]) -> torch.Tensor:
        output = torch.empty(bsz, dtype=dtype, device=device)
        for key, rows in rows_by_event.items():
            for row in rows:
                output[row] = mapping[key]
        return output

    row_advantages = event_map_to_rows(normalized_advantages)
    row_returns = event_map_to_rows(lambda_returns)
    row_values = event_map_to_rows(values)
    row_rewards = event_map_to_rows({key: event["reward"] for key, event in events.items()})
    row_raw_advantages = event_map_to_rows(raw_advantages)

    diagnostics = {
        "event_advantages": row_advantages,
        "event_raw_advantages": row_raw_advantages,
        "event_returns": row_returns,
        "event_values": row_values,
        "event_rewards": row_rewards,
        "event_team_advantages": event_map_to_rows(team_normalized_advantages),
        "event_team_raw_advantages": event_map_to_rows(team_raw_advantages),
        "event_team_returns": event_map_to_rows(team_lambda_returns),
    }
    if boundary_enabled:
        boundary_normalized = normalize_event_advantages(boundary_train_raw)
        normalization_mean = {key: 0.0 for key in events}
        normalization_std = {key: 0.0 for key in events}
        normalization_applied = {key: 0.0 for key in events}
        if normalize_advantages:
            cohorts = defaultdict(list)
            for key, event in events.items():
                cohorts[(event["uid"], event["agent_id"])].append(key)
            for keys in cohorts.values():
                counts = Counter(events[key]["traj_uid"] for key in keys)
                if len(counts) < 2:
                    continue
                weights = np.asarray([1.0 / counts[events[key]["traj_uid"]] for key in keys])
                samples = np.asarray([boundary_train_raw[key] for key in keys], dtype=np.float64)
                mean = float(np.sum(weights * samples) / np.sum(weights))
                variance = float(np.sum(weights * (samples - mean) ** 2) / np.sum(weights))
                std = math.sqrt(max(variance, 0.0))
                for key in keys:
                    normalization_mean[key] = mean
                    normalization_std[key] = std
                    normalization_applied[key] = 1.0
        boundary_maps = {
            "value_pre": boundary_result.pre,
            "value_post": boundary_result.post,
            "delta_env": boundary_delta_env,
            "delta_used": boundary_delta_used,
            "task_raw_advantages": boundary_task_raw,
            "aux_raw_advantages": {key: boundary_train_raw[key] - boundary_task_raw[key] for key in events},
            "raw_advantages": boundary_train_raw,
            "advantages": boundary_normalized,
            "reward_env": {key: event["env_reward"] for key, event in events.items()},
            "reward_invalid": {key: event["invalid_reward"] for key, event in events.items()},
            "reward_kl": {key: event["boundary_kl_shaping"] for key, event in events.items()},
            "reward_used": {key: event["boundary_reward"] for key, event in events.items()},
            "gamma": transition_gamma,
            "lambda": transition_lam,
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "normalization_applied": normalization_applied,
        }
        for output_name, detail_name in (
            ("parent", "parent"), ("question_parent", "qparent"),
            ("same_round_peer_count", "n_same"), ("kernel_mass", "mass"),
            ("kernel_ess", "ess"), ("positive_peer_count", "positive_peer_count"),
            ("cross_round_peer_count", "cross_round_peer_count"),
        ):
            boundary_maps[output_name] = {
                key: float(boundary_result.details.get(key, {}).get(detail_name, 0.0))
                for key in events
            }
        diagnostics.update({
            f"event_boundary_{name}": event_map_to_rows(mapping)
            for name, mapping in boundary_maps.items()
        })
        if diagnostics_meta is not None:
            diagnostics_meta.update({
                "schema_version": "team_event_boundary_credit_v1",
                "value_mode": boundary_config.mode,
                "application": boundary_config.application,
                "auxiliary_mode": boundary_config.auxiliary_mode,
                "idf_sha256": boundary_result.idf_sha256,
                "details_by_event": boundary_result.details,
            })
    if math_enabled:
        math_normalized = normalize_event_advantages(math_train_raw)
        math_normalization_mean = {key: 0.0 for key in events}
        math_normalization_std = {key: 0.0 for key in events}
        math_normalization_applied = {key: 0.0 for key in events}
        if normalize_advantages:
            cohorts = defaultdict(list)
            for key, event in events.items():
                cohorts[(event["uid"], event["agent_id"])].append(key)
            for keys in cohorts.values():
                counts = Counter(events[key]["traj_uid"] for key in keys)
                if len(counts) < 2:
                    continue
                weights = np.asarray([1.0 / counts[events[key]["traj_uid"]] for key in keys])
                samples = np.asarray([math_train_raw[key] for key in keys], dtype=np.float64)
                mean = float(np.sum(weights * samples) / np.sum(weights))
                variance = float(np.sum(weights * (samples - mean) ** 2) / np.sum(weights))
                std = math.sqrt(max(variance, 0.0))
                for key in keys:
                    math_normalization_mean[key] = mean
                    math_normalization_std[key] = std
                    math_normalization_applied[key] = 1.0
        math_maps = {
            "value_pre": math_result.pre,
            "value_post": math_result.post,
            "delta_env": math_delta_env,
            "delta_used": math_delta_used,
            "task_raw_advantages": math_task_raw,
            "aux_raw_advantages": {key: math_train_raw[key] - math_task_raw[key] for key in events},
            "raw_advantages": math_train_raw,
            "advantages": math_normalized,
            "reward_env": {key: event["env_reward"] for key, event in events.items()},
            "reward_invalid": {key: event["math_invalid_reward"] for key, event in events.items()},
            "reward_kl": {key: event["math_kl_shaping"] for key, event in events.items()},
            "reward_used": {key: event["math_used_reward"] for key, event in events.items()},
            "gamma": transition_gamma,
            "lambda": transition_lam,
            "normalization_mean": math_normalization_mean,
            "normalization_std": math_normalization_std,
            "normalization_applied": math_normalization_applied,
        }
        for output_name, detail_name in (
            ("parent", "parent"), ("question_parent", "question_parent"),
            ("same_round_peer_count", "same_round_peer_count"), ("kernel_mass", "kernel_mass"),
            ("kernel_ess", "kernel_ess"), ("effective_ess", "effective_ess"),
            ("max_effective_coefficient", "max_effective_coefficient"),
            ("used_peer_count", "used_peer_count"),
            ("cross_round_peer_count", "cross_round_peer_count"),
        ):
            math_maps[output_name] = {
                key: float(math_result.details.get(key, {}).get(detail_name, 0.0))
                for key in events
            }
        diagnostics.update({
            f"event_math_{name}": event_map_to_rows(mapping)
            for name, mapping in math_maps.items()
        })
        if diagnostics_meta is not None:
            diagnostics_meta.update({
                "schema_version": "team_event_math_credit_v1",
                "value_mode": math_config.mode,
                "auxiliary_mode": math_config.auxiliary_mode,
                "parser_version": math_config.parser_version,
                "details_by_event": math_result.details,
            })
    if local_mode != "off":
        diagnostics.update(
            {
                "event_local_advantages": event_map_to_rows(local_normalized_advantages),
                "event_local_raw_advantages": event_map_to_rows(local_raw_advantages),
                "event_local_returns": event_map_to_rows(local_lambda_returns),
                "event_local_deltas": event_map_to_rows(local_deltas),
                "event_local_segment_rewards": event_map_to_rows(local_segment_rewards),
                "event_local_segment_gammas": event_map_to_rows(local_segment_gammas),
                "event_local_segment_lambdas": event_map_to_rows(local_segment_lambdas),
                "event_local_span_events": event_map_to_rows(local_span_events),
                "event_local_span_env_steps": event_map_to_rows(local_span_env_steps),
                "event_mix_alpha": torch.full(
                    (bsz,),
                    float(agent_local_mix_alpha) if local_mode == "blend" else 0.0,
                    dtype=dtype,
                    device=device,
                ),
            }
        )
    return (
        row_advantages.unsqueeze(-1) * response_mask,
        row_returns.unsqueeze(-1) * response_mask,
        diagnostics,
    )


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    print("group_by_agent_id: ", group_by_agent_id)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj_accumulator = defaultdict(list)
    traj2avg = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])
        
        for (idx, t_idx), reward_list in traj_accumulator.items():
            if group_by_agent_id:
                id2score[idx].extend(reward_list)
            else:
                avg_score = torch.stack(reward_list).mean()
                traj2avg[(idx, t_idx)] = avg_score
                id2score[idx].append(avg_score)
        if not group_by_agent_id:
            for i in range(bsz):
                scores[i] = traj2avg[(index[i], traj_index[i])]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_drmas_fixed_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """Dr.MAS-fixed (trajectory-balanced Dr.MAS, TB-Dr.MAS) outcome advantage.

    Removes the original Dr.MAS event-row over-weighting: each (task, trajectory,
    role) contributes exactly ONE macro score = mean of its real event scores, and
    macro scores are standardized across the DISTINCT trajectories of the same
    (task, role). A trajectory with more events no longer dominates the role
    baseline.

    Contract (EVENT_HCAPO_END_TO_END_GUIDE.md §2.2 / SEARCH_PARALLEL_EXPERIMENT_GUIDE.md §8):
      - ``index`` MUST already encode (task, agent_id); ``traj_index`` is traj_uid.
      - S_(q,tau,role) = mean(s_e over real events of this trajectory-role).
      - A_fixed = (S - mean_unique_trajectory_scores) / (std + eps) within (task, role).
      - cohorts with <2 distinct trajectories OR zero variance -> A_fixed = 0.
      - a trajectory-role with a single event is NOT a cohort singleton (not zeroed).
      - normalization uses DISTINCT trajectories, so duplicating/reordering whole
        trajectories does not change any real event's advantage; ``response_mask``
        zeroes padding-row per-token advantage.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group key = (task, agent_id)
        traj_index: (bs,) traj_uid
    """
    scores = token_level_rewards.sum(dim=-1)  # per-event s_e, (bs,)

    traj_accumulator: dict = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])

        # macro score per (task, traj, role) = mean of its real event scores
        traj2macro: dict = {}
        for key, reward_list in traj_accumulator.items():
            traj2macro[key] = torch.stack(reward_list).mean()

        # collect DISTINCT-trajectory macro scores per (task, role)
        id2macros: dict = defaultdict(list)
        for (idx, _traj), macro in traj2macro.items():
            id2macros[idx].append(macro)

        id2mean: dict = {}
        id2std: dict = {}
        degenerate: set = set()  # cohorts with <2 distinct trajs or zero variance -> A=0
        for idx, macros in id2macros.items():
            if len(macros) < 2:
                degenerate.add(idx)
                continue
            stacked = torch.stack(macros)
            std = torch.std(stacked)
            if (not torch.isfinite(std)) or std.item() == 0.0:
                degenerate.add(idx)
                continue
            id2mean[idx] = torch.mean(stacked)
            id2std[idx] = std

        out = torch.zeros_like(scores)
        for i in range(bsz):
            if index[i] in degenerate:
                continue  # leave out[i] = 0 for degenerate cohorts
            macro = traj2macro[(index[i], traj_index[i])]
            if norm_adv_by_std_in_grpo:
                out[i] = (macro - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                out[i] = macro - id2mean[index[i]]
        out = out.unsqueeze(-1) * response_mask

    return out, out


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def _validate_sample_weight(sample_weight: torch.Tensor, loss_mat: torch.Tensor) -> torch.Tensor:
    """Return a finite, positive row-weight vector on ``loss_mat``'s device."""
    if sample_weight.ndim == 2 and sample_weight.shape[-1] == 1:
        sample_weight = sample_weight.squeeze(-1)
    if sample_weight.ndim != 1 or sample_weight.shape[0] != loss_mat.shape[0]:
        raise ValueError(
            "sample_weight must have shape (batch_size,) or (batch_size, 1), "
            f"got {tuple(sample_weight.shape)} for loss shape {tuple(loss_mat.shape)}"
        )
    sample_weight = sample_weight.to(device=loss_mat.device, dtype=torch.float32)
    if not torch.isfinite(sample_weight).all() or torch.any(sample_weight <= 0):
        raise ValueError("sample_weight must contain only finite positive values")
    return sample_weight


def compute_sample_weight_normalizer(
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    loss_agg_mode: str,
) -> torch.Tensor:
    """Compute the local denominator for duplicate-aware loss aggregation.

    The returned scalar is intentionally computed for a complete optimizer
    mini-batch.  Callers that split it into micro-batches must reuse the same
    denominator for every micro-batch; normalizing each micro-batch separately
    would reintroduce duplicate-dependent gradients.
    """
    if sample_weight.ndim == 2 and sample_weight.shape[-1] == 1:
        sample_weight = sample_weight.squeeze(-1)
    if sample_weight.ndim != 1 or sample_weight.shape[0] != loss_mask.shape[0]:
        raise ValueError(
            "sample_weight must align with loss_mask rows, "
            f"got weight={tuple(sample_weight.shape)}, mask={tuple(loss_mask.shape)}"
        )
    sample_weight = sample_weight.to(device=loss_mask.device, dtype=torch.float32)
    if not torch.isfinite(sample_weight).all() or torch.any(sample_weight <= 0):
        raise ValueError("sample_weight must contain only finite positive values")

    if loss_agg_mode == "token-mean":
        normalizer = torch.sum(loss_mask.to(dtype=sample_weight.dtype) * sample_weight.unsqueeze(-1))
    elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-mean"):
        normalizer = torch.sum(sample_weight)
    else:
        raise ValueError(
            "Duplicate-aware weighting currently supports token-mean, "
            "seq-mean-token-sum, and seq-mean-token-mean; "
            f"got {loss_agg_mode!r}"
        )
    if not torch.isfinite(normalizer) or normalizer <= 0:
        raise ValueError(f"sample_weight normalizer must be finite and positive, got {normalizer}")
    return normalizer


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    sample_weight: torch.Tensor | None = None,
    sample_weight_normalizer: torch.Tensor | float | None = None,
    sample_weight_scale: float = 1.0,
):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
        sample_weight: optional inverse-duplication weight per sequence.
        sample_weight_normalizer: optional denominator computed over the full
            optimizer mini-batch (and, for distributed actor training, all DP
            ranks). Reuse it across every micro-batch.
        sample_weight_scale: compensates for distributed gradient averaging;
            normally the DP world size when a global denominator is supplied.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if sample_weight is not None:
        # Accumulate weighted objectives in float32 even when the model runs
        # in FP16/BF16. This avoids precision loss in large copied/token
        # batches while preserving gradients through the dtype conversion.
        sample_weight = _validate_sample_weight(sample_weight, loss_mat).to(dtype=torch.float32)
        loss_values = loss_mat.to(dtype=torch.float32)
        mask = loss_mask.to(dtype=torch.float32)
        if loss_agg_mode == "token-mean":
            numerator = torch.sum(loss_values * mask * sample_weight.unsqueeze(-1))
        elif loss_agg_mode == "seq-mean-token-sum":
            seq_losses = torch.sum(loss_values * mask, dim=-1)
            numerator = torch.sum(seq_losses * sample_weight)
        elif loss_agg_mode == "seq-mean-token-mean":
            token_counts = torch.sum(mask, dim=-1)
            if torch.any(token_counts <= 0):
                raise ValueError("seq-mean-token-mean requires at least one valid token per weighted row")
            seq_losses = torch.sum(loss_values * mask, dim=-1) / token_counts
            numerator = torch.sum(seq_losses * sample_weight)
        else:
            raise ValueError(
                "Duplicate-aware weighting currently supports token-mean, "
                "seq-mean-token-sum, and seq-mean-token-mean; "
                f"got {loss_agg_mode!r}"
            )

        if sample_weight_normalizer is None:
            normalizer = compute_sample_weight_normalizer(loss_mask, sample_weight, loss_agg_mode)
        else:
            normalizer = torch.as_tensor(
                sample_weight_normalizer,
                dtype=torch.float32,
                device=loss_mat.device,
            )
            if not torch.isfinite(normalizer) or normalizer <= 0:
                raise ValueError(
                    "sample_weight_normalizer must be finite and positive, "
                    f"got {sample_weight_normalizer}"
                )
        if not math.isfinite(float(sample_weight_scale)) or float(sample_weight_scale) <= 0:
            raise ValueError(f"sample_weight_scale must be finite and positive, got {sample_weight_scale}")
        return numerator / normalizer * float(sample_weight_scale)

    if sample_weight_normalizer is not None or float(sample_weight_scale) != 1.0:
        raise ValueError("sample_weight_normalizer/scale require sample_weight")

    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    sample_weight: torch.Tensor | None = None,
    sample_weight_normalizer: torch.Tensor | float | None = None,
    sample_weight_scale: float = 1.0,
    sample_weight_metric_normalizer: torch.Tensor | float | None = None,
    sample_weight_metric_scale: float = 1.0,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        sample_weight (torch.Tensor, optional):
            Inverse event-duplication weight per sequence.
        sample_weight_normalizer (torch.Tensor or float, optional):
            Full mini-batch/global-DP denominator reused by every micro-batch.
        sample_weight_scale (float, optional):
            DP world-size compensation when gradients are averaged.
        sample_weight_metric_normalizer (torch.Tensor or float, optional):
            Full-mini/global-DP weighted-token denominator for PPO diagnostics.
        sample_weight_metric_scale (float, optional):
            Compensation for later averaging across DP ranks and micro-batches.
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    if sample_weight is None:
        if sample_weight_metric_normalizer is not None or float(sample_weight_metric_scale) != 1.0:
            raise ValueError("sample_weight_metric_normalizer/scale require sample_weight")
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    else:
        ppo_kl = agg_loss(
            loss_mat=-negative_approx_kl,
            loss_mask=response_mask,
            loss_agg_mode="token-mean",
            sample_weight=sample_weight,
            sample_weight_normalizer=sample_weight_metric_normalizer,
            sample_weight_scale=sample_weight_metric_scale,
        )

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    clipfrac_mat = torch.gt(pg_losses2, pg_losses1).float()
    if sample_weight is None:
        pg_clipfrac = verl_F.masked_mean(clipfrac_mat, response_mask)
    else:
        pg_clipfrac = agg_loss(
            loss_mat=clipfrac_mat,
            loss_mask=response_mask,
            loss_agg_mode="token-mean",
            sample_weight=sample_weight,
            sample_weight_normalizer=sample_weight_metric_normalizer,
            sample_weight_scale=sample_weight_metric_scale,
        )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    clipfrac_lower_mat = torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float()
    if sample_weight is None:
        pg_clipfrac_lower = verl_F.masked_mean(clipfrac_lower_mat, response_mask)
    else:
        pg_clipfrac_lower = agg_loss(
            loss_mat=clipfrac_lower_mat,
            loss_mask=response_mask,
            loss_agg_mode="token-mean",
            sample_weight=sample_weight,
            sample_weight_normalizer=sample_weight_metric_normalizer,
            sample_weight_scale=sample_weight_metric_scale,
        )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        sample_weight=sample_weight,
        sample_weight_normalizer=sample_weight_normalizer,
        sample_weight_scale=sample_weight_scale,
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
