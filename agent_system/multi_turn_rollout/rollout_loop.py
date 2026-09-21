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

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn

from agent_system.event_trace import (
    MATH_EVENT_TRACE_SCHEMA_VERSION,
    SEARCH_EVENT_TRACE_SCHEMA_VERSION,
    annotate_trajectory_events,
    build_offline_eval_info,
    build_search_transition_fields,
    build_task_uid,
    event_metadata_enabled,
    event_trace_option,
    resolve_search_env_action_owners,
)
from agent_system.math_event_trace import (
    annotate_math_step_events,
    finalize_math_trajectory_events,
)
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, filter_group_data, preprocess_batch
from agent_system.environments import EnvironmentManagerBase
from agent_system.agent import BaseOrchestra

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment

        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # success_rate = {}
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            trajectory_events = [data for data in total_batch_list[bs] if data['active_masks']]
            if event_metadata_enabled(self.config):
                task_types = {str(data.get('task_type')) for data in trajectory_events}
                if len(task_types) != 1:
                    raise ValueError(f"Trajectory {traj_uid[bs]} has inconsistent task types: {task_types}")
                task_type = next(iter(task_types))
                schema_version = (
                    SEARCH_EVENT_TRACE_SCHEMA_VERSION
                    if task_type == 'search'
                    else MATH_EVENT_TRACE_SCHEMA_VERSION
                )
                annotate_trajectory_events(
                    trajectory_events,
                    str(traj_uid[bs]),
                    schema_version=schema_version,
                )
                if task_type == 'math':
                    # Resolve submitted/transition UIDs and enforce the complete
                    # Math chain contract before any downstream consumer.
                    finalize_math_trajectory_events(
                        trajectory_events,
                        str(traj_uid[bs]),
                        max_loop_num=int(self.config.agent.orchestra.math.max_loop_num),
                    )
                for event in trajectory_events:
                    # Reuse the authoritative zxj event ordering for both mix
                    # stages and semantic prefix construction.
                    event['role_turn_index'] = int(event['role_event_index'])
                    if 'value_action_truncated' in event:
                        event['pure_entropy_truncated'] = bool(event['value_action_truncated'])
                    if 'value_question' in event:
                        event['value_action_index'] = int(event['event_index'])
                        event['value_action_text'] = event['executed_action_text']
            # sum the rewards for each data in total_batch_list[bs]
            for data in trajectory_events:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                # episode_rewards
                data['episode_rewards'] = episode_rewards[bs]
                # episode_lengths
                data['episode_lengths'] = episode_lengths[bs]
                # tool_callings
                data['tool_callings'] = tool_callings[bs]
                # success_rate
                # for key, value in success_rate.items():
                #     data[key] = value
                # pass
                data['pass'] = success['success_rate'][bs]

                effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)
        
        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = preprocess_batch(gen_batch=gen_batch, 
                                     obs=obs, 
                                     config=self.config, 
                                     tokenizer=self.tokenizer, 
                                     processor=self.processor,
                                     )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            effective_rollout_n: int,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
            if event_metadata_enabled(self.config):
                for trajectory_events in batch_list:
                    for event in trajectory_events:
                        event['sampling_try'] = try_count - 1
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Determine the effective rollout_n based on training/validation mode
        if is_train:
            effective_rollout_n = self.config.env.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
        else:
            # For validation, use val_n for pass@k and avg@k computation
            val_rollout_n = getattr(self.config.env.rollout, 'val_n', None)
            if val_rollout_n is not None and val_rollout_n > 1:
                effective_rollout_n = val_rollout_n
                gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
            else:
                effective_rollout_n = 1

        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output

# =============================================================================
# Multi‑Agent collector orchestrating a *team* of agents
# =============================================================================
class MultiAgentTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector that *delegates* action generation to a
    user‑configurable :class:`BaseOrchestra` (chain, hierarchy, etc.)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Any,
        wg_to_agents_mapping: Dict[str, List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any] = None,
    ):
        super().__init__(config=config, tokenizer=tokenizers, processor=processors)

        agent_ids = config.agent.agent_ids
        model_ids = config.agent.model_ids
        orchestra_type = config.agent.orchestra_type
        print("agent_ids: ", agent_ids)
        print("orchestra_type: ", orchestra_type)

        agents_to_wg_mapping = {}
        for wg_id, agents in wg_to_agents_mapping.items():
            for a in agents:
                agent_id = a['agent_id']
                agents_to_wg_mapping[agent_id] = wg_id

        if orchestra_type == "search":
            from agent_system.agent.orchestra.search import SearchMultiAgentOrchestra as orchestra
        elif orchestra_type == "math":
            from agent_system.agent.orchestra.math import MathMultiAgentOrchestra as orchestra
        else:
            raise ValueError(f"Unknown orchestra_type '{orchestra_type}'.")

        self.multiagent_orchestra: BaseOrchestra = orchestra(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )

    # ------------------------------------------------------------------
    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        effective_rollout_n: int,
    ):

        batch_size = len(gen_batch.batch)

        capture_event_trace = event_metadata_enabled(self.config)
        task_type = str(self.config.agent.orchestra_type)
        # dynamic_multi_turn_loop may retry with the same gen_batch.  Reading
        # instead of popping keeps retry behavior identical with trace on/off.
        env_kwargs = gen_batch.non_tensor_batch.get('env_kwargs', None)
        if getattr(self.multiagent_orchestra, "value_metadata_enabled", False):
            self.multiagent_orchestra.prepare_value_metadata(gen_batch, env_kwargs)
        reset_kwargs = deepcopy(env_kwargs)
        trace_env_kwargs = deepcopy(env_kwargs) if capture_event_trace else None
        obs, infos = envs.reset(kwargs=reset_kwargs)
        self.multiagent_orchestra.reset()
        
        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        terminal_actions = None
        if capture_event_trace:
            terminal_actions = np.empty(batch_size, dtype=object)
            terminal_actions[:] = None
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            ###############################
            text_actions, multiagent_batch_buffer = self.multiagent_orchestra.run(
                gen_batch=gen_batch,
                env_obs=obs,
                actor_rollout_wgs=actor_rollout_wg,
                active_masks=active_masks,
                step=_step+1
            )
            if task_type == 'search':
                search_action_mask = getattr(self.multiagent_orchestra, "last_search_action_mask", None)
                if search_action_mask is None:
                    raise RuntimeError(
                        "Search orchestra did not publish its route-derived "
                        "last_search_action_mask before env.step"
                    )
                next_obs, rewards, dones, infos = envs.step(
                    text_actions,
                    search_action_mask=search_action_mask,
                    active_mask=active_masks,
                )
            else:
                next_obs, rewards, dones, infos = envs.step(text_actions)
                if task_type == 'math' and capture_event_trace:
                    # Bind the single authoritative Math env.step to the real
                    # events in call order; reward/done come from env.step only.
                    annotate_math_step_events(
                        multiagent_batch_buffer,
                        active_masks,
                        rewards,
                        dones,
                    )
            if capture_event_trace:
                for item, is_active in enumerate(active_masks):
                    if is_active:
                        terminal_actions[item] = text_actions[item]
            ###############################
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            env_action_owners = None
            if capture_event_trace and task_type == 'search':
                route_targets = None
                agent_selections = []
                for data in multiagent_batch_buffer:
                    agent_id = data['agent_id']
                    agent_batch = data['batch']
                    if route_targets is None and 'route_target' in agent_batch.non_tensor_batch:
                        route_targets = agent_batch.non_tensor_batch['route_target']
                    selected_mask = agent_batch.non_tensor_batch.get('is_env_action')
                    if selected_mask is None:
                        raise ValueError(f"Search trace event from {agent_id} is missing is_env_action")
                    agent_selections.append((agent_id, selected_mask))
                if route_targets is None:
                    raise ValueError(f"Search trace is missing route_target at step {_step + 1}")
                env_action_owners = resolve_search_env_action_owners(
                    agent_selections,
                    active_masks,
                    route_targets,
                )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"

            for data in multiagent_batch_buffer:
                agent_id, agent_batch = data['agent_id'], data['batch']
                agent_batch.non_tensor_batch['agent_id'] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch['uid'] = uid_batch
                agent_batch.non_tensor_batch['traj_uid'] = traj_uid
                agent_batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                agent_batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
                agent_batch_list: list[dict] = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]['agent_active_mask']:
                        if capture_event_trace:
                            agent_batch_list[i]['task_type'] = task_type
                        if capture_event_trace and task_type == 'search':
                            event = agent_batch_list[i]
                            info = infos[i]
                            action_owner = env_action_owners[i]
                            event.update(
                                build_search_transition_fields(
                                    event,
                                    info,
                                    action_owner=action_owner,
                                    text_action=text_actions[i],
                                    reward=rewards[i],
                                    done=dones[i],
                                )
                            )
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        if capture_event_trace:
            if trace_env_kwargs is None or len(trace_env_kwargs) != batch_size:
                raise ValueError("Event tracing requires one env_kwargs item per rollout trajectory")
            include_offline_eval_info = bool(event_trace_option(self.config, 'include_offline_eval_info', False))
            task_uids = [build_task_uid(task_kwargs) for task_kwargs in trace_env_kwargs]
            offline_eval_infos = (
                [build_offline_eval_info(task_kwargs) for task_kwargs in trace_env_kwargs]
                if include_offline_eval_info
                else None
            )
            for item, trajectory_events in enumerate(total_batch_list):
                terminal_action = terminal_actions[item]
                if task_type == 'search':
                    final_answer = (
                        terminal_action
                        if isinstance(terminal_action, str) and '<answer>' in terminal_action.lower()
                        else None
                    )
                    forced_answer_events = [
                        event
                        for event in trajectory_events
                        if event.get('event_type') == 'final_answer' and bool(event.get('route_forced', False))
                    ]
                    answer_events = [
                        event for event in trajectory_events if event.get('event_type') == 'final_answer'
                    ]
                    if final_answer is not None:
                        stop_reason = (
                            'forced_answer_submitted'
                            if forced_answer_events
                            else 'answer_submitted'
                        )
                    elif forced_answer_events:
                        # SearchEnv ends the max-step transition even if the
                        # forced Answer projection is malformed.  Do not mix
                        # this routing/format failure with ordinary exhaustion.
                        stop_reason = 'forced_answer_invalid'
                    elif answer_events:
                        # Preserve a non-forced Answer formatting failure as a
                        # distinct terminal outcome instead of attributing it
                        # to search-budget exhaustion.
                        stop_reason = 'answer_invalid'
                    elif episode_lengths[item] >= self.config.env.max_steps:
                        stop_reason = 'max_steps_exhausted'
                    else:
                        stop_reason = 'env_done'
                else:
                    final_answer = terminal_action
                    stop_reason = None
                for event in trajectory_events:
                    event['task_type'] = task_type
                    event['task_uid'] = task_uids[item]
                    # Preserve the complete task question for every exported
                    # trajectory, including validation with semantic value off.
                    event['original_question'] = trace_env_kwargs[item]['question']
                    if task_type == 'math':
                        event['max_solver_turns'] = int(self.config.agent.orchestra.math.max_loop_num)
                    event['sampling_try'] = 0
                    event['terminal_action_text'] = terminal_action
                    event['final_answer_text'] = final_answer
                    if task_type == 'search':
                        event['orchestration_stop_reason'] = stop_reason
                    if include_offline_eval_info:
                        event['offline_eval_info'] = deepcopy(offline_eval_infos[item])
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
