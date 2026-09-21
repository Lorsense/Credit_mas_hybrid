from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
from copy import deepcopy
import hashlib
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from agent_system.search_protocol import (
    new_search_protocol_states,
    render_search_protocol_context,
    update_search_protocol_state,
)

import time


def _config_option(container: Any, key: str, default: Any) -> Any:
    """Read a dict/OmegaConf/object option without requiring a concrete type."""

    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _config_bool(container: Any, key: str, default: bool) -> bool:
    """Parse Hydra/dict booleans consistently with the Search orchestra."""

    value = _config_option(container, key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{key} must be a boolean-like value, got {value!r}")
    return bool(value)

class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)
        protocol_config = _config_option(_config_option(config, "env", None), "search_protocol", None)
        self.search_protocol_enabled = _config_bool(protocol_config, "enabled", True)
        self.search_protocol_history = max(0, int(_config_option(protocol_config, "max_history", 4)))
        self.search_protocol_query_chars = max(0, int(_config_option(protocol_config, "query_char_limit", 512)))
        self.search_protocol_evidence_chars = max(0, int(_config_option(protocol_config, "evidence_char_limit", 2400)))
        self.search_protocol_states: list[dict[str, Any]] = []

    def _protocol_contexts(self) -> list[str]:
        if not self.search_protocol_enabled:
            return ["" for _ in self.search_protocol_states]
        return [
            render_search_protocol_context(
                state,
                max_history=self.search_protocol_history,
                query_char_limit=self.search_protocol_query_chars,
                evidence_char_limit=self.search_protocol_evidence_chars,
            )
            for state in self.search_protocol_states
        ]

    def _protocol_observation_fields(self) -> dict[str, Any]:
        """Expose only factual, pre-action protocol state to the orchestra."""

        contexts = self._protocol_contexts()
        return {
            "search_protocol_context": contexts,
            "search_protocol_state": deepcopy(self.search_protocol_states),
            "search_protocol_context_sha256": [
                hashlib.sha256(context.encode("utf-8")).hexdigest() for context in contexts
            ],
            # In the raw-history ablation the state still advances for
            # diagnostics, but the structured ledger is deliberately absent
            # from the model prompt.  Preserve that distinction in traces.
            "search_protocol_ledger_visible": [
                self.search_protocol_enabled for _ in self.search_protocol_states
            ],
        }

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))
        self.search_protocol_states = new_search_protocol_states(
            batch_size=len(obs),
            max_steps=self.config.env.max_steps,
        )

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
            **self._protocol_observation_fields(),
        }
        
        return observations, infos

    def step(
        self,
        text_actions: List[str],
        *,
        search_action_mask: np.ndarray | List[bool] | None = None,
        active_mask: np.ndarray | List[bool] | None = None,
    ):
        """Execute one environment action and advance the factual protocol.

        ``search_action_mask`` is supplied by the orchestra's explicit route
        plan.  It is deliberately not inferred from text: an Answer response
        may contain arbitrary text, while an invalid Search response still
        needs to be represented as a failed retrieval attempt.  ``active_mask``
        prevents already-finished vectorized rows from advancing their ledger.
        """
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        if search_action_mask is None:
            search_action_mask = np.zeros(len(actions), dtype=bool)
        else:
            search_action_mask = np.asarray(search_action_mask, dtype=bool)
            if search_action_mask.shape != (len(actions),):
                raise ValueError(
                    "search_action_mask must contain one value per environment action: "
                    f"got {search_action_mask.shape} for {len(actions)} actions"
                )
        if active_mask is None:
            active_mask = np.ones(len(actions), dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)
            if active_mask.shape != (len(actions),):
                raise ValueError(
                    "active_mask must contain one value per environment action: "
                    f"got {active_mask.shape} for {len(actions)} actions"
                )

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"SearchEnv step time: {time2 - time1:.4f} seconds")

        # Behavior-neutral transition metadata for offline event tracing.  Keep
        # the raw tool observation separate from the rendered next prompt.
        for observation, info in zip(next_obs, infos):
            info["tool_observation_text"] = observation if info.get("tool_calling", False) else None
            info.setdefault("tool_name", None)
            info.setdefault("tool_query", info.get("tool_input"))
            info.setdefault("tool_query_valid", None)
            info.setdefault("tool_status", None)
            info.setdefault("tool_result_count", None)
            info.setdefault("tool_error", None)

        for state, action, observation, info, is_search_action, is_active in zip(
            self.search_protocol_states,
            actions,
            next_obs,
            infos,
            search_action_mask,
            active_mask,
        ):
            if not is_active:
                continue
            update_search_protocol_state(
                state,
                action=action,
                info=info,
                observation=observation,
                is_search_action=bool(is_search_action),
                history_limit=self.search_protocol_history,
                query_storage_char_limit=self.search_protocol_query_chars,
                evidence_storage_char_limit=self.search_protocol_evidence_chars,
            )

        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
            **self._protocol_observation_fields(),
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        use_protocol_history = self.search_protocol_enabled and self.config.agent.multi_agent
        if not init and self.config.env.history_length > 0 and not use_protocol_history:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0 or use_protocol_history:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
                else:
                    obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
            else:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context="{memory}" if self.config.agent.use_agent_memory else memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
                else:
                    obs_i = SEARCH_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context=memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            


class MathEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for MathEnv.
    """
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        observations = {
            "text": self.build_text_obs(obs),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"MathEnv step time: {time2 - time1:.4f} seconds")

        next_observations = {
            "text": None,
            "image": None,
            "anchor": None
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        for i in range(len(text_obs)):
            if self.config.agent.multi_agent:
                obs_i = MATH_MULTIAGENT_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = MATH_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # Get validation rollout n for pass@k and avg@k computation
    val_group_n = getattr(config.env.rollout, 'val_n', 1)

    if "math" in config.env.env_name.lower():
        from agent_system.environments.env_package.math import build_math_envs, math_projection
        _envs = build_math_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_math_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False)
        
        projection_f = partial(math_projection)
        envs = MathEnvironmentManager(_envs, projection_f, config)
        val_envs = MathEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
