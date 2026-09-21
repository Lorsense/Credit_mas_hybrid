# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
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

from __future__ import annotations
"""
Agent definitions.
"""
from typing import Dict, Any, List, Tuple
import copy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.protocol import extract_dataproto_via_active_mask, restore_dataproto_via_active_mask
from transformers import PreTrainedTokenizer
import numpy as np

from agent_system.event_trace import event_metadata_enabled

class BaseAgent:
    """Abstract agent.  All subclasses *must* implement :py:meth:`act`."""

    def __init__(self, name: str, prompt: str, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        self.name = name
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        self.wg_id = wg_id

        self.start_tag = None
        self.end_tag = None

        # Check if prompt is defined for this agent via calling the property
        if not hasattr(self, 'prompt') or not isinstance(self.prompt, str):
            raise ValueError(f"Agent '{self.name}' must define a 'prompt' property.")

    def reset(self):
        pass

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int) -> str:
        """Build the prompt for the agent based on the observation."""
        # Naive Implementation
        obs = copy.deepcopy(env_obs)
        bs = len(obs['text'])
        protocol_contexts = obs.get("search_protocol_context")
        if protocol_contexts is None:
            protocol_contexts = [""] * bs
        if len(protocol_contexts) != bs:
            raise ValueError(
                "search_protocol_context must have one entry per observation: "
                f"got {len(protocol_contexts)} for batch size {bs}"
            )
        for i in range(bs):
            if self.start_tag is not None and self.end_tag is not None:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i],
                                                    step=step,
                                                    protocol_context=protocol_contexts[i],
                                                    start_tag=self.start_tag,
                                                    end_tag=self.end_tag)
            else:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i],
                                                    step=step,
                                                    protocol_context=protocol_contexts[i])
        return obs


    def _generate_with_llm(self, batch: DataProto, actor_rollout_wg, agent_active_mask: np.ndarray, meta_info) -> Tuple[DataProto, List[str]]:
        """Helper: prompt → input_ids → actor_rollout_wg → decoded str."""
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

        if event_metadata_enabled(self.config):
            batch_input.meta_info = dict(meta_info)
            batch_input.meta_info['capture_event_trace'] = True
        else:
            batch_input.meta_info = meta_info
        
        # extract activate data
        batch_input_extracted = extract_dataproto_via_active_mask(batch_input, agent_active_mask)
        # pad to be divisible by dp_size
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input_extracted, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        # unpad
        batch_output_extracted = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        # restorr
        batch_output = restore_dataproto_via_active_mask(batch_output_extracted, agent_active_mask)

        batch = batch.union(batch_output)
        
        text_repsonses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

        if event_metadata_enabled(self.config):
            # Projection may normalize or replace the returned strings.  Keep
            # the model's decoded response before that happens; response token
            # IDs in batch['responses'] remain the authoritative actor target.
            batch.non_tensor_batch['raw_action_text'] = np.array(text_repsonses, dtype=object)
            response_length = batch.batch['responses'].shape[-1]
            response_masks = batch.batch['attention_mask'][:, -response_length:].bool()
            eos_token_id = self.tokenizer.eos_token_id
            ended_with_eos = []
            for response_ids, response_mask in zip(batch.batch['responses'], response_masks):
                valid_response_ids = response_ids[response_mask]
                ended_with_eos.append(
                    bool(valid_response_ids.numel() > 0 and eos_token_id is not None and valid_response_ids[-1].item() == eos_token_id)
                )
            batch.non_tensor_batch['ended_with_eos'] = np.array(ended_with_eos, dtype=bool)
            finish_reasons = batch.non_tensor_batch.get(
                'generation_finish_reason', np.full(len(batch), 'unknown', dtype=object)
            )
            backend_truncated = np.asarray([
                str(reason).lower() in {'length', 'max_tokens', 'max_new_tokens'} for reason in finish_reasons
            ], dtype=bool)
            full_response = response_masks.sum(-1).detach().cpu().numpy() >= response_length
            # A semantic prefix's validity must not depend on entropy-feature
            # availability. Retain truncation independently for the value path.
            batch.non_tensor_batch['value_action_truncated'] = backend_truncated | full_response
            agent_ids = list(self.config.agent.agent_ids)
            model_ids = list(self.config.agent.model_ids)
            model_id = model_ids[agent_ids.index(self.name)]
            batch.non_tensor_batch['model_id'] = np.array([model_id] * len(batch), dtype=object)

        # insert model name
        batch.non_tensor_batch['wg_id'] = np.array([self.wg_id] * len(batch), dtype=object)
        batch.non_tensor_batch['agent_active_mask'] = agent_active_mask

        return batch, text_repsonses

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a response based on the observation and the batch.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            team_context (List[str]): Contextual information from the team.
            actor_rollout_wg: The LLM policy for acting.
            step: environment step
        Returns:
            Tuple[DataProto, List[str], List[str]]:
                - batch (DataProto): The processed batch after generation.
                - text_repsonses (List[str]): The generated text responses.
        """
        raise NotImplementedError

__all__ = [
    "BaseAgent",
]
