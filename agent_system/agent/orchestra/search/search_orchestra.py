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
"""Agent execution structures – chain & hierarchy.
"""
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
# from agent_system.agent.agents import *
import importlib
from verl import DataProto
import numpy as np

from agent_system.event_trace import (
    event_metadata_enabled,
    parse_search_verifier_decisions,
)
from agent_system.search_protocol import (
    plan_search_routes,
    search_protocol_metadata,
    validate_independent_search_role_topology,
)


def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the observation dictionary with the text response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    # Naive append of the latest responses to observations
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the text actions with the latest response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions


def _config_option(container: Any, key: str, default: Any) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _config_bool(container: Any, key: str, default: bool) -> bool:
    """Read a boolean safely from Hydra, dictionaries, or shell-style strings."""

    value = _config_option(container, key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"{key} must be boolean-like, got {value!r}")
    return bool(value)


def _object_array(values: List[Any]) -> np.ndarray:
    """Preserve nested trace values as one object per batch row."""

    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def build_route_handoff(
    team_context: List[str],
    active_mask: np.ndarray,
    route_targets: np.ndarray,
    route_reasons: np.ndarray,
) -> List[str]:
    """Expose only the parsed route handoff, never the Verifier's raw CoT."""

    for item, is_active in enumerate(active_mask):
        if not is_active:
            continue
        decision = "answer_now" if route_targets[item] == "Answer Agent" else "need_more_information"
        team_context[item] = (
            f'<ROUTE_HANDOFF source="Verifier" decision="{decision}" '
            f'reason="{route_reasons[item]}"/>'
        )
    return team_context


def attach_protocol_trace_metadata(
    batch: DataProto,
    protocol_states: Any,
    *,
    protocol_context_hashes: Any,
    protocol_ledger_visible: Any,
    route_reasons: np.ndarray,
    forced_final_answer_mask: np.ndarray,
    step: int,
    max_steps: int,
) -> None:
    """Attach compact, pre-action protocol facts to every Search trace event."""

    batch_size = len(batch)
    if protocol_states is None or len(protocol_states) != batch_size:
        protocol_states = [
            {
                "turn_count": max(0, int(step) - 1),
                "max_steps": int(max_steps),
                "queries": [],
                "evidence": [],
            }
            for _ in range(batch_size)
        ]
    metadata = [search_protocol_metadata(state) for state in protocol_states]
    if protocol_context_hashes is None or len(protocol_context_hashes) != batch_size:
        protocol_context_hashes = [hashlib.sha256(b"").hexdigest()] * batch_size
    if protocol_ledger_visible is None or len(protocol_ledger_visible) != batch_size:
        protocol_ledger_visible = [True] * batch_size
    protocol_ledger_visible = np.asarray(protocol_ledger_visible, dtype=bool)
    batch.non_tensor_batch["protocol_version"] = np.array(
        ["drmas.search_protocol.v2"] * batch_size,
        dtype=object,
    )
    for field in (
        "protocol_turn_index",
        "remaining_search_slots",
        "query_history_count",
        "retrieval_attempt_count",
        "successful_retrieval_count",
        "evidence_ledger_count",
        "duplicate_query_count",
        "last_tool_status",
    ):
        batch.non_tensor_batch[field] = np.array([item[field] for item in metadata], dtype=object)
    for field in (
        "visible_query_ids",
        "visible_query_hashes",
        "visible_evidence_ids",
        "visible_evidence_sha256s",
    ):
        batch.non_tensor_batch[field] = _object_array(
            [item[field] if protocol_ledger_visible[index] else [] for index, item in enumerate(metadata)]
        )
    batch.non_tensor_batch["ledger_snapshot_sha256"] = np.array(protocol_context_hashes, dtype=object)
    batch.non_tensor_batch["protocol_ledger_visible"] = protocol_ledger_visible.copy()
    batch.non_tensor_batch["route_reason"] = route_reasons.copy()
    batch.non_tensor_batch["route_forced"] = forced_final_answer_mask.astype(bool, copy=True)


class SearchMultiAgentOrchestra(BaseOrchestra):
    """
    The architecture consists of:
    1. Search Agent: Generates search queries to gather information
    2. Verifier Agent: Determines if information is sufficient to answer
    3. Answer Agent: Generates final answer when information is sufficient
    

   Verifier Agent (Router) → Evaluates if historical information is sufficient
        ├─ If "no" → 2a. Search Agent → Generates search query → Return search query
        └─ If "yes" → 2b. Answer Agent → Generates answer → Return answer

    Args:
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the orchestra.
    """
    # Agent type constants
    VERIFIER_AGENT = "Verifier Agent"
    SEARCH_AGENT = "Search Agent"
    ANSWER_AGENT = "Answer Agent"
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        """Initialize the search multi-agent orchestra.
        
        Args:
            agent_ids: List of agent names to be executed in sequence
            model_ids: List of model identifiers
            agents_to_wg_mapping: Mapping from agent names to worker group IDs
            tokenizers: Dictionary of tokenizers for each worker group
            processors: Dictionary of processors for each worker group
            config: Configuration object containing settings for the orchestra
        """
        # Import search agents module
        importlib.import_module("agent_system.agent.agents.search")

        # Initialize base class
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("Orchestra requires at least one agent.")

        # Validate that required agents are present
        if self.SEARCH_AGENT not in self.agent_ids:
            raise ValueError("Search Agent is required. Please add it to the agent_ids.")
        if self.VERIFIER_AGENT not in self.agent_ids:
            raise ValueError("Verifier Agent is required. Please add it to the agent_ids.")
        if self.ANSWER_AGENT not in self.agent_ids:
            raise ValueError("Answer Agent is required. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        protocol_config = _config_option(_config_option(self.config, "env", None), "search_protocol", None)
        # Preserve the old malformed-Verifier behavior by default, while
        # making it explicit and configurable for a dedicated ablation.
        invalid_route = str(_config_option(protocol_config, "invalid_verifier_route", "answer")).lower()
        if invalid_route not in {"search", "answer"}:
            raise ValueError(
                "env.search_protocol.invalid_verifier_route must be 'search' or 'answer', "
                f"got {invalid_route!r}"
            )
        self.invalid_verifier_route = invalid_route
        self.require_independent_models = _config_bool(
            protocol_config,
            "require_independent_models",
            True,
        )
        validate_independent_search_role_topology(
            self.agents_to_wg_mapping,
            model_sharing=_config_bool(_config_option(self.config, "agent", None), "model_sharing", False),
            require_independent_models=self.require_independent_models,
        )
        self.last_route_plan = None
        self.last_search_action_mask: Optional[np.ndarray] = None

    def reset(self):
        """Clear state that is meaningful only for the previous trajectory."""

        super().reset()
        self.last_route_plan = None
        self.last_search_action_mask = None

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the orchestra with the three-agent architecture using Verifier as router.
        
        Execution flow:
        1. Verifier Agent determines if current information is sufficient (routing decision)
        2. If sufficient (yes): Answer Agent generates final answer
        3. If insufficient (no): Search Agent generates search query while budget remains
        4. At the final environment step, Answer is forced for every active row
           so a route decision never produces an empty terminal action.
        """
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        agent_active_mask = np.logical_and(np.ones(len(gen_batch), dtype=bool), active_masks).astype(bool)
            
        # Step 1: Run Verifier Agent (Router)
        actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.VERIFIER_AGENT]]
        
        batch, text_repsonses = self.agents[self.VERIFIER_AGENT].call(
            gen_batch=gen_batch,
            env_obs=env_obs,
            team_context=team_context,
            actor_rollout_wg=actor_rollout_wg,
            agent_active_mask=agent_active_mask,
            step=step,
        )
        
        # Keep the raw model decision separate from the effective route.  An
        # invalid verifier response uses an explicit fallback; a final-step
        # no/invalid is overridden to Answer to avoid an empty terminal action.
        if "is_action_valid" not in batch.non_tensor_batch:
            raise RuntimeError("Verifier Agent batch is missing is_action_valid required for Search routing")
        verifier_valid_mask = np.asarray(batch.non_tensor_batch["is_action_valid"], dtype=bool)
        verifier_decisions = parse_search_verifier_decisions(
            text_repsonses,
            agent_active_mask,
            action_valid_mask=verifier_valid_mask,
        )
        max_steps = int(self.config.env.max_steps)
        route_plan = plan_search_routes(
            verifier_decisions,
            agent_active_mask,
            step=step,
            max_steps=max_steps,
            invalid_verifier_route=self.invalid_verifier_route,
        )
        route_targets = route_plan.targets
        route_reasons = route_plan.reasons
        forced_final_answer_mask = route_plan.forced_mask
        search_active_mask = route_plan.search_mask
        answer_active_mask = route_plan.answer_mask
        # This is read by the environment manager even when event tracing is
        # off; it prevents text heuristics from misclassifying Answer output as
        # a retrieval action in the factual ledger.
        self.last_route_plan = route_plan
        self.last_search_action_mask = search_active_mask.copy()
        team_context = build_route_handoff(
            team_context,
            agent_active_mask,
            route_targets,
            route_reasons,
        )

        if capture_event_trace := event_metadata_enabled(self.config):
            batch.non_tensor_batch['event_type'] = np.array(['verifier'] * len(batch), dtype=object)
            batch.non_tensor_batch['verifier_decision'] = verifier_decisions.copy()
            batch.non_tensor_batch['route_target'] = route_targets.copy()
            batch.non_tensor_batch['is_env_action'] = np.zeros(len(batch), dtype=bool)
            batch.non_tensor_batch['orchestration_stop_reason'] = np.array(['pending'] * len(batch), dtype=object)
            attach_protocol_trace_metadata(
                batch,
                env_obs.get("search_protocol_state"),
                protocol_context_hashes=env_obs.get("search_protocol_context_sha256"),
                protocol_ledger_visible=env_obs.get("search_protocol_ledger_visible"),
                route_reasons=route_reasons,
                forced_final_answer_mask=forced_final_answer_mask,
                step=step,
                max_steps=max_steps,
            )
        self.save_to_buffer(self.VERIFIER_AGENT, batch)
            
        # Step 2: Conditionally run Search Agent (when verification says no)
        if search_active_mask.any():
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.SEARCH_AGENT]]
            
            batch, text_repsonses = self.agents[self.SEARCH_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=search_active_mask,
                step=step,
            )
            
            if capture_event_trace:
                batch.non_tensor_batch['event_type'] = np.array(['search_query'] * len(batch), dtype=object)
                batch.non_tensor_batch['verifier_decision'] = verifier_decisions.copy()
                batch.non_tensor_batch['route_target'] = route_targets.copy()
                batch.non_tensor_batch['is_env_action'] = search_active_mask.copy()
                batch.non_tensor_batch['orchestration_stop_reason'] = np.array(['pending'] * len(batch), dtype=object)
                attach_protocol_trace_metadata(
                    batch,
                    env_obs.get("search_protocol_state"),
                    protocol_context_hashes=env_obs.get("search_protocol_context_sha256"),
                    protocol_ledger_visible=env_obs.get("search_protocol_ledger_visible"),
                    route_reasons=route_reasons,
                    forced_final_answer_mask=forced_final_answer_mask,
                    step=step,
                    max_steps=max_steps,
                )
            self.save_to_buffer(self.SEARCH_AGENT, batch)
            
            # Update text_actions with Search Agent output for samples needing more info
            text_actions = update_text_action(text_actions, text_repsonses, search_active_mask)
        
        # Step 3: Conditionally run Answer Agent (when verification says yes)
        if answer_active_mask.any():
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ANSWER_AGENT]]
            
            batch, text_repsonses = self.agents[self.ANSWER_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=answer_active_mask,
                step=step,
            )
            
            if capture_event_trace:
                batch.non_tensor_batch['event_type'] = np.array(['final_answer'] * len(batch), dtype=object)
                batch.non_tensor_batch['verifier_decision'] = verifier_decisions.copy()
                batch.non_tensor_batch['route_target'] = route_targets.copy()
                batch.non_tensor_batch['is_env_action'] = answer_active_mask.copy()
                batch.non_tensor_batch['orchestration_stop_reason'] = np.array(['pending'] * len(batch), dtype=object)
                attach_protocol_trace_metadata(
                    batch,
                    env_obs.get("search_protocol_state"),
                    protocol_context_hashes=env_obs.get("search_protocol_context_sha256"),
                    protocol_ledger_visible=env_obs.get("search_protocol_ledger_visible"),
                    route_reasons=route_reasons,
                    forced_final_answer_mask=forced_final_answer_mask,
                    step=step,
                    max_steps=max_steps,
                )
            self.save_to_buffer(self.ANSWER_AGENT, batch)
            
            # Update text_actions with Answer Agent output for verified samples
            text_actions = update_text_action(text_actions, text_repsonses, answer_active_mask)

        return text_actions, self.multiagent_batch_buffer
