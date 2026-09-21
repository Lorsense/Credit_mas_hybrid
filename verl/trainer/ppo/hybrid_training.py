"""Driver-side wiring for zxj advantages, mix credit and semantic entropy control."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.hybrid_credit import (
    prepare_hybrid_action_metadata, prepare_entropy_credit,
    prepare_sparse_entropy_credit_for_batch, finalize_sparse_entropy_credit_for_advantages,
    apply_entropy_credit_to_advantages,
)
from verl.utils.credit_resources import validate_pool_layout
from verl.utils.semantic_credit import build_trajectory_records, attach_value_predictions
from verl.utils.semantic_entropy_control import SemanticEntropyController
from verl.utils.hybrid_observability import compute_hybrid_observability


def _plain(config):
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)


def _metrics(prefix, values):
    return {f"semantic_value/{prefix}/{key}": float(value) for key, value in values.items()
            if isinstance(value, (int, float, bool, np.number))}


class HybridTrainingMixin:
    def _configure_hybrid(self):
        algorithm = self.config.algorithm
        self.entropy_credit_config = algorithm.get("entropy_credit", {})
        self.semantic_value_config = algorithm.get("semantic_value", {})
        self.semantic_control_config = algorithm.get("semantic_entropy_control", {})
        self.entropy_credit_enabled = bool(self.entropy_credit_config.get("enable", False))
        self.semantic_value_enabled = bool(self.semantic_value_config.get("enable", False))
        self.entropy_control_enabled = bool(self.semantic_control_config.get("enabled", False))
        self.hybrid_enabled = self.entropy_credit_enabled or self.semantic_value_enabled or self.entropy_control_enabled
        self.semantic_scorer = None
        self.semantic_ready, self.semantic_reliability = False, {}
        self.entropy_controller = None
        if algorithm.get("advantage_recovery", {}).get("enable", False):
            raise ValueError("Hybrid does not support zero-advantage recovery")
        if self.entropy_credit_config.get("value", {}).get("enable", False):
            raise ValueError("Use algorithm.semantic_value; the old entropy value branches were removed")
        if self.entropy_credit_config.get("sparse", {}).get("enable", False) and not self.entropy_credit_enabled:
            raise ValueError("Sparse credit requires entropy_credit.enable")
        if self.entropy_control_enabled and not self.semantic_value_enabled:
            raise ValueError("Semantic entropy control requires semantic_value.enable")
        for configs in self.wg_to_agents_mapping.values():
            for item in configs:
                cfg = item["config_actor_rollout_ref"]
                actor = cfg.actor
                enabled = bool(actor.get("entropy_control", {}).get("enabled", False))
                if enabled != self.entropy_control_enabled:
                    raise ValueError("Driver and every Actor must agree on entropy_control.enabled")
                if enabled and (actor.strategy not in ("fsdp", "fsdp2") or float(actor.entropy_coeff) != 0):
                    raise ValueError("Semantic entropy control requires FSDP and actor.entropy_coeff=0")
                if self.entropy_credit_enabled and (cfg.rollout.name != "sglang" or
                        cfg.rollout.get("mode", "sync") != "sync" or int(cfg.rollout.get("top_logprobs_num", 0)) != 16):
                    raise ValueError("Hybrid entropy credit requires synchronous SGLang Top16 log probabilities")
        if not self.hybrid_enabled:
            return
        if algorithm.adv_estimator != "team_event_gae" or self.config.agent.orchestra_type != "math":
            raise ValueError("Hybrid requires the zxj Math team_event_gae estimator")
        if algorithm.team_event_gae.math_value.mode != "math_answer_loto":
            raise ValueError("Hybrid requires zxj math_answer_loto values")
        if self.config.trainer.default_hdfs_dir is not None:
            raise ValueError("Hybrid checkpoints require a local/shared filesystem")
        if self.semantic_value_enabled:
            value = self.semantic_value_config
            if not value.get("model_path"):
                raise ValueError("semantic_value.model_path must identify the independent frozen encoder")
            if value.get("device", "cpu") not in ("cpu", "cuda:0") or float(value.get("num_cpus", 1)) <= 0:
                raise ValueError("Semantic scorer requires cpu or one dedicated cuda:0 GPU and positive CPUs")
            if value.get("initialization_mode", "qualified") not in ("qualified", "candidate"):
                raise ValueError("semantic initialization_mode must be qualified or candidate")
            if value.get("allow_missing_resume", False):
                raise ValueError("Hybrid resume requires complete semantic/controller state")
        if self.entropy_control_enabled:
            self.entropy_controller = SemanticEntropyController(_plain(self.semantic_control_config))

    def _init_semantic_scorer(self):
        if not self.semantic_value_enabled or self.config.trainer.get("val_only", False):
            return
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from verl.workers.semantic_value import SemanticValueWorker
        value = _plain(self.semantic_value_config)
        gpu = value.get("device", "cpu") == "cuda:0"
        node = ray.get_runtime_context().get_node_id()
        available = ray.state.available_resources_per_node()
        cpus = float(value.get("num_cpus", 1))
        validate_pool_layout(available, self.resource_pool_manager.resource_pool_spec,
                             reserved_node=node if gpu else None, cpus_per_gpu=1,
                             reserved_cpus=cpus if gpu else 0)
        if available.get(node, {}).get("CPU", 0) < cpus:
            raise ValueError("Trainer node has insufficient CPUs for the semantic scorer")
        self.semantic_scorer = ray.remote(SemanticValueWorker).options(
            num_cpus=cpus, num_gpus=int(gpu),
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
        ).remote(value)
        # Reserve and initialize the separate scorer BEFORE allocating Actor pools.
        ray.get(self.semantic_scorer.ready.remote(), timeout=float(value.get("startup_timeout", 1800)))

    def _initialize_fresh_hybrid(self):
        if self.semantic_scorer is None:
            return
        import ray
        value = self.semantic_value_config
        initial = value.get("initial_checkpoint")
        if not initial:
            if value.get("require_pretrained", True) or value.get("initialization_mode", "qualified") == "candidate":
                raise ValueError("Fresh hybrid training requires a semantic initial_checkpoint; pretrain it or explicitly allow cold training")
            return
        initial = os.path.abspath(os.path.expanduser(initial))
        if not os.path.isfile(initial):
            raise FileNotFoundError(initial)
        ray.get(self.semantic_scorer.load.remote(
            initial, resume=False, warm_start=value.get("initialization_mode", "qualified") == "candidate"))

    def _prepare_hybrid_rollout(self, batch):
        metrics = {}
        if not self.hybrid_enabled:
            return metrics
        prepare_hybrid_action_metadata(batch)
        if self.entropy_credit_enabled:
            _, result = prepare_entropy_credit(batch, self.entropy_credit_config)
            metrics.update(result)
            if self.entropy_credit_config.get("sparse", {}).get("enable", False):
                _, result = prepare_sparse_entropy_credit_for_batch(batch, self.entropy_credit_config.sparse)
                metrics.update(result)
        if self.semantic_scorer is not None:
            import ray
            required = {"value_question", "value_action_text", "value_action_index", "value_max_solver_turns",
                        "uid", "traj_uid", "agent_id", "role_turn_index", "is_action_valid", "pass"}
            if required.difference(batch.non_tensor_batch):
                raise ValueError(f"Missing semantic rollout metadata: {sorted(required.difference(batch.non_tensor_batch))}")
            records, result = build_trajectory_records(batch.non_tensor_batch, int(self.config.agent.orchestra.math.max_loop_num))
            metrics.update(result)
            scored = ray.get(self.semantic_scorer.prepare.remote(records))
            self.semantic_ready = bool(scored["ready"])
            self.semantic_reliability = scored["reliability"]
            metrics.update(_metrics("score", scored["metrics"]))
            batch.non_tensor_batch["value_credit_scorer_version"] = np.full(len(batch), int(scored["version"]), dtype=np.int64)
            metrics.update(attach_value_predictions(batch.non_tensor_batch, scored["values"], scored["ready"]))
        return metrics

    def _apply_hybrid_after_advantage(self, batch):
        metrics = {}
        if self.entropy_credit_enabled:
            if self.entropy_credit_config.get("sparse", {}).get("enable", False):
                _, result = finalize_sparse_entropy_credit_for_advantages(batch, self.entropy_credit_config.sparse)
                metrics.update(result)
            # Preserve explicit audit tensors. Only Actor advantages change.
            batch.batch["hybrid_base_advantages"] = batch.batch["advantages"].clone()
            apply_entropy_credit_to_advantages(batch)
        if self.entropy_controller is not None:
            arrays, result = self.entropy_controller.prepare(
                batch.non_tensor_batch, batch.non_tensor_batch["action_full_entropy"],
                self.global_steps, self.semantic_ready, self.semantic_reliability)
            metrics.update(result)
            for key, array in arrays.items():
                dtype = torch.bool if key == "entropy_control_valid" else torch.float32
                batch.batch[key] = torch.as_tensor(array, dtype=dtype, device=batch.batch["responses"].device).detach()
            for tensor_key, log_key in (("entropy_control_weight", "entropy_control_brake"),
                                        ("entropy_control_cap", "entropy_control_cap_value"),
                                        ("entropy_control_valid", "entropy_control_valid_action")):
                batch.non_tensor_batch[log_key] = arrays[tensor_key]
        if self.hybrid_enabled:
            metrics.update(compute_hybrid_observability(batch))
        return metrics

    def _update_semantic_after_actors(self):
        if self.semantic_scorer is None:
            return {}
        import ray
        return _metrics("train", ray.get(self.semantic_scorer.update.remote()))

    def _hybrid_manifest(self):
        return {"schema": "zxj-mix-semantic-v1", "entropy_credit": _plain(self.entropy_credit_config),
                "semantic_value_enabled": self.semantic_value_enabled,
                "semantic_entropy_control": _plain(self.semantic_control_config),
                "actor_entropy_control": {str(wg): [_plain(item["config_actor_rollout_ref"].actor.get("entropy_control", {}))
                                                      for item in items] for wg, items in self.wg_to_agents_mapping.items()}}

    def _save_hybrid_checkpoint(self, folder):
        if not self.hybrid_enabled:
            return
        folder = Path(folder)
        if self.semantic_scorer is not None:
            import ray
            ray.get(self.semantic_scorer.save.remote(str(folder / "semantic_value.pt")))
        if self.entropy_controller is not None:
            (folder / "semantic_entropy_controller.json").write_text(json.dumps(self.entropy_controller.state_dict()), encoding="utf-8")
        state = {"manifest": self._hybrid_manifest(), "actor_update_count": self._actor_update_count}
        (folder / "hybrid_state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _load_hybrid_checkpoint(self, folder):
        if self.config.trainer.get("val_only", False):
            return
        folder = Path(folder)
        if not self.hybrid_enabled:
            if (folder / "hybrid_state.json").is_file():
                raise ValueError("Cannot resume a hybrid checkpoint with all hybrid modules disabled; use a new run for ablations")
            return
        state = json.loads((folder / "hybrid_state.json").read_text(encoding="utf-8"))
        if state["manifest"] != self._hybrid_manifest():
            raise ValueError("Hybrid checkpoint method/config differs; use a new run for ablations")
        if not (folder / "data.pt").is_file():
            raise ValueError("Hybrid checkpoint is missing dataloader state")
        if self.entropy_controller is not None:
            self.entropy_controller.load_state_dict(json.loads((folder / "semantic_entropy_controller.json").read_text(encoding="utf-8")))
        if self.semantic_scorer is not None:
            import ray
            ray.get(self.semantic_scorer.load.remote(str(folder / "semantic_value.pt"), resume=True))
        self._actor_update_count = int(state["actor_update_count"])
