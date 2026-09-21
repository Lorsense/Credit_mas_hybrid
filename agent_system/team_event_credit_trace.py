"""Boundary-LOTO / Math answer-LOTO startup contracts and post-advantage,
unique-event audit IO.

This module deliberately has no Ray/DataProto import: configuration and trace
contracts can be tested on CPU, independently of the distributed launcher.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path


def _get(config, path, default=None):
    value = config
    for key in path.split("."):
        if not hasattr(value, "get"):
            return default
        value = value.get(key, default)
    return value


def validate_boundary_training_config(config):
    """Validate effective per-agent configuration, before any actor update.

    Environment rollouts implement grouped sampling in this repository;
    actor_rollout_ref.rollout.n must stay one (see main_ppo's assertion).
    val_n is intentionally not part of the training group-size contract.
    """
    from verl.trainer.ppo.team_event_boundary_value import BoundaryValueConfig, load_frozen_kernel

    value = BoundaryValueConfig.from_mapping(_get(config, "algorithm.team_event_gae.value", {}))
    if value.mode == "legacy_loto":
        return None
    value.validate_runtime(
        gamma=float(_get(config, "algorithm.gamma", 1.0)),
        internal_gamma=float(_get(config, "algorithm.team_event_gae.internal_gamma", 1.0)),
        agent_local_mode=str(_get(config, "algorithm.team_event_gae.agent_local.mode", "off")),
    )
    if _get(config, "agent.orchestra_type") != "search" or _get(config, "env.env_name") != "search":
        raise ValueError("boundary_loto requires the Search orchestra and Search environment")
    # val_only never computes the training estimator; inference may use n=1
    # without possessing the estimator's old training-time IDF artifact.
    if bool(_get(config, "trainer.val_only", False)):
        return None
    if int(value.expected_rollout_n) != 8 or int(_get(config, "env.rollout.n", 0)) != 8:
        raise ValueError("boundary_loto training requires expected_rollout_n=8 and env.rollout.n=8")
    if int(_get(config, "actor_rollout_ref.rollout.n", 1)) != 1:
        raise ValueError("boundary_loto uses env.rollout.n=8; actor_rollout_ref.rollout.n must remain 1")
    if value.auxiliary_mode == "env_only":
        if (bool(_get(config, "actor_rollout_ref.actor.use_invalid_action_penalty", True))
                and float(_get(config, "actor_rollout_ref.actor.invalid_action_penalty_coef", 0.0)) != 0):
            raise ValueError("boundary_loto env_only requires disabling the effective invalid-action penalty")
        if (bool(_get(config, "algorithm.use_kl_in_reward", False))
                and bool(_get(config, "algorithm.team_event_gae.include_kl_shaping", True))):
            raise ValueError("boundary_loto env_only requires disabling reward-KL shaping")
    every = _get(config, "algorithm.team_event_gae.credit_trace.every_n_steps", 1)
    if int(every) != every or int(every) < 1:
        raise ValueError("team_event_gae.credit_trace.every_n_steps must be a positive integer")
    if not value.idf_path:
        raise ValueError("boundary_loto requires a readable frozen IDF asset (value.idf_path)")
    asset = load_frozen_kernel(value)
    return value, asset


def build_boundary_run_manifest(config, *, run_id, worker_groups=None):
    if _get(config, "algorithm.adv_estimator") != "team_event_gae":
        return None
    validated = validate_boundary_training_config(config)
    if validated is None:
        return None
    value, asset = validated
    value_options = asdict(value)
    # Asset location may differ across nodes/restarts; its content may not.
    value_options.pop("idf_path", None)
    actor_keys = (
        "ppo_mini_update_num", "ppo_epochs", "loss_agg_mode", "use_dynamic_bsz",
        "ulysses_sequence_parallel_size", "ppo_micro_batch_size_per_gpu",
        "use_invalid_action_penalty", "invalid_action_penalty_coef",
        "use_kl_loss", "kl_loss_coef", "entropy_coeff", "optim",
    )
    def actor_contract(ref):
        actor = ref.get("actor", {})
        return {"actor": {k: actor.get(k) for k in actor_keys},
                "generation_n": _get(ref, "rollout.n", 1)}
    groups = {str(wg): [actor_contract(ref) for ref in refs]
              for wg, refs in (worker_groups or {}).items()}
    compatibility = {
        "value": value_options,
        "idf_sha256": asset.idf_sha256,
        "env_rollout_n": _get(config, "env.rollout.n"),
        "gamma": _get(config, "algorithm.gamma"),
        "lambda_env": _get(config, "algorithm.lam"),
        "gamma_internal": _get(config, "algorithm.team_event_gae.internal_gamma", 1.0),
        "lambda_internal": _get(config, "algorithm.team_event_gae.internal_lam", 1.0),
        "normalize_advantages": _get(config, "algorithm.team_event_gae.normalize_advantages", True),
        "norm_adv_by_std": _get(config, "algorithm.norm_adv_by_std_in_grpo", True),
        "include_kl_shaping": _get(config, "algorithm.team_event_gae.include_kl_shaping", True),
        "use_kl_in_reward": _get(config, "algorithm.use_kl_in_reward", False),
        "kl_penalty": _get(config, "algorithm.kl_penalty"),
        "kl_ctrl": _get(config, "algorithm.kl_ctrl"),
        "top_level_actor": actor_contract(_get(config, "actor_rollout_ref", {})),
        "worker_groups": groups,
        "search": _get(config, "env.search", {}),
        "search_protocol": _get(config, "env.search_protocol", {}),
        "max_steps": _get(config, "env.max_steps"),
        "max_response_length": _get(config, "data.max_response_length"),
        "agent_ids": _get(config, "agent.agent_ids"),
        "model_ids": _get(config, "agent.model_ids"),
        "model_sharing": _get(config, "agent.model_sharing"),
    }
    return {
        "schema_version": "team_event_boundary_manifest_v1", "run_id": str(run_id),
        "idf_path": str(Path(value.idf_path).resolve()),
        "compatibility": compatibility,
        "experiment_name": _get(config, "trainer.experiment_name"),
        "source_revision": _get(config, "trainer.event_trace.source_revision"),
        "sampling": _get(config, "actor_rollout_ref.rollout", {}),
        "data_seed": _get(config, "data.seed"), "env_seed": _get(config, "env.seed"),
    }


def save_boundary_run_manifest(manifest, output_dir, filename="boundary_loto_manifest.json"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    # Existing checkpoint manifests may be re-saved only if identical.
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Refusing to overwrite a different boundary manifest: {path}")
        return str(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return str(path)


def check_boundary_checkpoint_manifest(current, checkpoint_dir):
    path = Path(checkpoint_dir) / "boundary_loto_manifest.json"
    if not path.exists():
        if current is not None:
            raise ValueError("boundary_loto resume requires the checkpoint's boundary_loto_manifest.json; "
                             "a legacy checkpoint is not a same-experiment resume")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if current is None:
        raise ValueError("Cannot resume a boundary_loto checkpoint under a legacy value mode")
    if saved.get("compatibility") != current.get("compatibility"):
        raise ValueError("boundary_loto resume configuration/IDF digest differs from its checkpoint manifest")


_DIAGNOSTIC_FIELDS = {
    "value_pre": "value_pre", "value_post": "value_post",
    "delta_env": "delta_env", "delta_used": "delta_used",
    "task_raw_advantage": "task_raw_advantages", "aux_raw_advantage": "aux_raw_advantages",
    "train_raw_advantage": "raw_advantages", "candidate_advantage": "advantages",
    "env_reward": "reward_env", "invalid_reward": "reward_invalid", "kl_reward": "reward_kl",
    "reward_used": "reward_used", "gamma_edge": "gamma", "lambda_edge": "lambda",
    "normalization_mean": "normalization_mean", "normalization_std": "normalization_std",
    "normalization_applied": "normalization_applied",
    "question_parent": "question_parent", "same_round_parent": "parent",
    "same_round_peer_count": "same_round_peer_count", "kernel_mass": "kernel_mass",
    "kernel_ess": "kernel_ess", "positive_peer_count": "positive_peer_count",
    "cross_round_peer_count": "cross_round_peer_count",
}


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite scalar in boundary credit trace")
    return value


def build_boundary_credit_records(batch, manifest, *, global_step, split):
    """Read scalar diagnostics without re-estimating values or re-grouping peers."""
    nt, tensors = batch.non_tensor_batch, batch.batch
    metadata = batch.meta_info.get("boundary_loto", {})
    expected_hash = manifest["compatibility"]["idf_sha256"]
    if metadata.get("idf_sha256") != expected_hash:
        raise ValueError("Credit trace IDF digest differs from frozen run manifest")
    details = metadata.get("details_by_event", {})
    value = manifest["compatibility"]["value"]
    required = ["event_boundary_" + suffix for suffix in _DIAGNOSTIC_FIELDS.values()]
    missing = [field for field in required if field not in tensors]
    if missing:
        raise ValueError(f"Credit trace requires post-advantage diagnostics: {missing}")
    records = {}
    for row, raw_uid in enumerate(nt["event_uid"]):
        event_uid = str(raw_uid)
        record = {
            "schema_version": "team_event_boundary_credit_v1", "run_id": manifest["run_id"],
            "split": split, "global_step": int(global_step), "event_uid": event_uid,
            "value_mode": value["mode"], "application": value["application"],
            "auxiliary_mode": value["auxiliary_mode"], "idf_sha256": expected_hash,
        }
        for field in ("uid", "traj_uid", "event_index", "env_step_index", "agent_id", "event_type", "wg_id",
                      "route_target", "route_reason"):
            record[field] = _plain(nt[field][row]) if field in nt else None
        record["done"] = bool(nt["env_done"][row])
        for label, suffix in _DIAGNOSTIC_FIELDS.items():
            record[label] = _plain(tensors["event_boundary_" + suffix][row])
        # Shadow keeps the actor's legacy normalized advantage. Do not label
        # the candidate tensor as the actor input in this case.
        response_mask = tensors["response_mask"][row].bool()
        active = tensors["advantages"][row][response_mask]
        if active.numel() == 0:
            raise ValueError(f"No response tokens for credit trace event {event_uid}")
        first = active[0]
        if not bool((active == first).all()):
            raise ValueError(f"Non-uniform event advantage for {event_uid}")
        record["actor_advantage"] = _plain(first)
        detail = details.get(event_uid, {})
        for field in ("document_key_sha256", "previous_document_key_sha256"):
            record[field] = detail.get(field)
        if "matched_peers" in detail:
            record["matched_peers"] = detail["matched_peers"]
        if event_uid in records and records[event_uid] != record:
            raise ValueError(f"Conflicting duplicate credit rows for event {event_uid}")
        records[event_uid] = record
    return sorted(records.values(), key=lambda r: (str(r["uid"]), str(r["traj_uid"]), r["event_index"]))


def dump_boundary_credit_trace(batch, output_dir, manifest, *, global_step, split="train", part_index=0):
    records = build_boundary_credit_records(batch, manifest, global_step=global_step, split=split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_boundary_run_manifest(manifest, output, filename=f"manifest_{manifest['run_id']}.json")
    while True:
        path = output / f"credit_{split}_step_{int(global_step):08d}_part_{part_index:05d}.jsonl"
        try:
            stream = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            part_index += 1
    with stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return str(path), len(records), part_index


# ---------------------------------------------------------------------------
# Math answer-LOTO contracts
# ---------------------------------------------------------------------------

def validate_math_training_config(config):
    """Validate the effective Math answer-LOTO configuration before updates.

    Mirrors ``validate_boundary_training_config`` for the Math orchestra:
    env grouping implements the eight-trajectory question group, the actor
    rollout n stays one, and val-only runs never compute the estimator.
    """

    from verl.trainer.ppo.team_event_math_value import MathValueConfig

    value = MathValueConfig.from_mapping(_get(config, "algorithm.team_event_gae.math_value", {}))
    if value.mode == "off":
        return None
    value.validate_runtime(
        gamma=float(_get(config, "algorithm.gamma", 1.0)),
        internal_gamma=float(_get(config, "algorithm.team_event_gae.internal_gamma", 1.0)),
        agent_local_mode=str(_get(config, "algorithm.team_event_gae.agent_local.mode", "off")),
    )
    search_value = _get(config, "algorithm.team_event_gae.value", {})
    if str(search_value.get("mode", "legacy_loto")) == "boundary_loto":
        raise ValueError("math_answer_loto conflicts with the Search boundary_loto value mode")
    if _get(config, "agent.orchestra_type") != "math" or _get(config, "env.env_name") != "math":
        raise ValueError("math_answer_loto requires the Math orchestra and Math environment")
    if bool(_get(config, "trainer.val_only", False)):
        return None
    if int(_get(config, "env.rollout.n", 0)) != int(value.expected_rollout_n):
        raise ValueError(
            "math_answer_loto training requires env.rollout.n == math_value.expected_rollout_n "
            f"({value.expected_rollout_n})"
        )
    if int(_get(config, "actor_rollout_ref.rollout.n", 1)) != 1:
        raise ValueError("math_answer_loto uses env.rollout.n grouping; actor_rollout_ref.rollout.n must remain 1")
    if int(_get(config, "agent.orchestra.math.max_loop_num", 0)) != int(value.expected_max_loop_num):
        raise ValueError(
            "math_answer_loto training requires orchestra math.max_loop_num == "
            f"math_value.expected_max_loop_num ({value.expected_max_loop_num})"
        )
    if value.auxiliary_mode == "env_only":
        if (bool(_get(config, "actor_rollout_ref.actor.use_invalid_action_penalty", True))
                and float(_get(config, "actor_rollout_ref.actor.invalid_action_penalty_coef", 0.0)) != 0):
            raise ValueError("math_answer_loto env_only requires disabling the effective invalid-action penalty")
        if (bool(_get(config, "algorithm.use_kl_in_reward", False))
                and bool(_get(config, "algorithm.team_event_gae.include_kl_shaping", True))):
            raise ValueError("math_answer_loto env_only requires disabling reward-KL shaping")
    every = _get(config, "algorithm.team_event_gae.credit_trace.every_n_steps", 1)
    if int(every) != every or int(every) < 1:
        raise ValueError("team_event_gae.credit_trace.every_n_steps must be a positive integer")
    return value


def build_math_run_manifest(config, *, run_id, worker_groups=None):
    if _get(config, "algorithm.adv_estimator") != "team_event_gae":
        return None
    value = validate_math_training_config(config)
    if value is None:
        return None
    value_options = asdict(value)
    actor_keys = (
        "ppo_mini_update_num", "ppo_epochs", "loss_agg_mode", "use_dynamic_bsz",
        "ulysses_sequence_parallel_size", "ppo_micro_batch_size_per_gpu",
        "use_invalid_action_penalty", "invalid_action_penalty_coef",
        "use_kl_loss", "kl_loss_coef", "entropy_coeff", "optim",
    )

    def actor_contract(ref):
        actor = ref.get("actor", {})
        return {"actor": {k: actor.get(k) for k in actor_keys},
                "generation_n": _get(ref, "rollout.n", 1)}

    groups = {str(wg): [actor_contract(ref) for ref in refs]
              for wg, refs in (worker_groups or {}).items()}
    compatibility = {
        "value": value_options,
        "env_rollout_n": _get(config, "env.rollout.n"),
        "gamma": _get(config, "algorithm.gamma"),
        "lambda_env": _get(config, "algorithm.lam"),
        "gamma_internal": _get(config, "algorithm.team_event_gae.internal_gamma", 1.0),
        "lambda_internal": _get(config, "algorithm.team_event_gae.internal_lam", 1.0),
        "normalize_advantages": _get(config, "algorithm.team_event_gae.normalize_advantages", True),
        "norm_adv_by_std": _get(config, "algorithm.norm_adv_by_std_in_grpo", True),
        "include_kl_shaping": _get(config, "algorithm.team_event_gae.include_kl_shaping", True),
        "use_kl_in_reward": _get(config, "algorithm.use_kl_in_reward", False),
        "kl_penalty": _get(config, "algorithm.kl_penalty"),
        "kl_ctrl": _get(config, "algorithm.kl_ctrl"),
        "top_level_actor": actor_contract(_get(config, "actor_rollout_ref", {})),
        "worker_groups": groups,
        "orchestra": _get(config, "agent.orchestra", {}),
        "max_steps": _get(config, "env.max_steps"),
        "max_response_length": _get(config, "data.max_response_length"),
        "agent_ids": _get(config, "agent.agent_ids"),
        "model_ids": _get(config, "agent.model_ids"),
        "model_sharing": _get(config, "agent.model_sharing"),
    }
    return {
        "schema_version": "team_event_math_manifest_v1", "run_id": str(run_id),
        "compatibility": compatibility,
        "experiment_name": _get(config, "trainer.experiment_name"),
        "source_revision": _get(config, "trainer.event_trace.source_revision"),
        "sampling": _get(config, "actor_rollout_ref.rollout", {}),
        "data_seed": _get(config, "data.seed"), "env_seed": _get(config, "env.seed"),
    }


def save_math_run_manifest(manifest, output_dir, filename="math_loto_manifest.json"):
    return save_boundary_run_manifest(manifest, output_dir, filename=filename)


def check_math_checkpoint_manifest(current, checkpoint_dir):
    path = Path(checkpoint_dir) / "math_loto_manifest.json"
    if not path.exists():
        if current is not None:
            raise ValueError("math_answer_loto resume requires the checkpoint's math_loto_manifest.json; "
                             "a legacy checkpoint is not a same-experiment resume")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if current is None:
        raise ValueError("Cannot resume a math_answer_loto checkpoint with the estimator disabled")
    if saved.get("compatibility") != current.get("compatibility"):
        raise ValueError("math_answer_loto resume configuration differs from its checkpoint manifest")


_MATH_CREDIT_TRACE_SCHEMA = "team_event_math_credit_v1"

_MATH_DIAGNOSTIC_FIELDS = {
    "value_pre": "value_pre", "value_post": "value_post",
    "delta_env": "delta_env", "delta_used": "delta_used",
    "task_raw_advantage": "task_raw_advantages", "aux_raw_advantage": "aux_raw_advantages",
    "train_raw_advantage": "raw_advantages", "candidate_advantage": "advantages",
    "env_reward": "reward_env", "invalid_reward": "reward_invalid", "kl_reward": "reward_kl",
    "reward_used": "reward_used", "gamma_edge": "gamma", "lambda_edge": "lambda",
    "normalization_mean": "normalization_mean", "normalization_std": "normalization_std",
    "question_parent": "question_parent", "same_round_parent": "parent",
    "same_round_peer_count": "same_round_peer_count", "kernel_mass": "kernel_mass",
    "kernel_ess": "kernel_ess", "effective_ess": "effective_ess",
    "max_effective_coefficient": "max_effective_coefficient",
    "used_peer_count": "used_peer_count", "cross_round_peer_count": "cross_round_peer_count",
}


def build_math_credit_records(batch, manifest, *, global_step, split):
    """Read scalar diagnostics without re-estimating values or re-grouping peers."""

    nt, tensors = batch.non_tensor_batch, batch.batch
    metadata = batch.meta_info.get("math_answer_loto", {})
    if metadata.get("schema_version") != _MATH_CREDIT_TRACE_SCHEMA:
        raise ValueError("Math credit trace requires the math_answer_loto advantage metadata")
    details = metadata.get("details_by_event", {})
    value = manifest["compatibility"]["value"]
    required = ["event_math_" + suffix for suffix in _MATH_DIAGNOSTIC_FIELDS.values()]
    missing = [field for field in required if field not in tensors]
    if missing:
        raise ValueError(f"Math credit trace requires post-advantage diagnostics: {missing}")
    records = {}
    for row, raw_uid in enumerate(nt["event_uid"]):
        event_uid = str(raw_uid)
        record = {
            "schema_version": _MATH_CREDIT_TRACE_SCHEMA, "run_id": manifest["run_id"],
            "split": split, "global_step": int(global_step), "event_uid": event_uid,
            "value_mode": value["mode"], "auxiliary_mode": value["auxiliary_mode"],
            "parser_version": value["parser_version"],
        }
        for field in ("uid", "traj_uid", "event_index", "role_event_index", "env_step_index",
                      "agent_id", "event_type", "wg_id", "loop_index", "verifier_decision",
                      "submitted_solution_event_uid", "transition_owner_event_uid",
                      "orchestration_stop_reason"):
            record[field] = _plain(nt[field][row]) if field in nt else None
        record["done"] = bool(nt["env_done"][row])
        for label, suffix in _MATH_DIAGNOSTIC_FIELDS.items():
            record[label] = _plain(tensors["event_math_" + suffix][row])
        response_mask = tensors["response_mask"][row].bool()
        active = tensors["advantages"][row][response_mask]
        if active.numel() == 0:
            raise ValueError(f"No response tokens for math credit trace event {event_uid}")
        first = active[0]
        if not bool((active == first).all()):
            raise ValueError(f"Non-uniform event advantage for {event_uid}")
        record["actor_advantage"] = _plain(first)
        detail = details.get(event_uid, {})
        for field in ("parser_status", "branch", "answer_key_sha256"):
            record[field] = detail.get(field)
        if event_uid in records and records[event_uid] != record:
            raise ValueError(f"Conflicting duplicate math credit rows for event {event_uid}")
        records[event_uid] = record
    return sorted(records.values(), key=lambda r: (str(r["uid"]), str(r["traj_uid"]), r["event_index"]))


def dump_math_credit_trace(batch, output_dir, manifest, *, global_step, split="train", part_index=0):
    records = build_math_credit_records(batch, manifest, global_step=global_step, split=split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_math_run_manifest(manifest, output, filename=f"manifest_{manifest['run_id']}.json")
    while True:
        path = output / f"credit_{split}_step_{int(global_step):08d}_part_{part_index:05d}.jsonl"
        try:
            stream = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            part_index += 1
    with stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return str(path), len(records), part_index


# ---------------------------------------------------------------------------
# Task-dispatching wrappers (Search boundary_loto / Math math_answer_loto)
# ---------------------------------------------------------------------------

def build_team_event_run_manifest(config, *, run_id, worker_groups=None):
    """Build the single active team-event manifest for the configured task."""

    if _get(config, "algorithm.adv_estimator") != "team_event_gae":
        return None
    orchestra = str(_get(config, "agent.orchestra_type", ""))
    math_mode = str(_get(config, "algorithm.team_event_gae.math_value.mode", "off") or "off")
    value_mode = str(_get(config, "algorithm.team_event_gae.value.mode", "legacy_loto") or "legacy_loto")
    if orchestra == "math":
        if math_mode != "math_answer_loto" and value_mode != "legacy_loto":
            raise ValueError("The Math orchestra supports only math_value=math_answer_loto")
        return build_math_run_manifest(config, run_id=run_id, worker_groups=worker_groups)
    if orchestra == "search":
        if math_mode != "off":
            raise ValueError("math_value is the Math-orchestra estimator; disable it for Search runs")
        return build_boundary_run_manifest(config, run_id=run_id, worker_groups=worker_groups)
    raise ValueError(f"team_event_gae does not support orchestra_type={orchestra!r}")


def save_team_event_run_manifest(manifest, output_dir, **kwargs):
    if manifest is None:
        return None
    if manifest["schema_version"] == "team_event_math_manifest_v1":
        return save_math_run_manifest(manifest, output_dir, **kwargs)
    return save_boundary_run_manifest(manifest, output_dir, **kwargs)


def check_team_event_checkpoint_manifest(current, checkpoint_dir):
    boundary_path = Path(checkpoint_dir) / "boundary_loto_manifest.json"
    math_path = Path(checkpoint_dir) / "math_loto_manifest.json"
    current_schema = None if current is None else current.get("schema_version")
    if current_schema == "team_event_math_manifest_v1":
        if boundary_path.exists():
            raise ValueError("Cannot resume a Search boundary_loto checkpoint as a math_answer_loto run")
        return check_math_checkpoint_manifest(current, checkpoint_dir)
    if current_schema == "team_event_boundary_manifest_v1":
        if math_path.exists():
            raise ValueError("Cannot resume a math_answer_loto checkpoint as a Search boundary_loto run")
        return check_boundary_checkpoint_manifest(current, checkpoint_dir)
    # No active estimator: refuse either frozen-contract checkpoint.
    if math_path.exists():
        raise ValueError("Cannot resume a math_answer_loto checkpoint with the estimator disabled")
    return check_boundary_checkpoint_manifest(current, checkpoint_dir)


def dump_team_event_credit_trace(batch, output_dir, manifest, *, global_step, split="train", part_index=0):
    if manifest["schema_version"] == "team_event_math_manifest_v1":
        return dump_math_credit_trace(batch, output_dir, manifest, global_step=global_step,
                                      split=split, part_index=part_index)
    return dump_boundary_credit_trace(batch, output_dir, manifest, global_step=global_step,
                                      split=split, part_index=part_index)
