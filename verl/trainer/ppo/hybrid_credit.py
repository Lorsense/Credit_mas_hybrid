"""Detached mix credit applied once AFTER the unchanged zxj event estimator.

Reward, returns, LOTO values and base event diagnostics are never rewritten.
Entropy ranks are prepared on unique rollout events before optimizer padding.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING
import numpy as np
import torch

from verl.utils.entropy_credit import compute_entropy_credit_multipliers, first_unique_action_indices
from verl.utils.sparse_entropy_credit import prepare_sparse_entropy_credit, finalize_sparse_entropy_credit

if TYPE_CHECKING:
    from verl import DataProto


def normalize_agent_id(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def compute_response_mask(data):
    width = data.batch["responses"].shape[-1]
    return data.batch["attention_mask"][:, -width:]


def prepare_hybrid_action_metadata(data):
    """Check the one-to-one event/action identity and retain mask facts."""
    meta = data.non_tensor_batch
    required = {"event_uid", "event_index", "role_event_index", "traj_uid", "uid", "agent_id"}
    missing = required.difference(meta)
    if missing:
        raise ValueError(f"Hybrid requires the zxj event protocol: {sorted(missing)}")
    turns = np.asarray(meta["role_event_index"], dtype=np.int64)
    if "role_turn_index" in meta and not np.array_equal(turns, meta["role_turn_index"]):
        raise ValueError("role_turn_index must equal the same-role role_event_index")
    meta["role_turn_index"] = turns.copy()
    if "value_action_index" in meta and not np.array_equal(meta["value_action_index"], meta["event_index"]):
        raise ValueError("value_action_index must equal the global event_index")
    events, actions = {}, {}
    trajectories = {}
    for row in range(len(data)):
        event = str(meta["event_uid"][row])
        action = (str(meta["traj_uid"][row]), str(meta["agent_id"][row]), int(turns[row]))
        if not event or event in events or action in actions:
            raise ValueError("Hybrid entropy preparation requires one row per unique real event")
        events[event], actions[action] = action, event
        trajectories.setdefault(action[0], []).append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda row: int(meta["event_index"][row]))
        role_counts = {}
        for index, row in enumerate(rows):
            role = str(meta["agent_id"][row])
            role_index = role_counts.get(role, 0)
            if int(meta["event_index"][row]) != index or int(turns[row]) != role_index:
                raise ValueError("Hybrid requires contiguous global and same-role event indices")
            role_counts[role] = role_index + 1
    width = data.batch["responses"].shape[-1]
    lengths = compute_response_mask(data).sum(-1).detach().cpu().numpy().astype(np.int64)
    meta["pure_entropy_response_tokens"] = lengths
    # Preserve authoritative backend truncation when present. At a hard token
    # limit, conservatively exclude the event even if a backend omitted it.
    prior = np.asarray(meta.get("pure_entropy_truncated", np.zeros(len(data), dtype=bool)), dtype=bool)
    meta["pure_entropy_truncated"] = prior | (lengths >= width)
    return data


def prepare_entropy_credit(data: DataProto, entropy_credit_config) -> tuple[DataProto, dict[str, float]]:
    """Compute detached entropy-credit factors before any batch padding/copying.

    ``adjust_batch`` may randomly duplicate actions independently for each
    worker group.  Ranking must therefore happen on the original rollout set;
    the resulting non-tensor factors then follow each action through all later
    split, copy, concat, and reorder operations.
    """

    required = {
        "uid",
        "traj_uid",
        "agent_id",
        "role_turn_index",
        "pass",
        "top16_entropy",
        "top16_entropy_mean",
    }
    missing = required.difference(data.non_tensor_batch)
    if missing:
        raise KeyError(f"entropy credit is missing rollout fields: {sorted(missing)}")

    terminal_success = np.empty(len(data), dtype=bool)
    for row, value in enumerate(data.non_tensor_batch["pass"]):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pass[{row}] must be a binary terminal outcome, got {value!r}") from exc
        if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
            raise ValueError(f"pass[{row}] must be 0 or 1, got {value!r}")
        terminal_success[row] = bool(numeric_value)

    action_valid = np.asarray(
        data.non_tensor_batch.get("is_action_valid", np.ones(len(data), dtype=bool)),
        dtype=bool,
    )
    multipliers = compute_entropy_credit_multipliers(
        prompt_group_ids=data.non_tensor_batch["uid"],
        trajectory_ids=data.non_tensor_batch["traj_uid"],
        agent_ids=data.non_tensor_batch["agent_id"],
        role_turn_indices=data.non_tensor_batch["role_turn_index"],
        terminal_success=terminal_success,
        action_entropies=data.non_tensor_batch["top16_entropy_mean"],
        action_valid=action_valid,
        action_scale=float(entropy_credit_config.get("action_scale", 0.2)),
        # Sparse stage 2 replaces the legacy delta multiplier; stage 1 is unchanged.
        trajectory_scale=(
            0.0 if entropy_credit_config.get("sparse", {}).get("enable", False)
            else float(entropy_credit_config.get("trajectory_scale", 0.1))
        ),
        trajectory_deadzone=float(entropy_credit_config.get("trajectory_deadzone", 0.05)),
        multiplier_min=float(entropy_credit_config.get("multiplier_min", 0.8)),
        multiplier_max=float(entropy_credit_config.get("multiplier_max", 1.2)),
        final_multiplier_min=float(entropy_credit_config.get("final_multiplier_min", 0.8)),
        final_multiplier_max=float(entropy_credit_config.get("final_multiplier_max", 1.2)),
        relative_epsilon=float(entropy_credit_config.get("relative_epsilon", 1e-12)),
    )

    data.non_tensor_batch["entropy_credit_terminal_success"] = terminal_success
    data.non_tensor_batch["entropy_credit_action_multiplier"] = multipliers["action"]
    data.non_tensor_batch["entropy_credit_trajectory_multiplier"] = multipliers["trajectory"]
    data.non_tensor_batch["entropy_credit_final_multiplier"] = multipliers["final"]
    data.non_tensor_batch["entropy_credit_relative_change"] = multipliers["relative_change"]

    entropies = np.asarray(data.non_tensor_batch["top16_entropy_mean"], dtype=np.float64)
    entropy_valid = np.asarray(multipliers["entropy_valid"], dtype=bool)
    roles = np.asarray(data.non_tensor_batch["agent_id"], dtype=object)
    turns = np.asarray(data.non_tensor_batch["role_turn_index"], dtype=np.int64)
    entropy_stats = data.non_tensor_batch["top16_entropy"]
    raw_final = multipliers["action"] * multipliers["trajectory"]

    metrics: dict[str, float] = {}

    def add_scope_metrics(prefix: str, scope: np.ndarray) -> None:
        valid_scope = scope & action_valid
        entropy_scope = scope & entropy_valid
        valid_count = int(valid_scope.sum())
        metrics[f"{prefix}/action_count"] = float(scope.sum())
        metrics[f"{prefix}/valid_action_fraction"] = float(valid_count / max(int(scope.sum()), 1))
        metrics[f"{prefix}/entropy_action_coverage"] = float(entropy_scope.sum() / max(valid_count, 1))

        if entropy_scope.any():
            values = entropies[entropy_scope]
            metrics[f"{prefix}/top16_entropy_mean"] = float(values.mean())
            metrics[f"{prefix}/top16_entropy_std"] = float(values.std(ddof=0))
            metrics[f"{prefix}/top16_entropy_p10"] = float(np.quantile(values, 0.10))
            metrics[f"{prefix}/top16_entropy_p50"] = float(np.quantile(values, 0.50))
            metrics[f"{prefix}/top16_entropy_p90"] = float(np.quantile(values, 0.90))
            for outcome_name, outcome in (("success", True), ("failure", False)):
                outcome_scope = entropy_scope & (terminal_success == outcome)
                if outcome_scope.any():
                    metrics[f"{prefix}/{outcome_name}_top16_entropy_mean"] = float(
                        entropies[outcome_scope].mean()
                    )

            token_coverages = []
            effective_supports = []
            for row in np.flatnonzero(entropy_scope):
                stats = entropy_stats[row]
                if isinstance(stats, dict) and stats.get("coverage") is not None:
                    token_coverages.append(float(stats["coverage"]))
                if isinstance(stats, dict) and stats.get("effective_support") is not None:
                    effective_supports.append(float(stats["effective_support"]))
            if token_coverages:
                metrics[f"{prefix}/top16_token_coverage_mean"] = float(np.mean(token_coverages))
            if effective_supports:
                metrics[f"{prefix}/top16_effective_support_mean"] = float(np.mean(effective_supports))

        if valid_scope.any():
            for name in ("action", "trajectory", "final"):
                values = multipliers[name][valid_scope]
                metrics[f"{prefix}/{name}_multiplier_mean"] = float(values.mean())
                metrics[f"{prefix}/{name}_multiplier_min"] = float(values.min())
                metrics[f"{prefix}/{name}_multiplier_max"] = float(values.max())
            if not entropy_credit_config.get("sparse", {}).get("enable", False):
                metrics[f"{prefix}/final_clip_fraction"] = float(
                    np.mean(~np.isclose(raw_final[valid_scope], multipliers["final"][valid_scope]))
                )

        later_scope = valid_scope & (turns > 0)
        observed_scope = later_scope & np.isfinite(multipliers["relative_change"])
        metrics[f"{prefix}/trajectory_eligible_fraction"] = float(
            later_scope.sum() / max(valid_count, 1)
        )
        metrics[f"{prefix}/trajectory_observed_fraction"] = float(
            observed_scope.sum() / max(int(later_scope.sum()), 1)
        )
        if observed_scope.any():
            changes = multipliers["relative_change"][observed_scope]
            deadzone = float(entropy_credit_config.get("trajectory_deadzone", 0.05))
            metrics[f"{prefix}/entropy_rise_fraction"] = float(np.mean(changes > deadzone))
            metrics[f"{prefix}/entropy_fall_fraction"] = float(np.mean(changes < -deadzone))
            metrics[f"{prefix}/entropy_deadzone_fraction"] = float(np.mean(np.abs(changes) <= deadzone))
            metrics[f"{prefix}/trajectory_active_fraction"] = float(
                np.mean(~np.isclose(multipliers["trajectory"][observed_scope], 1.0))
            )

    add_scope_metrics("entropy_credit/global", np.ones(len(data), dtype=bool))
    for role in np.unique(roles):
        role_key = normalize_agent_id(str(role))
        add_scope_metrics(f"entropy_credit/{role_key}", roles == role)

    return data, metrics


def prepare_sparse_entropy_credit_for_batch(data: DataProto, sparse_config) -> tuple[DataProto, dict[str, float]]:
    """Freeze pair statistics on logical rollout actions before distributed padding."""
    response_width = data.batch["responses"].shape[-1]
    response_mask = (
        compute_response_mask(data) if response_width
        else data.batch["attention_mask"][:, :0]
    )
    response_tokens = response_mask.sum(-1).detach().cpu().numpy().astype(np.int64)
    data.non_tensor_batch["pure_entropy_response_tokens"] = response_tokens
    # Keep backend truncation and conservatively exclude full-width responses.
    data.non_tensor_batch["pure_entropy_truncated"] = (
        np.asarray(data.non_tensor_batch.get("pure_entropy_truncated", np.zeros(len(data), dtype=bool)), dtype=bool)
        | (response_tokens >= response_width))
    prepared, metrics = prepare_sparse_entropy_credit(data.non_tensor_batch, sparse_config)
    data.non_tensor_batch.update(prepared)
    return data, metrics


def finalize_sparse_entropy_credit_for_advantages(
    data: DataProto, sparse_config
) -> tuple[DataProto, dict[str, float]]:
    """Select gates with actual zxj event-advantage signs, then merge into the untouched stage 1."""
    advantages = data.batch["advantages"].detach().float()
    response_mask = data.batch["response_mask"].bool()
    if advantages.shape != response_mask.shape:
        raise ValueError("pure entropy credit requires advantages and response_mask with the same shape")
    token_counts = response_mask.sum(-1)
    # torch.where avoids contaminating a valid action with NaNs on masked pads.
    base_advantages = (
        torch.where(response_mask, advantages, torch.zeros_like(advantages)).sum(-1)
        / token_counts.clamp_min(1)
    ).cpu().numpy()
    result = finalize_sparse_entropy_credit(data.non_tensor_batch, base_advantages, sparse_config)
    data.non_tensor_batch.update(result["metadata"])
    data.non_tensor_batch["entropy_credit_trajectory_multiplier"] = result["trajectory_multipliers"]
    data.non_tensor_batch["entropy_credit_final_multiplier"] = result["final_multipliers"]
    metrics = dict(result["metrics"])

    # Report the factors actually used by PPO, once per logical action. These
    # overwrite the provisional stage-1-only summaries produced before event advantages.
    unique = first_unique_action_indices(
        data.non_tensor_batch["traj_uid"],
        data.non_tensor_batch["agent_id"],
        data.non_tensor_batch["role_turn_index"],
    )
    roles = np.asarray(data.non_tensor_batch["agent_id"], dtype=object)[unique]
    valid = np.asarray(
        data.non_tensor_batch.get("is_action_valid", np.ones(len(data), dtype=bool)), dtype=bool
    )[unique]
    lower = float(sparse_config.get("multiplier_min", 0.8))
    upper = float(sparse_config.get("multiplier_max", 1.2))
    scopes = [("global", np.ones(len(unique), dtype=bool))]
    scopes.extend((normalize_agent_id(str(role)), roles == role) for role in np.unique(roles))
    for scope_name, scope in scopes:
        rows = unique[scope & valid]
        prefix = f"entropy_credit/{scope_name}"
        for name in ("action", "trajectory", "final"):
            values = np.asarray(data.non_tensor_batch[f"entropy_credit_{name}_multiplier"])[rows]
            for statistic, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                metrics[f"{prefix}/{name}_multiplier_{statistic}"] = float(fn(values)) if len(values) else 1.0
        final = np.asarray(result["final_multipliers"])[rows]
        effective = np.asarray(result["trajectory_multipliers"])[rows]
        metrics[f"{prefix}/trajectory_active_fraction"] = (
            float(np.mean(np.abs(effective - 1.0) > 1e-12)) if len(rows) else 0.0
        )
        # Touching a bound differs from a clip operation or a normalization shift.
        metrics[f"{prefix}/final_bound_fraction"] = (
            float(np.mean(np.isclose(final, lower) | np.isclose(final, upper))) if len(rows) else 0.0
        )
    return data, metrics


def apply_entropy_credit_to_advantages(data: DataProto) -> DataProto:
    """Scale only the already-computed advantages by a detached action scalar."""

    if "advantages" not in data.batch:
        raise KeyError("base advantages must be computed before entropy credit is applied")
    if "entropy_credit_final_multiplier" not in data.non_tensor_batch:
        raise KeyError("entropy-credit factors must be prepared before batch adjustment")
    advantages = data.batch["advantages"]
    final_multiplier = torch.as_tensor(
        np.asarray(data.non_tensor_batch["entropy_credit_final_multiplier"], dtype=np.float64),
        device=advantages.device,
        dtype=advantages.dtype,
    ).detach()
    if final_multiplier.shape != (len(data),):
        raise ValueError(
            f"entropy-credit multiplier has shape {final_multiplier.shape}, expected ({len(data)},)"
        )
    data.batch["advantages"] = advantages * final_multiplier.unsqueeze(-1)
    return data
