"""Event-level diagnostics for the joined zxj/mix/semantic method.

Existing entropy_credit, pure_entropy, semantic_value and entropy_control
metrics remain authoritative for ranks, gate reasons, qualification, risk,
calibration and training quality. This module adds the missing before/after
advantage comparison and the cross-module coverage checks. It neither mutates
the batch nor changes training. Padding copies are counted once by event_uid.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy cannot represent bfloat16 tensors directly.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _role(value):
    role = str(value).strip().lower().removesuffix(" agent")
    if role not in ("solver", "verifier"):
        raise ValueError(f"hybrid observability expects Math roles, got {value!r}")
    return role


def compute_hybrid_observability(batch, zero_tolerance=1e-8):
    """Return finite flat metrics after final credit and entropy-control prepare.

    Advantage summaries average valid response tokens within each event, then
    give every unique event equal weight. A zero group is an observed (uid,role)
    group whose base advantages are all within ``zero_tolerance``; its retention
    metric does not claim that all homogeneous-outcome groups must be zero.

    If scaling is disabled, final advantages are the unchanged base advantages.
    Semantic and stage-two metrics are emitted only when their metadata exists.
    Duplicate copies must agree on each observed scalar and identity. NaNs on
    masked response padding are ignored; NaNs on actual response tokens fail.
    """
    if not math.isfinite(float(zero_tolerance)) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    tensors, meta = batch.batch, batch.non_tensor_batch
    if "advantages" not in tensors or "event_uid" not in meta:
        return {}
    final_tokens = _numpy(tensors["advantages"]).astype(np.float64)
    if final_tokens.ndim != 2:
        raise ValueError("advantages must have shape [events, response tokens]")
    size, width = final_tokens.shape
    if "response_mask" in tensors:
        mask = _numpy(tensors["response_mask"]).astype(bool)
    else:
        mask = _numpy(tensors["attention_mask"])[:, -width:].astype(bool)
    if mask.shape != final_tokens.shape:
        raise ValueError("response mask and advantages must have equal shapes")
    base_tokens = _numpy(tensors.get("hybrid_base_advantages", tensors["advantages"])).astype(np.float64)
    if base_tokens.shape != final_tokens.shape:
        raise ValueError("base and final advantages must have equal shapes")
    counts = mask.sum(-1)
    if not np.isfinite(base_tokens[mask]).all() or not np.isfinite(final_tokens[mask]).all():
        raise ValueError("nonfinite advantage on a real response token")
    base = np.where(mask, base_tokens, 0.).sum(-1) / np.maximum(counts, 1)
    final = np.where(mask, final_tokens, 0.).sum(-1) / np.maximum(counts, 1)

    def column(name, *, source=meta, dtype=None):
        values = _numpy(source[name])
        if values.shape != (size,):
            raise ValueError(f"{name} must have one scalar per event")
        return values.astype(dtype) if dtype is not None else values

    ids = column("event_uid").astype(str)
    if any(not identity for identity in ids):
        raise ValueError("event_uid cannot be empty")
    roles = np.asarray([_role(value) for value in column("agent_id")])
    tasks = column("uid").astype(str)
    _, unique = np.unique(ids, return_index=True)
    # np.unique orders canonical rows by event identity, giving the same
    # reduction order even after arbitrary repacking or padding.
    canonical = {ids[row]: row for row in unique}
    first = np.asarray([canonical[identity] for identity in ids], dtype=np.int64)

    def checked(values, name):
        values = np.asarray(values)
        if values.dtype.kind in "biufc":
            equal = np.allclose(values, values[first], atol=1e-8, rtol=1e-6, equal_nan=True)
        else:
            equal = np.array_equal(values, values[first])
        if not equal:
            raise ValueError(f"conflicting event_uid copies in {name}")
        return values

    for name, values in (("agent_id", roles), ("uid", tasks), ("response_tokens", counts),
                         ("base_advantages", base), ("final_advantages", final)):
        checked(values, name)
    for name in ("traj_uid", "role_event_index"):
        if name in meta:
            checked(column(name), name)
    metrics = {"hybrid/unique_events": float(len(unique))}

    def emit(prefix, name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"nonfinite hybrid diagnostic: {prefix}/{name}")
        metrics[f"{prefix}/{name}"] = value

    def mean(values):
        return float(np.mean(values)) if len(values) else 0.

    stage_two = "pure_entropy_selected_gate" in meta
    if stage_two:
        success = checked(column("pass", dtype=bool), "pass")
        correction = checked(column("pure_entropy_correction_gate", dtype=float), "correction_gate")
        regression = checked(column("pure_entropy_regression_gate", dtype=float), "regression_gate")
        candidate = np.where(success, correction, regression) > 0
        selected = checked(column("pure_entropy_selected_gate", dtype=float), "selected_gate") > 0
        c1 = checked(column("entropy_credit_action_multiplier", dtype=float), "stage_one_multiplier")
        cfinal = checked(column("entropy_credit_final_multiplier", dtype=float), "final_multiplier")
        if not np.isfinite(c1).all() or not np.isfinite(cfinal).all():
            raise ValueError("nonfinite credit coefficient")

    semantic = "value_control_eligible" in meta
    if semantic:
        eligible = checked(column("value_control_eligible", dtype=bool), "value_control_eligible")
        available = checked(column("value_credit_available", dtype=bool), "value_credit_available")
        delta = checked(column("value_credit_delta", dtype=float), "value_credit_delta")
        available = eligible & available & np.isfinite(delta)
        if "entropy_control_weight" in tensors:
            weights = checked(column("entropy_control_weight", source=tensors, dtype=float), "control_weight")
            caps = checked(column("entropy_control_cap", source=tensors, dtype=float), "control_cap")
            control_valid = checked(column("entropy_control_valid", source=tensors, dtype=bool), "control_valid")
            if not np.isfinite(weights).all() or not np.isfinite(caps[control_valid]).all():
                raise ValueError("nonfinite entropy control weight/cap")
            gated = control_valid & (weights > 0)
        else:
            weights, caps, gated = np.zeros(size), np.zeros(size), np.zeros(size, bool)

    for role in sorted(set(roles[unique])):
        all_rows = unique[roles[unique] == role]
        rows = all_rows[counts[all_rows] > 0]
        prefix = f"hybrid/{role}"
        emit(prefix, "events", len(all_rows))
        emit(prefix, "empty_response_events", len(all_rows) - len(rows))
        if not len(rows):
            emit(prefix, "zero_groups/count", 0)
            continue
        before, after = base[rows], final[rows]
        for stage, values in (("base", before), ("final", after)):
            emit(prefix, f"advantage/{stage}_mean", mean(values))
            emit(prefix, f"advantage/{stage}_std", np.std(values) if len(values) else 0.)
            emit(prefix, f"advantage/{stage}_abs_mean", mean(np.abs(values)))
            emit(prefix, f"advantage/{stage}_zero_fraction", mean(np.abs(values) <= zero_tolerance))
        emit(prefix, "advantage/sign_flip_fraction", mean((before * after < 0) &
             (np.abs(before) > zero_tolerance) & (np.abs(after) > zero_tolerance)))
        emit(prefix, "advantage/scaling_changed_fraction", mean(np.abs(after - before) > zero_tolerance))
        emit(prefix, "advantage/scaling_abs_change_mean", mean(np.abs(after - before)))
        zero = np.abs(before) <= zero_tolerance
        emit(prefix, "advantage/base_zero_became_nonzero_count", np.count_nonzero(zero & (np.abs(after) > zero_tolerance)))
        groups = defaultdict(list)
        for row in rows:
            groups[tasks[row]].append(row)
        zero_groups = [indices for indices in groups.values() if np.all(np.abs(base[indices]) <= zero_tolerance)]
        retained = sum(np.all(np.abs(final[indices]) <= zero_tolerance) for indices in zero_groups)
        emit(prefix, "zero_groups/count", len(zero_groups))
        emit(prefix, "zero_groups/final_nonzero_count", len(zero_groups) - retained)
        if zero_groups:
            emit(prefix, "zero_groups/preserved_fraction", retained / len(zero_groups))

        if stage_two:
            candidates = rows[candidate[rows]]
            emit(prefix, "stage2/candidate_edges", len(candidates))
            if len(candidates):
                emit(prefix, "stage2/candidate_skip_fraction", mean(~selected[candidates]))
            emit(prefix, "stage2/coefficient_changed_fraction", mean(np.abs(cfinal[rows] - c1[rows]) > 1e-12))
            emit(prefix, "stage2/coefficient_abs_change_mean", mean(np.abs(cfinal[rows] - c1[rows])))
        if semantic:
            eligible_rows = rows[eligible[rows]]
            predicted_rows = rows[available[rows]]
            gated_rows = rows[gated[rows]]
            emit(prefix, "semantic/eligible_events", len(eligible_rows))
            emit(prefix, "semantic/available_events", len(predicted_rows))
            emit(prefix, "semantic/gated_events", len(gated_rows))
            emit(prefix, "semantic/eligible_fraction", mean(eligible[rows]))
            if len(eligible_rows):
                emit(prefix, "semantic/available_given_eligible_fraction", mean(available[eligible_rows]))
                emit(prefix, "semantic/eligible_weight_mean", mean(weights[eligible_rows]))
            if len(predicted_rows):
                emit(prefix, "semantic/gated_given_available_fraction", mean(gated[predicted_rows]))
                emit(prefix, "semantic/delta_std", np.std(delta[predicted_rows]))
                emit(prefix, "semantic/negative_delta_fraction", mean(delta[predicted_rows] < 0))
            if len(gated_rows):
                emit(prefix, "semantic/gated_cap_mean", mean(caps[gated_rows]))
            if zero.any():
                emit(prefix, "semantic/base_zero_gated_fraction", mean(gated[rows[zero]]))
            emit(prefix, "semantic/gated_without_prediction_count", np.count_nonzero(gated[rows] & ~available[rows]))
    return metrics
