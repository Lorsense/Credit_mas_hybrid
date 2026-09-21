"""Checked Qwen3.5 weight updates for the pinned SGLang 0.5.10 runtime.

Keep HF tensor names intact: SGLang's model loader owns TP slicing, fused
projections and the stacked MoE expert layout.  This module validates that
loader's destinations so its warning-and-skip paths cannot hide missing weights.
"""

from collections.abc import Mapping
import re

import torch


STRICT_QWEN35_LOADER = "verl.utils.sglang_weight_sync.strict_load_weights"
DEFAULT_WEIGHT_BUCKET_BYTES = 128 * 1024 * 1024


def is_qwen35_config(config):
    return getattr(config, "model_type", "") in {
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"
    }


def validate_local_tp_topology(topology):
    """All entries are (hostname, global ranks of that rank's inference TP group)."""
    for _, ranks in topology:
        if len({topology[rank][0] for rank in ranks}) > 1:
            raise ValueError("SGLang tensor weight updates require each rollout TP group to fit on one node")


def qwen35_sglang_target_name(name):
    """The non-quantized HF -> SGLang 0.5.10 destination (not a tensor rewrite)."""
    name = name.replace("model.language_model.", "model.").replace(".self_attn.", ".")
    if "visual" in name:
        return name.replace("model.visual.", "visual.").replace("attn.qkv.", "attn.qkv_proj.")
    if name.endswith(".experts.gate_up_proj"):
        return name.removesuffix("gate_up_proj") + "w13_weight"
    if name.endswith(".experts.down_proj"):
        return name.removesuffix("down_proj") + "w2_weight"
    expert = re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name)
    if expert:
        destination = "w2_weight" if expert.group(1) == "down_proj" else "w13_weight"
        return name[:expert.start()] + ".experts." + destination
    for source, destination in (
        (".q_proj.", ".qkv_proj."), (".k_proj.", ".qkv_proj."), (".v_proj.", ".qkv_proj."),
        (".gate_proj.", ".gate_up_proj."), (".up_proj.", ".gate_up_proj."),
        (".in_proj_qkv.", ".in_proj_qkvz."), (".in_proj_z.", ".in_proj_qkvz."),
        (".in_proj_b.", ".in_proj_ba."), (".in_proj_a.", ".in_proj_ba."),
    ):
        if source in name:
            return name.replace(source, destination)
    return name


def _validate_stacked_expert_shape(name, tensor, config):
    if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
        return
    config = getattr(config, "text_config", config)
    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.moe_intermediate_size
    expected = ((experts, 2 * intermediate, hidden) if name.endswith("gate_up_proj")
                else (experts, hidden, intermediate))
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Invalid stacked Qwen3.5 expert weight {name}: {tuple(tensor.shape)} != {expected}")


def strict_load_weights(model, named_tensors):
    """SGLang custom loader: preflight all keys, then verify native loader results.

    Registered through ServerArgs.custom_weight_loader and invoked by
    ModelRunner.update_weights_from_tensor. This is intentionally limited to
    Qwen3.5's non-quantized, TP-only model loader (no pipeline/expert parallelism).
    Reference: sglang v0.5.10, srt/models/qwen3_5.py, load_weights methods.
    """
    if model.__class__.__name__ not in {
        "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM",
    }:
        raise TypeError(f"Qwen3.5 strict loader received {type(model).__name__}")
    named_tensors = list(named_tensors)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    expected = set()
    seen = set()
    for name, tensor in named_tensors:
        if name in seen:
            raise ValueError(f"Duplicate weight in one SGLang update: {name}")
        seen.add(name)
        # These deterministic rotary buffers are recomputed by the model.
        if name.endswith("rotary_emb.inv_freq"):
            continue
        target = qwen35_sglang_target_name(name)
        if target not in parameters:
            raise KeyError(f"SGLang has no destination for HF weight {name!r} (mapped to {target!r})")
        _validate_stacked_expert_shape(name, tensor, model.config)
        expected.add(target)
    loaded = model.load_weights(named_tensors)
    if not isinstance(loaded, (set, frozenset, list, tuple)):
        raise RuntimeError("SGLang Qwen3.5 loader did not report its loaded parameter names")
    missing = expected.difference(loaded)
    if missing:
        raise RuntimeError(f"SGLang skipped required Qwen3.5 weights: {sorted(missing)}")
    return loaded


def iter_weight_buckets(named_tensors, *, max_bytes=DEFAULT_WEIGHT_BUCKET_BYTES, materialize=lambda tensor: tensor):
    """Bound materialized full tensors; an indivisible large parameter is alone.

    Bucket membership uses global shapes before DTensor all-gather, so every
    FSDP rank enters the same collectives. Only the current bucket is gathered.
    """
    if max_bytes <= 0:
        raise ValueError("weight update bucket size must be positive")
    bucket, size = [], 0
    for name, tensor in named_tensors:
        nbytes = tensor.numel() * tensor.element_size()
        if bucket and size + nbytes > max_bytes:
            yield bucket
            bucket, size = [], 0
        tensor = materialize(tensor)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Weight {name} was not materialized as a torch.Tensor")
        bucket.append((name, tensor.detach().contiguous()))
        size += nbytes
        if size >= max_bytes:
            yield bucket
            bucket, size = [], 0
    if bucket:
        yield bucket


def require_engine_success(result, operation):
    """Accept documented bool/dict/tuple responses, reject missing acknowledgments."""
    if isinstance(result, bool):
        success = result
    elif isinstance(result, Mapping):
        success = result.get("success")
    elif isinstance(result, (tuple, list)) and result and isinstance(result[0], bool):
        success = result[0]
    else:
        success = getattr(result, "success", None)
    if success is not True:
        raise RuntimeError(f"SGLang {operation} failed or was not acknowledged: {result!r}")


def update_weight_bucket(engine, bucket, *, load_format=None):
    # Engine's public API performs CUDA IPC serialization itself. Passing the
    # removed ModelRunner.LocalSerializedTensor wrapper serializes the wrong
    # protocol on modern SGLang.
    result = engine.update_weights_from_tensor(named_tensors=bucket, load_format=load_format, flush_cache=False)
    require_engine_success(result, "weight update")


def flush_weight_caches(engine):
    # In v0.5.10 Scheduler.flush_cache resets tree_cache AND req_to_token_pool.
    # MambaRadixCache.reset + HybridReqToTokenPool.clear invalidate recurrent
    # state as well as ordinary KV/prefix state. A busy server returns False.
    require_engine_success(engine.flush_cache(), "cache flush after weight update")
