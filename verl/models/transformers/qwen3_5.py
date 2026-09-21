"""Native Hugging Face Qwen3.5 text-training integration.

The official dense and MoE checkpoints retain their full multimodal module
tree. No decoder forward is replaced and no token sequences are concatenated.
The visual tower is frozen, while text parameters keep FP32 master weights.
"""
from __future__ import annotations


QWEN3_5_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
QWEN3_5_TEXT_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})


def get_text_config(config):
    return getattr(config, "text_config", config)


def is_qwen3_5(config) -> bool:
    return getattr(config, "model_type", None) in QWEN3_5_MODEL_TYPES | QWEN3_5_TEXT_MODEL_TYPES


def select_auto_model_class(config):
    """Select the current multimodal auto class without importing removed APIs.

    Vision2Seq is only a compatibility fallback for older Transformers/Qwen3
    environments. Official Qwen3.5 checkpoints require ImageTextToText.
    """
    import transformers

    if is_qwen3_5(config):
        if getattr(config, "model_type", None) not in QWEN3_5_MODEL_TYPES or not hasattr(config, "text_config"):
            raise ValueError("Qwen3.5 training requires the complete official ConditionalGeneration checkpoint with text_config")
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None or type(config) not in model_class._model_mapping.keys():
            raise ValueError("This Qwen3.5 checkpoint requires the supported Transformers 5.3.0 runtime")
        return model_class
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None and type(config) in model_class._model_mapping.keys():
            return model_class
    return transformers.AutoModelForCausalLM


def validate_qwen3_5_execution(config, *, use_remove_padding=False, ulysses_sp_size=1,
                              use_fused_kernels=False, use_liger=False, lora_rank=0):
    """Reject legacy packed/attention-only patches before model allocation."""
    if not is_qwen3_5(config):
        return
    if getattr(config, "model_type", None) not in QWEN3_5_MODEL_TYPES or not hasattr(config, "text_config"):
        raise ValueError("Qwen3.5 requires a complete ConditionalGeneration checkpoint with text_config")
    if int(lora_rank) != 0:
        raise ValueError("Qwen3.5 currently supports full fine-tuning only; lora_rank must be 0")
    if getattr(config, "quantization_config", None) or getattr(config.text_config, "quantization_config", None):
        raise ValueError("Qwen3.5 requires an official non-quantized checkpoint; quantized Actor weights are unsupported")
    if use_remove_padding:
        raise ValueError("Qwen3.5 requires use_remove_padding=False; packed sequences share recurrent state")
    if int(ulysses_sp_size) != 1:
        raise ValueError("Qwen3.5 requires ulysses_sequence_parallel_size=1; legacy attention-only SP is unsupported")
    if use_fused_kernels or use_liger:
        raise ValueError("Qwen3.5 requires native HF forward: use_fused_kernels=False and use_liger=False")


def validate_qwen3_5_loading_info(loading_info):
    """A full checkpoint must not silently initialize missing text/vision weights.

    HF has already applied its model-specific ignore rules (e.g. optional MTP
    tensors) when returning this report; unexplained discrepancies fail closed.
    """
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise ValueError(f"Qwen3.5 checkpoint did not load completely: {key}={loading_info[key]}")


def configure_qwen3_5_native(config, *, attn_implementation="sdpa"):
    """Configure explicit native attention and stateless full-sequence training."""
    if not is_qwen3_5(config):
        return
    validate_qwen3_5_execution(config)
    if attn_implementation not in ("sdpa", "eager"):
        raise ValueError("Qwen3.5 native training currently supports attn_implementation=sdpa or eager")
    config._attn_implementation = attn_implementation
    config.text_config._attn_implementation = attn_implementation
    config.vision_config._attn_implementation = attn_implementation
    config.use_cache = False
    config.text_config.use_cache = False
    # PPO supplies its own loss and does not train an auxiliary router loss.
    config.text_config.output_router_logits = False


def _mask_padding_states(hidden_states, attention_mask):
    """HF 5.3's linear-attention helper incorrectly skips one-row batches."""
    if attention_mask is None:
        return hidden_states
    if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Qwen3.5 linear-attention padding mask must match [batch, sequence]")
    return hidden_states * attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)


def install_qwen3_5_padding_fix(config):
    """Fix only dense/MoE padding masking, leaving native HF forwards intact.

    In Transformers 5.3.0 the original helper requires batch > 1. Therefore a
    one-action PPO microbatch can feed left-padding embeddings into recurrent
    state. Semantic scoring uses this same helper for consistent prefix values.
    """
    if not is_qwen3_5(config):
        return
    import importlib

    family = "qwen3_5_moe" if "moe" in config.model_type else "qwen3_5"
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    if not hasattr(module, "apply_mask_to_padding_states"):
        raise ValueError("Qwen3.5 padding adapter requires the supported Transformers 5.3.0 model implementation")
    module.apply_mask_to_padding_states = _mask_padding_states


def prepare_qwen3_5_for_text_training(model):
    """Keep official names, freeze vision, and retain sensitive state in FP32.

    FSDP must likewise use FP32 parameter/reduction/buffer dtypes. Casting only
    A_log back to float after a BF16 FSDP gather would already lose precision.
    The Actor's existing autocast still uses BF16 for eligible operations.
    """
    if not is_qwen3_5(model.config):
        return
    install_qwen3_5_padding_fix(model.config)
    visual = getattr(getattr(model, "model", None), "visual", None)
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if visual is None or language_model is None:
        raise ValueError("Qwen3.5 must retain model.visual and model.language_model from the complete checkpoint")
    model.float()
    visual.requires_grad_(False)
    visual.eval()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    # Use the classes actually present; this avoids stale TextDecoderLayer
    # aliases in intermediate HF base-model _no_split_modules declarations.
    decoder_types = {layer.__class__.__name__ for layer in language_model.layers}
    if not decoder_types:
        raise ValueError("Qwen3.5 text decoder has no layers to shard")
    # Wrap individual visual blocks, not their parent tower: the generic FSDP2
    # helper separately shards Embeddings (including visual.pos_embed), so
    # wrapping the whole tower first would violate child-before-parent order.
    vision_block_types = {block.__class__.__name__ for block in visual.blocks}
    model._no_split_modules = sorted(decoder_types | vision_block_types)
