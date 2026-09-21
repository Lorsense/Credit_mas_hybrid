#!/usr/bin/env bash
# Qwen3.5 dense/MoE, native padded HF training + SGLang 0.5.10.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SOLVER_MODEL=${SOLVER_MODEL:-Qwen/Qwen3.5-4B}
export VERIFIER_MODEL=${VERIFIER_MODEL:-Qwen/Qwen3.5-4B}
# Match pretrain_semantic_value_qwen35.sh; supply a newly trained matching head.
export VALUE_ENCODER=${VALUE_ENCODER:-Qwen/Qwen3.5-4B}
# Requalify the pretrained semantic predictor on the online policy's trajectories.
export VALUE_INIT_MODE=${VALUE_INIT_MODE:-candidate}
export USE_REMOVE_PADDING=False
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}
export USE_TORCH_COMPILE=False
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_ENFORCE_EAGER=True
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
case "${DRY_RUN:-False}" in
  True|true|1) ;;
  *) "${PYTHON:-python3}" "$SCRIPT_DIR/../../scripts/check_qwen35_runtime.py" --require-kernels ;;
esac
exec bash "$SCRIPT_DIR/run_math_hybrid.sh" "$@"
