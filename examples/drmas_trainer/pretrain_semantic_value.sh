#!/usr/bin/env bash
# Explicit offline fitting; the output is a semantic-only checkpoint.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$SCRIPT_DIR/../.."
: "${VALUE_INPUT:?Set VALUE_INPUT to a rollout JSONL file}"
: "${VALUE_OUTPUT:?Set VALUE_OUTPUT to the output semantic_value.pt path}"
args=(
  --input "$VALUE_INPUT" --output "$VALUE_OUTPUT"
  --config "${VALUE_CONFIG:-$SCRIPT_DIR/pretrain_semantic_value.yaml}"
  --model-path "${VALUE_ENCODER:-Qwen/Qwen3-4B}"
  --device "${VALUE_DEVICE:-cuda:0}" --rounds "${VALUE_ROUNDS:-1}"
  --epochs "${VALUE_EPOCHS:-5}" --seed "${VALUE_SEED:-0}"
)
if [[ -n "${VALUE_WARM_START:-}" ]]; then args+=(--warm-start "$VALUE_WARM_START"); fi
case "${DRY_RUN:-False}" in
  True|true|1) printf '%q ' "${PYTHON:-python3}" -m examples.drmas_trainer.pretrain_semantic_value "${args[@]}" "$@"; printf '\n' ;;
  *) exec "${PYTHON:-python3}" -m examples.drmas_trainer.pretrain_semantic_value "${args[@]}" "$@" ;;
esac
