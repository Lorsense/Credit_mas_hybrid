#!/usr/bin/env bash
# Frozen Qwen3.5 text encoder + a fresh semantic head, matching online training.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VALUE_ENCODER=${VALUE_ENCODER:-Qwen/Qwen3.5-4B}
# The shared launcher passes VALUE_ENCODER as --model-path, overriding the
# base YAML model name while keeping dtype, head, and feature settings aligned.
exec bash "$SCRIPT_DIR/pretrain_semantic_value.sh" "$@"
