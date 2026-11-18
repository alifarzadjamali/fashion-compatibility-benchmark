#!/usr/bin/env bash
# G1 followed by G2 only. Later gates require verified data and original v1 evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
attempt="${1:?Usage: bash scripts/v2/run_readiness.sh UNIQUE_ATTEMPT_NAME}"
if [[ ! "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'Attempt name must contain only letters, numbers, underscores or hyphens' >&2
    exit 2
fi
export REPBENCH_DATA_ROOT="${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}"
export REPBENCH_CACHE_ROOT="${REPBENCH_CACHE_ROOT:-/data/$USER/repbench/cache}"
export HF_HOME="${HF_HOME:-$REPBENCH_CACHE_ROOT/huggingface}"
if [[ -z "${HF_TOKEN_PATH:-}" && -f "/data/$USER/.cache/huggingface/token" ]]; then
    export HF_TOKEN_PATH="/data/$USER/.cache/huggingface/token"
fi
export TORCH_HOME="${TORCH_HOME:-$REPBENCH_CACHE_ROOT/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPBENCH_CACHE_ROOT/uv}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20260912
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
out="research/v2/readiness/$attempt"
mkdir -p "$out"
if [[ -e "$out/g1" ]]; then
    echo 'Existing attempt: choose a fresh name; caches will be reused' >&2
    exit 2
fi
timeleft
time bash scripts/v2/bootstrap.sh --output "$out/g1"
time uv run --frozen python scripts/v2/g2_models.py --g1 "$out/g1/gate.json" --output "$out/g2"
