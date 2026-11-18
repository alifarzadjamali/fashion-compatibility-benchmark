#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
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
mkdir -p "$REPBENCH_DATA_ROOT" "$REPBENCH_CACHE_ROOT"
uv sync --locked --extra dev --python 3.12
exec uv run --frozen python scripts/v2/g1_environment.py "$@"
