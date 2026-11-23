#!/usr/bin/env bash
# Phase 2b (deviation D2): one OutfitTransformer stream (dataset x input encoder), so the 9 streams
# run in parallel on the GPU (sequential run_phase2b_gpu.sh would not fit the remaining booking).
# Commands and arguments are identical to run_full_gpu.sh / run_phase2b_gpu.sh.
# Usage: run_phase2b_ot.sh <dataset> <encoder>
set -uo pipefail
cd "$(dirname "$0")/../.."
d=${1:?dataset}; m=${2:?encoder}
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}; F=$R/v2/full; PY=.venv/bin/python
out=$F/outfit_transformer/$d/$m
step() {
  echo "=== $* $(date '+%F %T')"
  "$@" || { echo "retry $(date '+%F %T')"; "$@"; } || { echo "FAILED: $*" | tee -a $F/failed/GPU_STEPS; return 1; }
  echo "ok $(date '+%F %T')"
}
[ -f $out/OT_STREAM_DONE ] && { echo "already done $d $m"; exit 0; }
step $PY scripts/v2/run_outfit_transformer_v2.py train --protocol-root $R/v2/$d/build_a --dataset-name $d \
  --embeddings $F/emb/$d/$m --encoder $m --output-root $out --lrs 1e-4 3e-4 --seeds 42 43 44 45 46 \
  --max-epochs 100 --patience 10 --device cuda || exit 1
step $PY scripts/v2/run_outfit_transformer_v2.py select --output-root $out --min-seeds 5 || exit 1
step $PY scripts/v2/run_outfit_transformer_v2.py evaluate --protocol-root $R/v2/$d/build_a --dataset-name $d \
  --embeddings $F/emb/$d/$m --output-root $out --split test --confirm-protocol-frozen || exit 1
touch $out/OT_STREAM_DONE; echo "OT_STREAM_DONE $d $m $(date '+%F %T')"
