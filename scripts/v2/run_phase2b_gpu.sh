#!/usr/bin/env bash
# Phase 2b GPU lane (restore on Slurm job 37 after the 2026-11-20 booking loss, deviation D2).
# Extraction and MLLM scoring finished before the loss (restored from verified copies), so
# this runs only the remaining GPU steps of run_full_gpu.sh with identical commands:
# OutfitTransformer (after the validation-only selection) -> profiling. For isolated
# timings, CPU-lane processes are paused (SIGSTOP) during profiling and resumed after.
# Usage: run_phase2b_gpu.sh <g2 gate.json>
set -uo pipefail
cd "$(dirname "$0")/../.."
g2=${1:?G2 gate.json bound to the frozen code}
freeze=configs/v2_protocol_hash.json
[ -f "$freeze" ] || { echo "Not frozen (no $freeze); refusing Phase 2"; exit 2; }
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
F=$R/v2/full
PY=.venv/bin/python
STOP=/data/$USER/repbench/STOP_GPU
DATASETS="iqon3000_v2_clean pog_v2_clean polyvore_v2_clean"
BASE="resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224
 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384"
CPU_PAT='scripts/v2/(run_phase2b_(cpu|sens|nd)\.sh|pipeline\.py|run_sensitivities\.py|near_duplicate_audit\.py|a100_v2\.py|make_tables_v2\.py|extract_iqon_images\.py)'
mkdir -p $F/failed
step() {  # run once, retry once; on second failure record it
  [ -f "$STOP" ] && { echo "STOP requested"; exit 0; }
  echo "=== $* $(date '+%F %T')"
  "$@" || { echo "retry $(date '+%F %T')"; "$@"; } || { echo "FAILED: $*" | tee -a $F/failed/GPU_STEPS; return 1; }
  echo "ok $(date '+%F %T')"
}
until [ -f $F/results/selection.json ] || [ -f $F/STATS_FAILED ]; do sleep 120; done
for d in $DATASETS; do
  inputs=$($PY -c "
import json,os
p='$F/results/selection.json'; s=json.load(open(p)).get('$d', {}) if os.path.exists(p) else {}
picked=[s.get('generic'), s.get('fashion')]
print(' '.join(dict.fromkeys(['resnet50']+[m for m in picked if m])))")
  [ "$inputs" = resnet50 ] && echo "WARNING: selection missing for $d; OutfitTransformer on resnet50 only" | tee -a $F/failed/GPU_STEPS
  for m in $inputs; do
    out=$F/outfit_transformer/$d/$m
    step $PY scripts/v2/run_outfit_transformer_v2.py train --protocol-root $R/v2/$d/build_a --dataset-name $d \
      --embeddings $F/emb/$d/$m --encoder $m --output-root $out --lrs 1e-4 3e-4 --seeds 42 43 44 45 46 \
      --max-epochs 100 --patience 10 --device cuda
    step $PY scripts/v2/run_outfit_transformer_v2.py select --output-root $out --min-seeds 5
    step $PY scripts/v2/run_outfit_transformer_v2.py evaluate --protocol-root $R/v2/$d/build_a --dataset-name $d \
      --embeddings $F/emb/$d/$m --output-root $out --split test --confirm-protocol-frozen
  done
done
echo OUTFIT_TRANSFORMER_DONE
touch $F/OT_DONE
# Profiling needs the Polyvore images (unpacked from the verified archive).
until [ -f $R/raw/polyvore_outfits/IMAGES_READY ]; do sleep 120; done
paused=$(pgrep -f -d ' ' "$CPU_PAT" || true)
resume() { [ -n "$paused" ] && kill -CONT $paused 2>/dev/null; echo "resumed CPU lane: ${paused:-none} $(date '+%F %T')"; }
trap resume EXIT
[ -n "$paused" ] && kill -STOP $paused && echo "paused CPU lane for profiling: $paused $(date '+%F %T')"
sleep 10
for m in $BASE qwen3_vl_emb_2b qwen3_vl_emb_8b; do
  step $PY scripts/v2/profile_efficiency_v2.py --model $m --images-dir $R/raw/polyvore_outfits/images \
    --n-images 64 --energy-seconds 30 --output $F/profile/$m.json --g2 "$g2"
done
resume; trap - EXIT
touch $F/PROFILE_DONE
echo GPU_LANE_DONE
