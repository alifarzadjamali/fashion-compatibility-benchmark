#!/usr/bin/env bash
# Phase 2 GPU lane (runs only after the G7 freeze). Strict order, no idle gaps:
# test-split extraction (13 encoders) -> Qwen-2B -> Qwen-8B -> zero-shot MLLM ->
# OutfitTransformer (needs the CPU lane's validation-only selection) -> profiling
# (waits for the CPU lane to finish so timings are isolated). Resumable: completed
# stores/records are verified and skipped. Usage: run_full_gpu.sh <g2 gate.json>
set -uo pipefail
cd "$(dirname "$0")/../.."
g2=${1:?G2 gate.json bound to the frozen code}
freeze=configs/v2_protocol_hash.json
[ -f "$freeze" ] || { echo "Not frozen (no $freeze); refusing Phase 2"; exit 2; }
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/data/$USER/.cache/huggingface/token} TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
F=$R/v2/full
LOG=${LOG:-/data/$USER/repbench/logs}
PY=.venv/bin/python
STOP=/data/$USER/repbench/STOP_GPU
DATASETS="iqon3000_v2_clean pog_v2_clean polyvore_v2_clean"
declare -A IMG=([iqon3000_v2_clean]=$R/iqon3000_clean/images [pog_v2_clean]=$R/raw/pog/mirror
                [polyvore_v2_clean]=$R/raw/polyvore_outfits/images)
BASE="resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224
 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384"
step() { [ -f "$STOP" ] && { echo "STOP requested"; exit 0; }; echo "=== $* $(date '+%F %T')"; "$@"; echo "rc=$? $(date '+%F %T')"; }

# Pre-freeze train/valid extraction must have finished.
until grep -q TV_CHAIN_DONE "$LOG/full-tv.log" 2>/dev/null; do sleep 60; done
for d in $DATASETS; do for m in $BASE; do
  step $PY scripts/v2/extract_v2.py --g2 "$g2" --dataset $d --protocol $R/v2/$d/build_a \
    --images-root ${IMG[$d]} --model $m --splits test --output $F/emb/$d/$m/test
done; done
for m in qwen3_vl_emb_2b qwen3_vl_emb_8b; do for d in $DATASETS; do
  step $PY scripts/v2/extract_v2.py --g2 "$g2" --dataset $d --protocol $R/v2/$d/build_a \
    --images-root ${IMG[$d]} --model $m --splits train,valid,test --output $F/emb/$d/$m/all
done; done
echo EXTRACTION_DONE
for d in $DATASETS; do
  step $PY scripts/v2/mllm_zeroshot.py score --ids research/v2/freeze/mllm_ids_$d.json \
    --protocol-root $R/v2/$d/build_a --images-root ${IMG[$d]} --output-dir $F/mllm/$d --confirm-protocol-frozen
done
if [ -f research/v2/freeze/mllm_ids_a100.json ]; then
  step $PY scripts/v2/mllm_zeroshot.py score --ids research/v2/freeze/mllm_ids_a100.json \
    --a100-root $R/v2/a100 --images-root $R/v2/a100/images --output-dir $F/mllm/a100 --confirm-protocol-frozen
fi
# OutfitTransformer needs the validation-only selection written by the CPU lane.
until [ -f $F/results/selection.json ]; do sleep 120; done
for d in $DATASETS; do
  inputs=$($PY -c "import json,sys;s=json.load(open('$F/results/selection.json'))['$d'];print(' '.join(dict.fromkeys(['resnet50',s['generic'],s['fashion']])))")
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
# Isolated profiling: wait for the CPU lane so nothing else competes.
until [ -f $F/CPU_LANE_DONE ]; do sleep 120; done
for m in $BASE qwen3_vl_emb_2b qwen3_vl_emb_8b; do
  step $PY scripts/v2/profile_efficiency_v2.py --model $m --images-dir $R/raw/polyvore_outfits/images \
    --n-images 64 --energy-seconds 30 --output $F/profile/$m.json --g2 "$g2"
done
echo GPU_LANE_DONE
