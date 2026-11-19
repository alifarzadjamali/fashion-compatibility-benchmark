#!/usr/bin/env bash
# Pre-freeze development extraction (TRAIN+VALID only; extract_v2 refuses test before
# the freeze) for the 13 fixed-resolution encoders on the three v2-Clean protocols.
# Writes $F/emb/<dataset>/<model>/trainvalid and ends with TV_CHAIN_DONE in the log,
# which run_full_gpu.sh waits for. Polyvore reuses verified pre-Decision-E stores for
# identical JPEG items (decoding unchanged). STOP_FULL_TV stops between encoders.
# Usage: run_full_trainvalid.sh <g2 gate.json>   (log: $LOG/full-tv.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
g2=${1:?G2 gate.json}
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/data/$USER/.cache/huggingface/token} TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
F=$R/v2/full
declare -A IMG=([iqon3000_v2_clean]=$R/iqon3000_clean/images [pog_v2_clean]=$R/raw/pog/mirror
                [polyvore_v2_clean]=$R/raw/polyvore_outfits/images)
for d in iqon3000_v2_clean pog_v2_clean polyvore_v2_clean; do
 for m in resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224 \
          siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384; do
  [ -f /data/$USER/repbench/STOP_FULL_TV ] && { echo "STOP requested"; exit 0; }
  reuse=""; [ $d = polyvore_v2_clean ] && [ -d $R/v2/dev/emb/polyvore_d_clean/$m ] && reuse="--reuse-from $R/v2/dev/emb/polyvore_d_clean/$m"
  .venv/bin/python scripts/v2/extract_v2.py --g2 "$g2" --dataset $d --protocol $R/v2/$d/build_a --images-root ${IMG[$d]} \
    --model $m --output $F/emb/$d/$m/trainvalid $reuse 2>&1 | grep -v -i warn | tail -1
  echo "DONE $d $m rc=${PIPESTATUS[0]} $(date +%T)"
 done
done
echo TV_CHAIN_DONE
