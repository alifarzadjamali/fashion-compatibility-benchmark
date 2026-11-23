#!/usr/bin/env bash
# Phase 2b: profiling after all 9 OutfitTransformer streams finish; same commands as
# run_phase2b_gpu.sh (CPU-lane processes paused with SIGSTOP for isolated timings, then resumed).
# Usage: run_phase2b_profile.sh <g2 gate.json>
set -uo pipefail
cd "$(dirname "$0")/../.."
g2=${1:?G2 gate.json}
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}; F=$R/v2/full; PY=.venv/bin/python
BASE="resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224
 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384"
CPU_PAT='scripts/v2/(run_phase2b_(cpu|sens|nd|lookbench)\.sh|pipeline\.py|run_sensitivities\.py|near_duplicate_audit\.py|a100_v2\.py|make_tables_v2\.py|extract_iqon_images\.py)'
streams="polyvore_v2_clean/resnet50 polyvore_v2_clean/siglip2_so400m_384 polyvore_v2_clean/marqo_fashionclip
 iqon3000_v2_clean/resnet50 iqon3000_v2_clean/siglip2_so400m_384 iqon3000_v2_clean/marqo_fashionsiglip
 pog_v2_clean/resnet50 pog_v2_clean/siglip2_so400m_384 pog_v2_clean/marqo_fashionsiglip"
for s in $streams; do
  until [ -f $F/outfit_transformer/$s/OT_STREAM_DONE ] || grep -q "outfit_transformer/$s " $F/failed/GPU_STEPS 2>/dev/null; do sleep 120; done
done
touch $F/OT_DONE; echo "OUTFIT_TRANSFORMER_DONE $(date '+%F %T')"
until [ -f $R/raw/polyvore_outfits/IMAGES_READY ]; do sleep 120; done
paused=$(pgrep -f -d ' ' "$CPU_PAT" || true)
resume() { [ -n "$paused" ] && kill -CONT $paused 2>/dev/null; echo "resumed CPU lane: ${paused:-none} $(date '+%F %T')"; }
trap resume EXIT
[ -n "$paused" ] && kill -STOP $paused && echo "paused CPU lane for profiling: $paused $(date '+%F %T')"
sleep 10
for m in $BASE qwen3_vl_emb_2b qwen3_vl_emb_8b; do
  echo "=== profile $m $(date '+%F %T')"
  $PY scripts/v2/profile_efficiency_v2.py --model $m --images-dir $R/raw/polyvore_outfits/images \
    --n-images 64 --energy-seconds 30 --output $F/profile/$m.json --g2 "$g2" \
    || $PY scripts/v2/profile_efficiency_v2.py --model $m --images-dir $R/raw/polyvore_outfits/images \
    --n-images 64 --energy-seconds 30 --output $F/profile/$m.json --g2 "$g2" \
    || echo "FAILED: profile $m" | tee -a $F/failed/GPU_STEPS
done
resume; trap - EXIT
touch $F/PROFILE_DONE; echo GPU_LANE_DONE
