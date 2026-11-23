#!/usr/bin/env bash
# Phase 2b: main paired statistics with 20,000 bootstrap replicates (owner decision 2026-11-23).
# With 2,000 replicates the p-value floor (1/2001) x 105 Holm comparisons per block exceeds 0.05,
# so no pairwise test could reach significance. The protocol pre-registers "2,000+ replicates".
# Identical stats command to run_phase2b_cpu.sh plus --replicates 20000, written to a separate
# results_r20k/ (predictions linked read-only), leaving the 2,000-replicate results untouched.
set -uo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}; F=$R/v2/full; PY=.venv/bin/python
O=$F/results_r20k
mkdir -p $O
[ -e $O/predictions ] || ln -s $F/results/predictions $O/predictions
[ -f $O/manifest.json ] || cp $F/results/manifest.json $O/manifest.json
declare -A GRP=([polyvore_v2_clean]=component [iqon3000_v2_clean]=two_way [pog_v2_clean]=two_way)
spec() { echo "$1=$R/v2/$1/build_a=${GRP[$1]}"; }
specs="$(spec polyvore_v2_clean),$(spec iqon3000_v2_clean),$(spec pog_v2_clean)"
models=resnet50,fashionclip2,openclip_b32_laion2b,marqo_fashionclip,marqo_fashionsiglip,siglip_b16_224,siglip2_b16_384,dinov3_vitl16,gr_lite,dinov3_vitl16_336,clip_vitl14_336,pe_core_l14_336,siglip2_so400m_384,qwen3_vl_emb_2b,qwen3_vl_emb_8b
echo "=== stats r20k start $(date '+%F %T')"
$PY scripts/v2/pipeline.py --stage stats --datasets "$specs" --models $models --embeddings $F/emb \
  --results $O --eval-split test --frozen-protocol-hash configs/v2_protocol_hash.json --replicates 20000 \
  --equivalence polyvore_v2_clean=$R/v2/image_equivalence/polyvore/equivalence_groups.json \
  iqon3000_v2_clean=$R/v2/image_equivalence/iqon/equivalence_groups.json pog_v2_clean=$R/v2/image_equivalence/pog/equivalence_groups.json
rc=$?; echo "=== stats r20k end rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && touch $O/STATS_DONE
cmp -s $F/results/selection.json $O/selection.json && echo "selection.json identical to 2,000-replicate run" || echo "WARNING: selection.json differs"
exit $rc
