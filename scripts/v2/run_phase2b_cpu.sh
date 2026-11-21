#!/usr/bin/env bash
# Phase 2b (restore on Slurm job 37 after the 2026-11-20 booking loss, deviation D2).
# Identical to the probe + stats part of run_full_cpu.sh (same commands, same arguments);
# only WORKERS defaults to 6. Probe cells already present (restored DGX predictions) are
# skipped by pipeline.py; missing cells are fitted. Ends after stats (selection.json),
# so the GPU lane can start OutfitTransformer; sensitivities run separately.
set -uo pipefail
cd "$(dirname "$0")/../.."
freeze=configs/v2_protocol_hash.json
[ -f "$freeze" ] || { echo "Not frozen (no $freeze); refusing Phase 2"; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
F=$R/v2/full
PY=.venv/bin/python
WORKERS=${WORKERS:-6}
mkdir -p $F/results $F/done
declare -A GRP=([polyvore_v2_clean]=component [iqon3000_v2_clean]=two_way [pog_v2_clean]=two_way)
MODELS="resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224
 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384
 qwen3_vl_emb_2b qwen3_vl_emb_8b"
complete() {  # dataset model
  local s=$F/emb/$1/$2
  [ -f $s/all/metadata.json ] || { [ -f $s/trainvalid/metadata.json ] && [ -f $s/test/metadata.json ]; }
}
spec() { echo "$1=$R/v2/$1/build_a=${GRP[$1]}"; }
EQ="--equivalence polyvore_v2_clean=$R/v2/image_equivalence/polyvore/equivalence_groups.json \
 iqon3000_v2_clean=$R/v2/image_equivalence/iqon/equivalence_groups.json pog_v2_clean=$R/v2/image_equivalence/pog/equivalence_groups.json"
mkdir -p $F/failed
remaining=1
while [ $remaining -gt 0 ]; do
  remaining=0
  for d in iqon3000_v2_clean pog_v2_clean polyvore_v2_clean; do for m in $MODELS; do
    [ -f $F/done/$d.$m ] && continue
    [ -f $F/failed/$d.$m ] && continue   # extraction or probes failed: reported, not retried forever
    remaining=$((remaining + 1))
    complete $d $m || continue
    echo "=== probes $d $m $(date '+%F %T')"
    if $PY scripts/v2/pipeline.py --stage probes --datasets "$(spec $d)" --models $m \
      --embeddings $F/emb --results $F/results --eval-split test --frozen-protocol-hash $freeze \
      --workers $WORKERS $EQ; then touch $F/done/$d.$m; else
      n=$(( $(cat $F/failed/$d.$m.tries 2>/dev/null || echo 0) + 1 )); echo $n > $F/failed/$d.$m.tries
      [ $n -ge 3 ] && { echo "probes failed 3x: $d $m" > $F/failed/$d.$m; echo "FAILED probes $d $m"; }
    fi
    echo "probes done $d $m $(date '+%F %T')"
  done; done
  [ $remaining -gt 0 ] && sleep 120
done
all=$(echo $MODELS | tr ' ' ',')
specs="$(spec polyvore_v2_clean),$(spec iqon3000_v2_clean),$(spec pog_v2_clean)"
echo "=== stats $(date '+%F %T')"
ok_models=$(for m in $MODELS; do ls $F/failed/*.$m >/dev/null 2>&1 || echo $m; done | tr '\n' ',' | sed 's/,$//')
[ "$ok_models" != "$all" ] && echo "WARNING: stats without failed models; see $F/failed/"
if ! $PY scripts/v2/pipeline.py --stage stats --datasets "$specs" --models $ok_models --embeddings $F/emb \
  --results $F/results --eval-split test --frozen-protocol-hash $freeze $EQ; then
  echo "stats failed" > $F/STATS_FAILED; echo "FAILED stats"
fi
touch $F/STATS_DONE; echo PHASE2B_CPU_STATS_DONE
