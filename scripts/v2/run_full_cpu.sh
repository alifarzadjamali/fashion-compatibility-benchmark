#!/usr/bin/env bash
# Phase 2 CPU lane (only after the G7 freeze). As soon as an encoder's stores for a
# dataset are complete (trainvalid+test, or all/ for Qwen), fit all its probe cells
# (fit TRAIN, select/calibrate VALID, evaluate TEST). When all 15 x 3 are done: paired
# statistics + validation-only selection (unblocks the GPU lane's OutfitTransformer),
# sensitivities, A100, tables, number check. Usage: run_full_cpu.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
freeze=configs/v2_protocol_hash.json
[ -f "$freeze" ] || { echo "Not frozen (no $freeze); refusing Phase 2"; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
F=$R/v2/full
PY=.venv/bin/python
WORKERS=${WORKERS:-3}
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
remaining=1
while [ $remaining -gt 0 ]; do
  remaining=0
  for d in iqon3000_v2_clean pog_v2_clean polyvore_v2_clean; do for m in $MODELS; do
    [ -f $F/done/$d.$m ] && continue
    remaining=$((remaining + 1))
    complete $d $m || continue
    echo "=== probes $d $m $(date '+%F %T')"
    $PY scripts/v2/pipeline.py --stage probes --datasets "$(spec $d)" --models $m \
      --embeddings $F/emb --results $F/results --eval-split test --frozen-protocol-hash $freeze \
      --workers $WORKERS && touch $F/done/$d.$m
    echo "rc=$? $(date '+%F %T')"
  done; done
  [ $remaining -gt 0 ] && sleep 120
done
all=$(echo $MODELS | tr ' ' ',')
specs="$(spec polyvore_v2_clean),$(spec iqon3000_v2_clean),$(spec pog_v2_clean)"
echo "=== stats $(date '+%F %T')"
$PY scripts/v2/pipeline.py --stage stats --datasets "$specs" --models $all --embeddings $F/emb \
  --results $F/results --eval-split test --frozen-protocol-hash $freeze; echo "rc=$?"
# Sensitivities and A100 (scripts from the analysis modules; skipped if absent).
[ -f scripts/v2/run_sensitivities.py ] && { $PY scripts/v2/run_sensitivities.py all --results $F/results \
  --embeddings $F/emb --data-root $R --frozen-protocol-hash $freeze; echo "sensitivities rc=$?"; }
[ -f scripts/v2/a100_v2.py ] && { $PY scripts/v2/a100_v2.py score --embeddings $F/emb --data-root $R \
  --models $all --output $F/a100 && $PY scripts/v2/a100_v2.py summarize --input $F/a100 --output $F/results/a100; echo "a100 rc=$?"; }
$PY scripts/v2/make_tables_v2.py --results $F/results --out $F/generated; echo "tables rc=$?"
$PY scripts/v2/check_paper_numbers.py --generated $F/generated; echo "numbers rc=$?"
touch $F/CPU_LANE_DONE
echo CPU_LANE_DONE
