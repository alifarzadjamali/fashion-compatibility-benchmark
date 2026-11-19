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
# Near-duplicate audit of each v2-Clean protocol (pHash + embedding cosine >= 0.95 with
# DINOv3-L, pre-specified), then all sensitivities (label fractions, construction seeds,
# provenance, subgroups/failure consensus, near-duplicate exclusion, LookBench).
declare -A IMG=([iqon3000_v2_clean]=$R/iqon3000_clean/images [pog_v2_clean]=$R/raw/pog/mirror
                [polyvore_v2_clean]=$R/raw/polyvore_outfits/images)
affected=""
for d in polyvore_v2_clean iqon3000_v2_clean pog_v2_clean; do
  [ -f $F/near_duplicates/$d/audit.json ] || $PY scripts/v2/near_duplicate_audit.py --protocol-root $R/v2/$d/build_a \
    --dataset-name $d --images-root ${IMG[$d]} --embeddings $F/emb/$d/dinov3_vitl16 --cosine 0.95 \
    --output-dir $F/near_duplicates/$d --workers 8
  affected="$affected --affected $d=$F/near_duplicates/$d/affected_test_outfits.json"
done
$PY scripts/v2/run_sensitivities.py all --results $F/results --embeddings $F/emb --eval-split test \
  --frozen-protocol-hash $freeze --dataset-config configs/v2_sensitivity_datasets.json \
  --selection-split valid --lookbench-scores configs/lookbench_v3_scores.json $affected $EQ
echo "sensitivities rc=$?"
# Wait for the GPU lane's A100 work (finished or recorded as failed — never forever).
until { [ -f $F/emb/a100/qwen3_vl_emb_8b/test/metadata.json ] || [ -f $F/failed/a100.qwen3_vl_emb_8b ]; } && \
      { [ -f $F/mllm/a100_aat/summary.json ] || grep -q "a100_aat" $F/failed/GPU_STEPS 2>/dev/null; }; do sleep 120; done
a100_models=$(for m in ${ok_models//,/ }; do [ -f $F/failed/a100.$m ] || echo $m; done | tr '\n' ',' | sed 's/,$//')
corpora="polyvore_v2_clean=$R/v2/polyvore_v2_clean/build_a,iqon3000_v2_clean=$R/v2/iqon3000_v2_clean/build_a,pog_v2_clean=$R/v2/pog_v2_clean/build_a"
$PY scripts/v2/a100_v2.py score --a100 $R/v2/a100_v2 --embeddings $F/emb --corpora "$corpora" \
  --models $a100_models --output $F/a100_scores --verify-pipeline-results $F/results; echo "a100 score rc=$?"
$PY scripts/v2/a100_v2.py summarize --a100 $R/v2/a100_v2 --scores $F/a100_scores --output $F/a100_summary \
  --mllm $F/mllm/a100_lat $F/mllm/a100_aat; echo "a100 summarize rc=$?"
$PY scripts/v2/make_tables_v2.py --results $F/results --out $F/generated; echo "tables rc=$?"
$PY scripts/v2/check_paper_numbers.py --generated $F/generated; echo "numbers rc=$?"
touch $F/CPU_LANE_DONE
echo CPU_LANE_DONE
