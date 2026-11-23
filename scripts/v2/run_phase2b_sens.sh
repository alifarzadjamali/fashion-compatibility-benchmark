#!/usr/bin/env bash
# Phase 2b (deviation D2): one step of the run_full_cpu.sh post-stats CPU lane, so steps run as
# parallel tmux sessions. Same scripts, arguments and output paths as run_full_cpu.sh; the
# sensitivity subcommands write to <results>/sensitivities/<step>/ exactly as `all` does.
# Usage: run_phase2b_sens.sh <label-fractions|construction-seeds|provenance|subgroups|lookbench|
#                              nd-polyvore|nd-pog|nd-iqon|near-duplicates|a100-tables>
set -uo pipefail
cd "$(dirname "$0")/../.."
step=${1:?step}
freeze=configs/v2_protocol_hash.json
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=20260912
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}; F=$R/v2/full; PY=.venv/bin/python
EQ="--equivalence polyvore_v2_clean=$R/v2/image_equivalence/polyvore/equivalence_groups.json \
 iqon3000_v2_clean=$R/v2/image_equivalence/iqon/equivalence_groups.json pog_v2_clean=$R/v2/image_equivalence/pog/equivalence_groups.json"
COMMON="--results $F/results --eval-split test --frozen-protocol-hash $freeze --dataset-config configs/v2_sensitivity_datasets.json"
PROBE="--embeddings $F/emb --selection-split valid"
waitf() { until [ -f "$1" ]; do [ -f $F/STATS_FAILED ] && { echo "STATS_FAILED; abort $step"; exit 1; }; sleep 120; done; }
nd() {  # dataset images_root ready_marker
  waitf "$3"
  [ -f $F/near_duplicates/$1/audit.json ] || $PY scripts/v2/near_duplicate_audit.py --protocol-root $R/v2/$1/build_a \
    --dataset-name $1 --images-root $2 --embeddings $F/emb/$1/dinov3_vitl16 --cosine 0.95 \
    --output-dir $F/near_duplicates/$1 --workers 8
}
echo "=== $step start $(date '+%F %T')"
case $step in
  nd-polyvore) nd polyvore_v2_clean $R/raw/polyvore_outfits/images $R/raw/polyvore_outfits/IMAGES_READY ;;
  nd-pog)      nd pog_v2_clean $R/raw/pog/mirror $R/raw/pog/IMAGES_READY ;;
  nd-iqon)     nd iqon3000_v2_clean $R/iqon3000_clean/images $R/iqon3000_clean/IMAGES_READY ;;
  label-fractions|provenance|subgroups)
    waitf $F/STATS_DONE
    extra=""; [ $step = label-fractions ] && extra="$PROBE"
    $PY scripts/v2/run_sensitivities.py $step $COMMON $extra $EQ ;;
  construction-seeds)
    waitf $F/STATS_DONE; waitf $R/v2/iqon_preparation/catalog/CATALOG_READY
    $PY scripts/v2/run_sensitivities.py $step $COMMON $PROBE $EQ ;;
  lookbench)
    waitf $F/STATS_DONE
    [ -f $F/results/sensitivities/lookbench/manifest.json ] || $PY scripts/v2/run_sensitivities.py lookbench \
      --metrics $F/results/metrics.csv --out $F/results/sensitivities/lookbench --lookbench-scores configs/lookbench_v3_scores.json ;;
  near-duplicates)
    waitf $F/STATS_DONE; aff=""
    for d in polyvore_v2_clean iqon3000_v2_clean pog_v2_clean; do
      waitf $F/near_duplicates/$d/audit.json; aff="$aff --affected $d=$F/near_duplicates/$d/affected_test_outfits.json"; done
    $PY scripts/v2/run_sensitivities.py near-duplicates $COMMON $aff $EQ ;;
  a100-tables)
    waitf $F/STATS_DONE
    MODELS="resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip siglip_b16_224 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 pe_core_l14_336 siglip2_so400m_384 qwen3_vl_emb_2b qwen3_vl_emb_8b"
    a100_models=$(for m in $MODELS; do [ -f $F/failed/a100.$m ] || echo $m; done | tr '\n' ',' | sed 's/,$//')
    corpora="polyvore_v2_clean=$R/v2/polyvore_v2_clean/build_a,iqon3000_v2_clean=$R/v2/iqon3000_v2_clean/build_a,pog_v2_clean=$R/v2/pog_v2_clean/build_a"
    $PY scripts/v2/a100_v2.py score --a100 $R/v2/a100_v2 --embeddings $F/emb --corpora "$corpora" \
      --models $a100_models --output $F/a100_scores --verify-pipeline-results $F/results; echo "a100 score rc=$?"
    $PY scripts/v2/a100_v2.py summarize --a100 $R/v2/a100_v2 --scores $F/a100_scores --output $F/a100_summary \
      --mllm $F/mllm/a100_lat $F/mllm/a100_aat; echo "a100 summarize rc=$?"
    for s in label_fractions construction_seeds provenance subgroups near_duplicates lookbench; do
      waitf $F/results/sensitivities/$s/manifest.json; done
    $PY scripts/v2/make_tables_v2.py --results $F/results --out $F/generated; echo "tables rc=$?"
    $PY scripts/v2/check_paper_numbers.py --generated $F/generated; echo "numbers rc=$?"
    touch $F/CPU_LANE_DONE ;;
  *) echo "unknown step $step"; exit 2 ;;
esac
rc=$?; echo "=== $step end rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && echo "$step rc=$rc" >> $F/failed/SENS_STEPS
exit $rc
