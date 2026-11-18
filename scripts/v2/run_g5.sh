#!/usr/bin/env bash
# G5 pipeline dry run (Phase 1): ~2% subsample of each clean dataset x 3 encoders
# (small: resnet50, large: siglip2_so400m_384, MLLM embedder: qwen3_vl_emb_2b),
# extraction -> probes/sensitivities -> paired stats -> calibration -> rank stats ->
# tables/macros/figures -> number check, plus OutfitTransformer, zero-shot MLLM,
# efficiency profiler and near-duplicate audit. Train/valid only; test never scored.
# Each stage records its exit status; the gate passes only if every stage succeeds.
set -uo pipefail
cd "$(dirname "$0")/../.."
g2=${1:?G2 gate.json bound to the current code}
attempt=${2:?attempt name}
datasets=${3:-polyvore_d_clean,iqon3000_clean}
export REPBENCH_DATA_ROOT=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/data/$USER/.cache/huggingface/token}
export TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
R=$REPBENCH_DATA_ROOT
W=$R/v2/g5/$attempt
E=research/v2/readiness/$attempt/g5
mkdir -p "$W" "$E"
PY=.venv/bin/python
MODELS="resnet50 siglip2_so400m_384 qwen3_vl_emb_2b"
status="$E/stages.tsv"
stage() {  # stage NAME CMD...
  local name=$1; shift
  if grep -q "^$name	0	" "$status" 2>/dev/null; then echo "skip $name (done)"; return 0; fi
  local t0=$(date +%s)
  echo "=== $name: $*"
  "$@"
  local rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$status"
  return 0
}
declare -A SRC=([polyvore_d_clean]=$R/v2/g3_polyvore/rebuild_a [iqon3000_clean]=$R/v2/g3_iqon/rebuild_a
                [pog_clean]=$R/v2/pog_clean/build_a)
declare -A IMG=([polyvore_d_clean]=$R/raw/polyvore_outfits/images [iqon3000_clean]=$R/iqon3000_clean/images
                [pog_clean]=$R/raw/pog/mirror)
declare -A GRP=([polyvore_d_clean]=component [iqon3000_clean]=user [pog_clean]=two_way)
spec=""
for d in ${datasets//,/ }; do
  stage "subsample_$d" $PY scripts/v2/subsample_protocol.py --source "${SRC[$d]}" \
    --output "$W/protocols/$d" --fraction 0.02
  for m in $MODELS; do
    stage "extract_${d}_$m" $PY scripts/v2/extract_v2.py --g2 "$g2" --dataset "$d" \
      --protocol "$W/protocols/$d" --images-root "${IMG[$d]}" --model "$m" --output "$W/emb/$d/$m"
  done
  spec="$spec${spec:+,}$d=$W/protocols/$d=${GRP[$d]}"
done
stage pipeline $PY scripts/v2/pipeline.py --datasets "$spec" --models "${MODELS// /,}" \
  --embeddings "$W/emb" --results "$W/results"
stage tables $PY scripts/v2/make_tables_v2.py --results "$W/results" --out "$W/generated"
stage number_check $PY scripts/v2/check_paper_numbers.py --generated "$W/generated"
d0=${datasets%%,*}
stage outfit_transformer_train $PY scripts/v2/run_outfit_transformer_v2.py train \
  --protocol-root "$W/protocols/$d0" --dataset-name "$d0" --embeddings "$W/emb/$d0/resnet50" \
  --encoder resnet50 --output-root "$W/outfit_transformer" --lrs 1e-4 3e-4 --seeds 42 43 \
  --max-epochs 8 --patience 3 --device cuda
stage outfit_transformer_select $PY scripts/v2/run_outfit_transformer_v2.py select \
  --output-root "$W/outfit_transformer" --min-seeds 2
stage outfit_transformer_eval $PY scripts/v2/run_outfit_transformer_v2.py evaluate \
  --protocol-root "$W/protocols/$d0" --dataset-name "$d0" --embeddings "$W/emb/$d0/resnet50" \
  --output-root "$W/outfit_transformer" --split valid
stage mllm_freeze $PY scripts/v2/mllm_zeroshot.py freeze-ids --protocol-root "$W/protocols/$d0" \
  --dataset-name "$d0" --split valid --n 24 --seed 20261118 --output "$W/mllm_ids.json"
stage mllm_score $PY scripts/v2/mllm_zeroshot.py score --ids "$W/mllm_ids.json" \
  --protocol-root "$W/protocols/$d0" --images-root "${IMG[$d0]}" --output-dir "$W/mllm"
for m in $MODELS; do
  stage "profile_$m" $PY scripts/v2/profile_efficiency_v2.py --model "$m" --images-dir "${IMG[polyvore_d_clean]}" \
    --n-images 64 --energy-seconds 15 --output "$W/profile/$m.json" --g2 "$g2"
done
stage near_duplicates $PY scripts/v2/near_duplicate_audit.py --protocol-root "${SRC[polyvore_d_clean]}" \
  --dataset-name polyvore_d_clean --images-root "${IMG[polyvore_d_clean]}" \
  --output-dir "$W/near_duplicates_polyvore" --workers 12
stage env_capture $PY scripts/capture_env.py --output "$E/environment.json"
# Compact evidence (CSV/JSON only; predictions and weights stay on the DGX).
mkdir -p "$E/results" "$E/generated"
cp "$W"/results/*.csv "$W"/results/*.json "$E/results/" 2>/dev/null
cp "$W"/generated/*.tex "$W"/generated/*.json "$W"/generated/*.pdf "$E/generated/" 2>/dev/null
cp "$W"/results/*.pdf "$E/results/" 2>/dev/null
cp "$W"/outfit_transformer/selection.json "$E/outfit_transformer_selection.json" 2>/dev/null
cp "$W"/mllm/summary.json "$E/mllm_summary.json" 2>/dev/null
cp -r "$W"/profile "$E/" 2>/dev/null
cp "$W"/near_duplicates_polyvore/audit.json "$E/near_duplicates_polyvore.json" 2>/dev/null
for d in ${datasets//,/ }; do cp "$W/protocols/$d/manifest.json" "$E/subsample_$d.json"; done
$PY - "$E" "$g2" <<'EOF'
import json, sys
from pathlib import Path
from datetime import datetime, UTC
evidence, g2 = Path(sys.argv[1]), sys.argv[2]
rows = [l.split("\t") for l in (evidence / "stages.tsv").read_text().splitlines() if l.strip()]
stages = {name: {"exit": int(rc), "seconds": int(sec)} for name, rc, sec in rows}
failed = sorted(n for n, s in stages.items() if s["exit"])
from repbench.v2.state import input_hashes
gate = {"gate": "G5", "status": "PASS" if not failed else "FAIL", "utc": datetime.now(UTC).isoformat(),
        "scope": "dry run: ~2% train/valid subsample, 3 encoders; test never read or scored",
        "g2_evidence": g2, "failed_stages": failed, "stages": stages, "input_sha256": input_hashes()}
path = evidence / "gate.json"
if path.exists():
    path = evidence / f"gate-{datetime.now(UTC):%H%M%S}.json"
path.write_text(json.dumps(gate, indent=2))
print(gate["status"], failed)
EOF
