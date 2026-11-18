#!/usr/bin/env bash
# G6 speed test: waits for other GPU jobs (IQON G3), then measures each encoder in
# its own process (required four first), then writes the G6 summary.
set -uo pipefail
cd "$(dirname "$0")/../.."
attempt=${1:?attempt name}
export REPBENCH_DATA_ROOT=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/data/$USER/.cache/huggingface/token}
export TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
g2=research/v2/readiness/2026-11-18-a03/g2/gate.json
out=research/v2/readiness/$attempt/g6
root=$REPBENCH_DATA_ROOT
while tmux has-session -t repbench-iqon-g3 2>/dev/null; do sleep 30; done
models="resnet50 clip_vitl14_336 siglip2_so400m_384 qwen3_vl_emb_2b qwen3_vl_emb_8b
 dinov3_vitl16 dinov3_vitl16_336 pe_core_l14_336 siglip2_b16_384 siglip_b16_224
 openclip_b32_laion2b fashionclip2 marqo_fashionsiglip marqo_fashionclip gr_lite"
for model in $models; do
  [ -f "$out/$model.json" ] && continue
  timeout 1800 .venv/bin/python scripts/v2/g6_speed.py measure --g2 "$g2" --model "$model" \
    --polyvore-root "$root/raw/polyvore_outfits" \
    --train-json "$root/v2/g3_polyvore/rebuild_a/train.json" \
    --output "$out/$model.json" \
    --note "concurrent: POG CDN image download (network-bound, ~1 CPU core)" \
    || echo "G6 measure failed: $model (exit $?)"
done
.venv/bin/python scripts/v2/g6_speed.py summarize --input "$out" --output "$out/gate.json"
