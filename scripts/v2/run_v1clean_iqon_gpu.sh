#!/usr/bin/env bash
# Post-run sweep B7 (2026-11-24): v1-Clean vs v2-Clean leakage-inflation analysis (protocol:
# "v1-Clean splits evaluated only as leakage-inflation analysis"). v1-Clean IQON references
# 74k items absent from the v2 stores: extract their images (existing images are checksum-
# verified and kept) and embed them with the as-run encoder code, reusing every v2 vector.
# Encoders: ResNet-50 + the validation-selected generic and fashion encoders for IQON.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}; F=$R/v2/full; PY=.venv/bin/python
export HF_HOME=${HF_HOME:-/data/$USER/repbench/cache/huggingface} TORCH_HOME=${TORCH_HOME:-/data/$USER/repbench/cache/torch}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/data/$USER/.cache/huggingface/token} TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912
P=$R/v2/iqon_preparation; V=$R/v1clean
echo "=== images $(date '+%F %T')"
[ -f $V/IQON_IMAGES_READY ] || { $PY scripts/v2/extract_iqon_images.py --protocol $V/iqon --catalog $P/catalog \
  --archive $R/iqon_audit/IQON3000.zip --output $R/iqon3000_clean/images --report $V/iqon_extract_report.json \
  && touch $V/IQON_IMAGES_READY; } || { echo "FAILED images"; exit 1; }
for m in resnet50 siglip2_so400m_384 marqo_fashionsiglip; do
  echo "=== embed $m $(date '+%F %T')"
  $PY scripts/v2/extract_v2.py --g2 research/v2/readiness/2026-11-21-a09/g2/gate.json --dataset iqon3000_clean \
    --protocol $V/iqon --images-root $R/iqon3000_clean/images --splits train,valid,test --model $m \
    --output $V/emb/iqon3000_clean/$m/all \
    --reuse-from $F/emb/iqon3000_v2_clean/$m/trainvalid $F/emb/iqon3000_v2_clean/$m/test || echo "FAILED embed $m"
done
touch $V/IQON_EMB_DONE; echo "=== done $(date '+%F %T')"
