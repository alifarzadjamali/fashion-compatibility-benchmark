#!/usr/bin/env bash
cd ~/fcb-freeze
R=/data/a_jamali2/repbench/data; F=$R/v2/full
export HF_HOME=/data/a_jamali2/repbench/cache/huggingface TORCH_HOME=/data/a_jamali2/repbench/cache/torch HF_TOKEN_PATH=/data/a_jamali2/.cache/huggingface/token CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260912 TOKENIZERS_PARALLELISM=false
D="--dataset polyvore_v2_clean=$R/raw/polyvore_outfits/images=$R/v2/polyvore_v2_clean/build_a=$F/emb --dataset iqon3000_v2_clean=$R/iqon3000_clean/images=$R/v2/iqon3000_v2_clean/build_a=$F/emb --dataset pog_v2_clean=$R/raw/pog/mirror=$R/v2/pog_v2_clean/build_a=$F/emb"
.venv/bin/python scripts/v2/spot_checks_v2.py --check precision --models gr_lite,siglip2_so400m_384,marqo_fashionsiglip --n 512 --out $F/spot_checks $D || echo FAIL precision
.venv/bin/python scripts/v2/spot_checks_v2.py --check qwen_pixels --models qwen3_vl_emb_2b,qwen3_vl_emb_8b --n 256 --out $F/spot_checks $D || echo FAIL qwen
touch $F/SPOT_DONE
