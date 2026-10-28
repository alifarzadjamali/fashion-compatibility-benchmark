# Fashion Compatibility Benchmark

RepBench-Fashion compares seven frozen image representations for outfit compatibility
prediction (CP) and fill-in-the-blank completion (FITB). Each representation uses the same
train-only preprocessing, PCA-256, symmetric pair features, and logistic classifier.
The encoders are ResNet50, DINOv3, CLIP, SigLIP2, FashionCLIP, Marqo-FashionSigLIP, and GR-Lite.
Checkpoint revisions and experimental settings are in
[`configs/final_experimental_protocol.yaml`](configs/final_experimental_protocol.yaml).

## Repository contents

- `src/repbench/`: datasets, encoders, features, models, and evaluation functions.
- `scripts/`: data preparation, protocol audits, extraction, training, and analysis.
- `configs/`: checkpoint settings and the frozen experimental protocol.
- `tests/`: unit tests and optional checks against locally available datasets.
- [`paper_elsevier_ivc/`](paper_elsevier_ivc/main.pdf): Image and Vision Computing manuscript and supplement.
- [`paper_springer_mva/`](paper_springer_mva/main.pdf): Machine Vision and Applications manuscript and supplement.

Datasets, weights, embeddings, predictions, and generated reports stay local
and are ignored by Git. The two journal manuscript packages are tracked. Scripts create `data/`, `artifacts/`, and `reports/` as needed.
The source licenses are separate from the [Apache-2.0 code license](LICENSE); see [NOTICE](NOTICE).

## Install

Use Python 3.12 and an NVIDIA GPU compatible with the locked CUDA 12.8 PyTorch wheels.
The recorded experiments used Windows 11 and an RTX 5070 Ti with 16 GB VRAM.
`uv.lock` pins the Python dependencies. Commands below run from the repository root.

```powershell
uv sync --locked --extra dev
uv run --frozen python -m pytest
```

The recorded versions were Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 4.57.6,
and scikit-learn 1.9.1. Different hardware and numerical libraries can change timings and
floating-point results. Dataset-dependent tests skip when the required local files are absent.

## Obtain the inputs

Acquire the Polyvore Outfits, IQON3000, and A100 research datasets under their respective
terms. The pipeline does not redistribute source images or annotations. Model weights come
from the providers named in the encoder registry. DINOv3 requires approved Hugging Face
access and local authentication.

The commands below use these local locations, matching the analysis scripts' defaults:

| Input | Local location |
| --- | --- |
| Polyvore metadata, disjoint annotations, and images | `data/raw/polyvore_outfits/` |
| IQON3000 source ZIP | `.venv/iqon_audit/IQON3000.zip` |
| A100 source ZIP | `.venv/a100_audit/A100.zip` |

Polyvore needs `images/`, `disjoint/`, and `polyvore_item_metadata.json`. For the supported
Kaggle archive layout, `scripts/prepare_polyvore.py --kaggle-archive PATH` extracts these
files. Its audit rejects the historical split's known item overlap after extraction;
construct the clean protocol below before using it for the primary comparison.
The preparation script's default metadata download does not supply the complete image archive.

Keep these dataset directories when recreating the Python environment: some historical
script defaults place data and embeddings inside `.venv/`. Most preparation and extraction
scripts expose path overrides through `--help`; several analysis scripts use the defaults.

## Construct the protocols

```powershell
uv run --frozen python scripts/build_clean_protocol.py --seed 20260912
uv run --frozen python scripts/validate_clean_protocol.py

uv run --frozen python scripts/audit_iqon3000.py --archive .venv/iqon_audit/IQON3000.zip --output artifacts/iqon3000_source_audit.json --catalog-dir .venv/iqon_audit/catalog
uv run --frozen python scripts/audit_iqon_images.py --archive .venv/iqon_audit/IQON3000.zip --items .venv/iqon_audit/catalog/items.parquet --output-parquet .venv/iqon_audit/catalog/image_audit.parquet --output-report artifacts/iqon3000_image_audit.json
uv run --frozen python scripts/resolve_iqon_image_variants.py --archive .venv/iqon_audit/IQON3000.zip --report artifacts/iqon3000_image_audit.json --image-audit .venv/iqon_audit/catalog/image_audit.parquet --output .venv/iqon_audit/catalog/image_audit_resolved.parquet
uv run --frozen python scripts/build_iqon_clean_protocol.py --extract-images --seed 20260912
uv run --frozen python scripts/audit_iqon_selection.py
uv run --frozen python scripts/prepare_a100.py
```

Expected positive-outfit counts (train / validation / test) are 16,991 / 2,999 / 15,142
for Polyvore-D-Clean and 86,800 / 12,400 / 24,800 for IQON3000-Clean. CP adds one synthetic
negative per positive; FITB has four candidates. A100 retains its released five candidates
and 100 questions per task, with no training or tuning on its labels.

Both clean protocols enforce item and exact-image separation. IQON retains 41.62% of valid
source outfits and is not user-disjoint. Exact-image audits do not exclude all transformed
near-duplicates or overlap with upstream pretraining data.

## Extract frozen features

The following PowerShell loop extracts all seven representations. On other shells, run the
two extraction commands for each model key. Historical Polyvore embeddings cover the pooled
items used by the clean split; the clean runner repartitions them by the clean protocol.

```powershell
$models = @('resnet50', 'dinov3_vitl16', 'clip_vitl14_336', 'siglip2_b16_384', 'fashionclip2', 'marqo_fashionsiglip', 'gr_lite')
foreach ($model in $models) {
    uv run --frozen python scripts/extract_embeddings.py --model-key $model --dataset-key historical_polyvore_d --batch-size 64
    if ($LASTEXITCODE -ne 0) { throw "Polyvore extraction failed: $model" }
    uv run --frozen python scripts/extract_embeddings.py --root .venv/iqon3000_clean --protocol-dir data/protocols/iqon3000_clean --dataset-key iqon3000_clean --output-root .venv/iqon3000_clean/embeddings --model-key $model --batch-size 32
    if ($LASTEXITCODE -ne 0) { throw "IQON extraction failed: $model" }
}
uv run --frozen python scripts/extract_a100_embeddings.py
```

IQON extraction writes a dataset subdirectory. Pass that directory to the runner below.
Extraction is the most expensive stage. `--skip-valid-existing` can reuse checked caches
where the script supports it; do not mix checkpoint revisions or partial extraction runs.

## Train and evaluate

```powershell
uv run --frozen python scripts/run_benchmark.py --protocol-dir data/protocols/polyvore_d_clean --dataset-name polyvore_d_clean --prefix clean_primary
uv run --frozen python scripts/restore_polyvore_classifiers.py
uv run --frozen python scripts/run_final_iqon.py --embedding-root .venv/iqon3000_clean/embeddings/iqon3000_clean
uv run --frozen python scripts/evaluate_a100.py
```

Primary outputs are `artifacts/clean_primary_results.csv`,
`artifacts/final/iqon3000_clean/primary_results.csv`, and `artifacts/final/a100/summary.csv`.
Several final-stage scripts refuse to overwrite completed outputs. Use a fresh working copy
for a new experiment or preserve the previous output directory before rerunning.

Reference results from the frozen PCA-256 experiment:

| Representation | Polyvore CP AUC | Polyvore FITB | IQON CP AUC | IQON FITB |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | .6938 | .4063 | .5852 | .3166 |
| DINOv3 | .7822 | .5048 | .6419 | .3660 |
| CLIP | .8883 | .6500 | .7042 | .4364 |
| SigLIP2 | .8892 | .6528 | .7031 | .4330 |
| FashionCLIP | .8690 | .6175 | .6981 | .4275 |
| Marqo-FashionSigLIP | .8897 | .6479 | .7221 | .4451 |
| GR-Lite | .8277 | .5573 | .6511 | .3832 |

FITB chance accuracy is .25. These values compare complete released checkpoints;
they do not isolate architecture, training data, or loss. Marqo and GR-Lite have greater
fashion-data provenance uncertainty. Excluding them preserves the broad VLM advantage.

## Secondary experiments and analysis

The primary commands above reproduce the main frozen-probe comparison. The full analysis
also requires the following stages. Run each script's `--help` for dataset and output
arguments before launching an experiment. Scripts without CLI arguments use repository defaults.

| Analysis | Entry points and prerequisites |
| --- | --- |
| Historical comparison | `run_benchmark.py --allow-known-overlap --prefix primary`; repeat with `--track native --prefix native` |
| PCA width | Repeat historical and clean runs with `--pca-components 128` and `512`, using prefixes `historical_pca128`, `historical_pca512`, `clean_pca128`, and `clean_pca512` |
| Clean native features | Repeat the clean `run_benchmark.py` command with `--track native --prefix clean_native` |
| IQON native features | `run_benchmark.py` with the IQON root and protocol, `--embedding-root .venv/iqon3000_clean/embeddings --embedding-dataset-key iqon3000_clean --dataset-name iqon3000_clean --track native --prefix iqon_native` |
| Historical question bridge | `build_regenerated_historical_protocol.py`, then `run_benchmark.py` with that protocol, `--allow-known-overlap --dataset-name historical_polyvore_d_regenerated --prefix historical_regenerated` |
| Label fractions and alternative learners | `run_robustness.py` for historical and clean Polyvore; for IQON use `--jobs xgboost --prefix iqon_xgboost --pca-root artifacts/final/iqon3000_clean` with the IQON dataset, protocol, and embedding arguments |
| Construction seeds | `build_construction_stability.py`, then `run_construction_stability.py` for both clean datasets |
| Contextual baseline | `run_outfit_transformer_baseline.py` for each clean dataset, then `evaluate_a100_outfit_transformer.py` |
| Provenance and efficiency | `write_model_audit.py`, `extend_provenance_for_iqon.py`, `profile_efficiency.py` |
| Polyvore intervals and comparisons | `analyze_benchmark.py`, `paper_statistics.py`, `cluster_bootstrap.py`, then `make_paper_artifacts.py`; the last step needs efficiency measurements and robustness runs |
| IQON and A100 statistics | `final_phase_statistics.py` after their predictions exist |
| Calibration and subgroups | `final_calibration.py`, `final_subgroup_analysis.py`, `render_reliability.py` |
| Additional validity checks | `audit_question_construction.py`, `audit_iqon_dependence.py`, `audit_external_image_overlap.py` |
| Final tables and figures | `make_final_outputs.py`, `make_paper_revision_figures.py`, then `validate_final_package.py` after all required experiments and analyses |

The optional feature-standardization diagnostic needs a preserved development run and is
skipped when that run is absent. It is not required for the final experimental package.

Generated outputs stay under ignored `artifacts/` and `reports/`. The validator checks
experiment payloads, numerical tables, and figures; it does not require editorial reports
or journal manuscripts. No full experiment is launched by the unit test command.

The frozen protocol text is preserved verbatim in
[`configs/final_experimental_protocol.txt`](configs/final_experimental_protocol.txt), including
its historical local-path references. The YAML is its machine-readable counterpart.
[`configs/final_experimental_protocol_hash.json`](configs/final_experimental_protocol_hash.json)
records their LF-normalized hashes and the original combined hash
`96ffd69883c381f83a853ca6f73fbc30cbd891366ba34f79ceb314cbe8d0f16b`.
These files specify the experiment; generated audit inventories are local outputs.

## Citation

Author and repository citation metadata are in [CITATION.cff](CITATION.cff).
