# Reproducibility guide

This repository supports two review paths: inspection of the frozen evidence without rerunning any
vision model, and full regeneration from publicly acquired source datasets and provider checkpoints.
The scientific results reported in the manuscripts are frozen.

## Inspect the frozen evidence

No encoder run is needed to inspect the reported evidence. The principal reviewer entry points are:

- `configs/final_experimental_protocol.yaml` and `docs/final_experimental_protocol.md`: locked design,
  seeds, checkpoint identifiers, contrast family, and analysis rules;
- `artifacts/final_experimental_protocol_hash.json`: SHA-256 lock for the protocol documents;
- `data/protocols/*/manifest.json`: committed protocol statistics and file hashes;
- `artifacts/*audit*`, `artifacts/model_audit.csv`, and `artifacts/experiment_registry.csv`: split,
  image, provenance, and execution audits;
- `reports/tables/final/`: curated numerical result tables, predefined effects, sensitivity analyses,
  calibration, efficiency, provenance, subgroup, and task-specific baseline summaries;
- `reports/figures/final/`: publication figures derived from the frozen results;
- `reports/final_*.md`: narrative reports linking the frozen evidence to the paper claims;
- `paper_springer_mva/` and `paper_elsevier_ivc/`: independently compilable journal manuscripts.

The protocol lock begins `96ffd69883c381f8`. The IQON split-manifest hash begins
`f5081ec5d2ea0262`; complete hashes are stored in the committed inventories.

## Environment

The locked run used Python 3.12.13, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
Transformers 4.57.6, scikit-learn 1.9.1, CUDA 12.8, and an NVIDIA RTX 5070 Ti. The dependency
declaration is `pyproject.toml`; `uv.lock` is the resolved lock file. Detailed run metadata are in
`artifacts/paper_reproducibility_metadata.json` and `artifacts/efficiency_environment.json`.

Linux/macOS setup:

```bash
uv venv --python 3.12 .venv
uv sync --extra dev
.venv/bin/python -m pytest
```

Windows PowerShell setup:

```powershell
uv venv --python 3.12 .venv
uv sync --extra dev
.venv\Scripts\python.exe -m pytest
```

Tests that require datasets skip when the independently acquired archives and generated payloads are
absent. Unit tests do not require private data.

## Acquire data and weights

The repository does not redistribute Polyvore, IQON3000, or A100 source images or annotations. Obtain
them from their official research releases or an authorized source and comply with their licenses and
terms. Pretrained weights are obtained from the original providers. DINOv3 requires gated Hugging Face
access; authenticate locally and never commit a token.

The seven checkpoint names, immutable revisions, input resolutions, and extraction precisions are in
`configs/final_experimental_protocol.yaml` and the per-encoder files under `configs/encoders/`.

## Regenerate protocols and embeddings

The scripts expose path arguments through `--help`; paths below are the repository defaults or
illustrative local paths.

```bash
# Polyvore preparation and leakage-audited reconstruction
.venv/bin/python scripts/prepare_polyvore.py --help
.venv/bin/python scripts/build_clean_protocol.py \
  --root data/raw/polyvore_outfits \
  --output data/protocols/polyvore_d_clean \
  --seed 20260912

# IQON source audit and leakage-audited reconstruction
.venv/bin/python scripts/audit_iqon3000.py \
  --archive /path/to/IQON3000.zip \
  --output artifacts/iqon3000_source_audit.json \
  --catalog-dir /path/to/iqon_catalog
.venv/bin/python scripts/build_iqon_clean_protocol.py --help

# A100 preparation
.venv/bin/python scripts/prepare_a100.py --help

# Frozen feature extraction; repeat for each model key and dataset protocol
.venv/bin/python scripts/extract_embeddings.py --help
```

Model keys are `resnet50`, `dinov3_vitl16`, `clip_vitl14_336`, `siglip2_b16_384`,
`fashionclip2`, `marqo_fashionsiglip`, and `gr_lite`. Extraction uses the revisions recorded in the
locked protocol.

## Recompute evaluation and reports

After embeddings and protocol payloads exist locally:

```bash
.venv/bin/python scripts/run_benchmark.py --help
.venv/bin/python scripts/run_final_iqon.py --help
.venv/bin/python scripts/final_phase_statistics.py
.venv/bin/python scripts/make_final_outputs.py
.venv/bin/python scripts/validate_final_package.py
```

Secondary analysis entry points include `scripts/run_construction_stability.py`,
`scripts/run_robustness.py`, `scripts/analyze_native_robustness.py`,
`scripts/final_calibration.py`, `scripts/final_subgroup_analysis.py`, and
`scripts/evaluate_a100.py`. The exact settings are recorded in the locked protocol and
`configs/experiments/paper_robustness.yaml`.

## Artifact boundary

The Git repository includes curated derived tables and figures, protocol manifests, audit summaries,
hash inventories, source code, configuration, and manuscript sources. It does not include source
images, model weights, cached frozen embeddings, fitted joblib objects, raw Parquet predictions, or
other large run caches. See `ARTIFACTS.md` for a path-by-path inventory. In particular,
`artifacts/final_artifact_manifest.csv` records the frozen private-run package and its hashes; it does
not imply that every listed file is distributed through Git.

## Compile the manuscripts

From each manuscript directory, use the recorded sequence in `compile-main.txt` and
`compile-supplementary.txt`. The Springer and Elsevier versions intentionally differ in document
class, front matter, declaration headings, and bibliography style while retaining the same scientific
body, results, and conclusions.
