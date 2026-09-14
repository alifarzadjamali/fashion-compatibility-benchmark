# Fashion Compatibility Benchmark

RepBench-Fashion is a controlled benchmark of seven frozen visual representations for outfit
Compatibility Prediction (CP) and Fill-in-the-Blank (FITB). It holds the downstream method fixed—
train-only preprocessing, PCA-256, symmetric pair features, and Logistic Regression—so the image
representation is the main independent variable.

The release includes source code, exact model revisions, construction/evaluation configs, tests,
audit manifests, final reports, and publication tables/figures. It deliberately does **not** contain
datasets, images, embeddings, model checkpoints, raw predictions, caches, credentials, or logs.

## Evidence stack

- `historical_polyvore_d`: unchanged packaged historical protocol, retained only for comparison;
  it is not described as item-disjoint or leakage-free.
- `polyvore_d_clean`: corrected item-, exact-file-, decoded-pixel-, and outfit-disjoint protocol.
- `iqon3000_clean`: independent prospective replication with the same leakage controls.
- A100: unchanged external human/expert-grounded LAT/AAT evaluation; never used for fitting or tuning.

The seven frozen representations are ResNet50 IMAGENET1K_V2, DINOv3 ViT-L/16, CLIP ViT-L/14@336,
SigLIP2-B/16@384, FashionCLIP 2.0, Marqo-FashionSigLIP, and GR-Lite. Exact checkpoint revisions and
provenance caveats are locked in `configs/final_experimental_protocol.yaml`.

## Installation

Python 3.12 and an NVIDIA CUDA environment are used by the locked experiments.

```powershell
uv venv --python 3.12 .venv
uv sync --extra dev
.venv\Scripts\python.exe -m pytest
```

Unit tests run without private datasets. Dataset-dependent integration tests skip until the
independently obtained archives and generated protocols are present.

## Data and access

Dataset files are not redistributed. Obtain each source under its own terms:

1. Polyvore Outfits from the official/publicly documented source or an authorized mirror.
2. IQON3000 from its research release.
3. A100 from the AiDLab-fAshIon-Data research archive.
4. Request Hugging Face gated access for `facebook/dinov3-vitl16-pretrain-lvd1689m` and authenticate
   locally. Never place a Hugging Face token in this repository.

Preparation and audit entry points:

```powershell
.venv\Scripts\python.exe scripts\prepare_polyvore.py --help
.venv\Scripts\python.exe scripts\audit_iqon3000.py --help
.venv\Scripts\python.exe scripts\prepare_a100.py --help
.venv\Scripts\python.exe scripts\build_clean_protocol.py --help
.venv\Scripts\python.exe scripts\build_iqon_clean_protocol.py --help
```

Generated data remain under ignored `data/`, `.venv/`, and `artifacts/` paths. The committed protocol
manifests provide construction hashes and statistics without redistributing split payloads.

## Reproducing the benchmark

Read `docs/final_experimental_protocol.md` before execution. The high-level sequence is:

```powershell
# 1. Prepare/audit datasets and construct clean protocols.
# 2. Extract each frozen representation once.
.venv\Scripts\python.exe scripts\extract_embeddings.py --help

# 3. Run the locked controlled probes and secondary analyses.
.venv\Scripts\python.exe scripts\run_benchmark.py --help
.venv\Scripts\python.exe scripts\run_final_iqon.py --help
.venv\Scripts\python.exe scripts\final_phase_statistics.py --help

# 4. Validate and assemble paper artifacts.
.venv\Scripts\python.exe scripts\make_final_outputs.py
.venv\Scripts\python.exe scripts\validate_final_package.py
```

Large generated outputs are intentionally absent. `artifacts/final_artifact_manifest.csv` records the
validated private-run package, while `artifacts/final_report_manifest.csv` records the report hashes.

## Results

The broad modern-VLM advantage replicates from Polyvore-D-Clean to IQON3000-Clean. Exact top-model
ordering is dataset-dependent, fashion specialization is not uniformly beneficial, and retrieval
ranking is a weak proxy for compatibility ranking in the exact LookBench overlap. The task-specific
baseline wins IQON CP but not IQON FITB and transfers less well to A100.

Start with:

- `reports/final_experiment_report.md`
- `reports/final_statistics_report.md`
- `reports/final_reviewer_red_team.md`
- `reports/paper_readiness_report_v2.md`

## Reproducibility and limitations

- Test data never fit PCA, scalers, classifiers, hyperparameters, thresholds, or calibration.
- A100 remains strictly external and uses its existing candidates and annotations.
- Marqo-FashionSigLIP and GR-Lite remain flagged for uncertain fashion-training provenance.
- IQON-Clean is item/image-disjoint, not user-disjoint, and retains 41.62% of valid source outfits.
- Exact-byte and decoded-pixel duplicates are excluded; transformed near-duplicates remain possible.
- No code license has been selected yet; public visibility does not itself grant reuse rights.

See `docs/decision_log.md` and `reports/final_reviewer_red_team.md` for the complete audit trail.

