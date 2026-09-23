# Fashion Compatibility Benchmark

RepBench-Fashion is a controlled benchmark of seven frozen visual representations for relational
outfit Compatibility Prediction (CP) and Fill-in-the-Blank (FITB). It holds train-only preprocessing,
PCA-256, symmetric pair features, Logistic Regression, labels, candidates, and validation logic fixed
so that the released representation checkpoint is the experimental unit.

The broad modern contrastive-VLM tier advantage replicates across the two leakage-audited corpora and
is corroborated at tier level by A100. The principal conclusion remains stable across construction
seeds, feature dimensionality, nonlinear downstream learning, provenance restrictions, label
availability, and dependence-aware sensitivity analyses.

The public review repository is
[github.com/alifarzadjamali/fashion-compatibility-benchmark](https://github.com/alifarzadjamali/fashion-compatibility-benchmark).
Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the execution path and
[ARTIFACTS.md](ARTIFACTS.md) for the exact boundary between committed and regeneration-required
artifacts.

## Evidence stack

- `historical_polyvore_d`: unchanged packaged historical protocol, retained only for comparison;
  it does not satisfy the audited split-integrity controls.
- `polyvore_d_clean`: leakage-audited item-, encoded-image-, decoded-pixel-, and outfit-disjoint
  protocol.
- `iqon3000_clean`: prospective replication on a frozen leakage-audited IQON target.
- A100: unchanged external human/expert-grounded LAT/AAT evaluation; never used for fitting or tuning.

The seven frozen representations are ResNet50 IMAGENET1K_V2, DINOv3 ViT-L/16, CLIP ViT-L/14@336,
SigLIP2-B/16@384, FashionCLIP 2.0, Marqo-FashionSigLIP, and GR-Lite. Exact checkpoint revisions and
provenance caveats are locked in `configs/final_experimental_protocol.yaml`.

## Installation

Python 3.12 and an NVIDIA CUDA environment were used by the locked experiments. `pyproject.toml`
declares the dependency ranges and `uv.lock` records the resolved environment.

Linux/macOS:

```bash
uv venv --python 3.12 .venv
uv sync --extra dev
.venv/bin/python -m pytest
```

Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv sync --extra dev
.venv\Scripts\python.exe -m pytest
```

Unit tests run without private datasets. Dataset-dependent integration tests skip until the
independently obtained archives and generated protocols are present.

## Data and access

Original source images and annotations are not redistributed. Obtain each source under its own terms:

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
manifests provide construction hashes and statistics without redistributing source images. Pretrained
weights are downloaded from their original providers and remain subject to provider terms.

## Reproducing or inspecting the benchmark

Reviewers can inspect all frozen summary results, audits, tables, figures, protocol hashes, and source
without rerunning an encoder. A full numerical reproduction requires the public source datasets,
provider weights, and a fresh embedding extraction because cached frozen embeddings are not committed.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[`docs/final_experimental_protocol.md`](docs/final_experimental_protocol.md) before execution. The
high-level sequence is:

```powershell
# 1. Prepare/audit datasets and construct clean protocols.
# 2. Extract each frozen representation once.
.venv\Scripts\python.exe scripts\extract_embeddings.py --help

# 3. Run the locked controlled probes and secondary analyses.
.venv\Scripts\python.exe scripts\run_benchmark.py --help
.venv\Scripts\python.exe scripts\run_final_iqon.py --help

# Once immutable predictions exist, run scripts/final_phase_statistics.py.

# 4. Validate and assemble paper artifacts.
.venv\Scripts\python.exe scripts\make_final_outputs.py
.venv\Scripts\python.exe scripts\validate_final_package.py
```

Large generated outputs are intentionally absent. In particular, the Git repository does not contain
cached frozen embeddings, fitted binary artifacts, raw per-example predictions, source images, or
model weights. `artifacts/final_artifact_manifest.csv` records paths and hashes from the frozen run;
it is an integrity inventory, not a statement that every listed file is distributed. Curated derived
result tables and figures are committed under `reports/`.

## Results

Modern contrastive VLM representations form a consistently stronger compatibility tier than ResNet50
and generic DINOv3. The family-level hierarchy replicates on the frozen IQON3000-Clean protocol and is
corroborated by A100. Close checkpoint ordering is dataset-dependent, fashion specialization is
recipe-dependent, and retrieval ranking is not a reliable compatibility-selection rule in the exact
LookBench overlap. Calibration, efficiency, and provenance provide complementary selection criteria.

Start with:

- `reports/final_experiment_report.md`
- `reports/final_statistics_report.md`
- `reports/final_reviewer_red_team.md`
- `reports/paper_readiness_report_v2.md`

For repository-level guidance, see [`docs/final_experimental_protocol.md`](docs/final_experimental_protocol.md),
[`docs/release_artifact_policy.md`](docs/release_artifact_policy.md), and the
[maintenance checklist](docs/maintenance_checklist.md).

## Reproducibility and limitations

- Test data never fit PCA, scalers, classifiers, hyperparameters, thresholds, or calibration.
- A100 remains strictly external and uses its existing candidates and annotations.
- Marqo-FashionSigLIP and GR-Lite have the greatest fashion-data provenance uncertainty; excluding
  both preserves the family-level result but does not certify other web-pretrained models.
- IQON-Clean is item/image-disjoint rather than user-disjoint. Its retained-subset audit preserves
  98.74% of source users and the category mixture while identifying shifts in outfit length and item
  repetition.
- Exact-byte and decoded-pixel duplicates are excluded; transformed near-duplicates remain possible.
- Original repository code and original documentation are licensed under Apache-2.0. Datasets,
  annotations, model weights, and trademarks remain governed by their respective owners' terms.

## License

This repository's original code, documentation, configuration, and original publication artifacts
are available under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for exclusions and
third-party-material boundaries.

See `docs/decision_log.md` and `reports/final_reviewer_red_team.md` for the complete audit trail.

## Authors and citation

Author order: Ali Jamali (first and corresponding), Ali Alameer, Maryam Vadikheil, Parham Imanzadeh
Charandabi, and Taha Mansouri (senior/last). Machine-readable citation metadata are in
[CITATION.cff](CITATION.cff). No archival DOI is claimed.
