# Artifact inventory

This inventory distinguishes files available in the public Git repository from files that must be
regenerated locally. The repository is public at
https://github.com/alifarzadjamali/fashion-compatibility-benchmark.

## Committed and directly inspectable

| Evidence | Location | Contents |
|---|---|---|
| Locked design | `configs/final_experimental_protocol.yaml`, `docs/final_experimental_protocol.md` | Models, revisions, seeds, protocol roles, metrics, contrasts, and robustness plan |
| Protocol hashes | `artifacts/final_experimental_protocol_hash.json` | Complete SHA-256 values for the locked design |
| Protocol manifests | `data/protocols/*/manifest.json` | Construction statistics, split counts, source hashes, question rules, and generated-file hashes |
| Audit summaries | `artifacts/*.json`, `artifacts/*.csv` | Image duplication, split integrity, IQON selection, provenance, model, and environment audits |
| Curated results | `reports/tables/final/` | Main results, predefined effects, A100, robustness, calibration, efficiency, provenance, and subgroup tables |
| Publication figures | `reports/figures/final/`, `paper_*/figures/` | Frozen figures used by the manuscripts |
| Narrative reports | `reports/final_*.md` | Cross-dataset, statistics, calibration, A100, efficiency, and reviewer-audit reports |
| Reproduction code | `src/`, `scripts/`, `tests/`, `configs/` | Protocol construction, extraction, evaluation, statistics, validation, and tests |
| Environment | `pyproject.toml`, `uv.lock`, `artifacts/paper_reproducibility_metadata.json` | Declared, resolved, and recorded run environments |
| Manuscripts | `paper_springer_mva/`, `paper_elsevier_ivc/` | Journal-specific LaTeX, bibliography, supplement, and compiled PDFs |

The curated CSV and Markdown tables are derived numerical results. They permit direct inspection of
the paper's point estimates, confidence intervals, predefined contrasts, rank correlations, and
sensitivity analyses without rerunning an encoder.

## Recorded by hash but not distributed in Git

`artifacts/final_artifact_manifest.csv` inventories the complete frozen run package, including large
or restricted files that are absent from Git. Typical regeneration-required entries are:

- cached frozen item embeddings;
- fitted PCA, scaler, classifier, and task-specific baseline objects;
- raw CP and FITB per-example predictions and candidate scores;
- bootstrap replicate files and large intermediate tables;
- source dataset payloads and extracted images;
- local logs, caches, and checkpoint downloads.

The manifest is an integrity and traceability record, not a distribution claim. Some paths in it refer
to the original frozen run environment.

## Never redistributed here

- Original Polyvore, IQON3000, and A100 images or source archives.
- Pretrained checkpoint weights obtained from torchvision, Hugging Face, OpenCLIP, or other original
  providers.
- Authentication tokens, credentials, or gated-model access material.

These materials remain governed by their source licenses and access terms.

## Regeneration expectation

A full reproduction requires acquiring the public research datasets, obtaining provider checkpoints,
rebuilding or validating the leakage-audited protocols, extracting all seven frozen representations,
and running the configured downstream and statistical pipeline. `REPRODUCIBILITY.md` gives the
execution order. No cached embedding availability is assumed.
