# Contributing

Thank you for helping improve RepBench-Fashion. This repository preserves a locked benchmark;
changes should make provenance, reproducibility, or review clearer without silently changing an
experimental claim.

## Before opening a change

- Read `REPRODUCIBILITY.md`, `ARTIFACTS.md`, and the relevant protocol documentation.
- Keep generated data, embeddings, model weights, and credentials out of commits.
- Use focused commits with an imperative subject line.
- Explain any effect on protocol definitions, frozen result tables, or publication artifacts.

## Development checks

Create the documented Python 3.12 environment, then run the checks relevant to your change:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Dataset-dependent tests may skip when independently obtained source archives are unavailable. Do
not replace a skipped integration test with fabricated local data.

## Benchmark-affecting changes

Changes to data construction, frozen encoders, feature extraction, evaluation, or reported results
need an accompanying provenance update and reviewer-visible rationale. Do not overwrite committed
summary results in place; record the revision and retain the prior evidence trail.

## Security and data access

Report vulnerabilities through the process in `SECURITY.md`. Respect the source datasets' and model
providers' access terms; neither source images nor access tokens belong in this repository.
