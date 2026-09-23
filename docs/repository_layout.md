# Repository layout

This guide helps reviewers and contributors locate repository material without changing the locked
benchmark workflow.

| Path | Purpose |
| --- | --- |
| `configs/` | Versioned dataset, encoder, and experiment definitions. |
| `data/protocols/` | Committed manifests describing derived protocol construction; source datasets are not included. |
| `src/repbench/` | Python package for data validation, frozen encoders, features, models, and evaluation helpers. |
| `scripts/` | Explicit command-line entry points for preparation, evaluation, audits, and artifact assembly. |
| `tests/` | Unit and protocol-validation tests; data-dependent checks skip without authorized inputs. |
| `artifacts/` | Committed audit summaries and integrity inventories, not generated embeddings or raw data. |
| `reports/` | Curated tables, figures, and reviewer-facing summaries from the frozen release. |
| `paper*/` | Publication-source packages and their supporting material. |
| `docs/` | Protocol, maintenance, decision-log, release, and contributor documentation. |

For execution order, use `REPRODUCIBILITY.md`. For the boundary between committed evidence and
regeneration-required material, use `ARTIFACTS.md`.
