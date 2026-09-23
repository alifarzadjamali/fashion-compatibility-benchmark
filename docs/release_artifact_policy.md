# Public release artifact policy

The public repository is a reproducibility package, not a redistribution bundle.

Included:

- source, scripts, exact configuration, and tests;
- protocol manifests and hashes without split payloads;
- model/provenance and integrity audit summaries;
- final reports and publication-ready summary tables/figures;
- manifests describing the complete private experimental package.

Excluded:

- source datasets, derived split payloads, item/image files, or annotations;
- virtual environments and package/model caches;
- embeddings, checkpoints, fitted PCA/scalers/classifiers, and raw predictions;
- logs, temporary files, credentials, tokens, and internal handoff notes;
- redundant intermediate and provisional generated artifacts.

The private artifact manifest may list files absent from the public repository. This is intentional:
it provides an auditable inventory and hashes for the completed run without publishing restricted or
large data products.

## Release review

Before publishing a release, verify that the public tree contains the expected
source, configuration, tests, manifests, and report artifacts. Re-run the
credential and oversized-file scans, confirm that generated private outputs are
still ignored, and compare the final artifact manifest with the release tag.
