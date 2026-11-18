# v2 decision log

## 2026-11-18 — first readiness booking

- Owner authorizes Phase 1 only. G3 may reproduce the **unchanged v1** ResNet50
  and CLIP PCA-256 test results. This exception does not permit new-setting or
  POG test evaluation. G5 will use train/validation and synthetic A100 fixtures.
- Initial `timeleft`: 2:59:29. Do not assume a 48-hour allocation. Reserve at
  least 30 minutes for preservation and integrity checks. Gates remain ordered;
  preparatory implementation is not evidence that a gate passed.
- Official guide checked: https://spark.salford.cloud/guide. Allocation is 18
  CPU cores, **100 GB memory**, personal data quota 1 TB. Host is ARM64 GB10;
  Python 3.12.3, uv 0.13.9, tmux and Apptainer available. Shared model and image
  inventories are checked before downloading. `/data/shared/models` was empty.
- Use `/data/a_jamali2/repbench` for this booking's data, caches and logs. All
  of these remain ephemeral. No cross-booking cache reuse is assumed.
- Preserve original environment files in `configs/v1_environment/`; keep the
  frozen v1 protocol and original seven-entry registry unchanged. Separate v2
  entry points must not silently change v1 settings.
- Try pinned PyTorch 2.11.0 / torchvision 0.26.0 CUDA 13.0 wheels on Python 3.12;
  actual build and GPU checks determine suitability, not package availability.
- No HF token was present initially. Gated DINOv3 access and original v1 split
  hashes/reference artifacts requested from owner. Do not substitute rebuilt
  hashes for original evidence.
- Compact evidence will be tracked in `research/v2/`; raw data, weights,
  embeddings and credentials remain ignored. GitHub push works from the local
  workspace; unattended remote upload authentication must be verified before
  relying on a laptop-independent experiment.
- G1 initial clean build passed (20 tests passed, 11 data-dependent tests skipped).
  Cold setup plus checks: 45.095 s; lock resolution: 0.370 s. After adding v2
  foundational tests and Qwen dependency, attempt a02 passed G1 with 29 tests
  passing, 11 skipped. The skipped tests are not data-readiness evidence.
- New v2 backends use uniform bf16; v1 backends retain their original precision.
  GR-Lite's pinned remote source was inspected. It implements eager attention;
  v2 substitutes mathematically equivalent torch SDPA, with a CPU equivalence
  test. This is recorded explicitly and must receive bf16/fp32 spot checks.
  Its source docstring says DINOv2 while the handoff identifies DINOv3; verify
  actual upstream parent provenance before making the matched-parent claim.
- Qwen image adapter follows the pinned provider's instruction, 4–1800 visual
  tokens' pixel bounds (4*32² to 1800*32²), and last-valid-token pooling. Unlike
  the reference's NULL fallback, preprocessing failures stop the job. G2's MLLM
  smoke tests next-token probabilities; an MLLM is not an image embedder.
- G2 uses synthetic inputs only. Source/lock hashes bind gate evidence; code
  changes invalidate earlier evidence. Reports are not overwritten. Include
  both Qwen embedding sizes for readiness; optional scope has not been cut.
- GitHub documentation checked 2026-11-18: ordinary Git warns above 50 MiB and
  rejects above 100 MiB; releases permit up to 1000 assets, each under 2 GiB,
  and state no total-release-size or bandwidth limit. Use a conservative
  <2 GB per asset if releases become necessary. LFS is not enabled: account
  storage/bandwidth allowance and billing have not been established. Release
  limits do not grant permission to redistribute datasets or weights.
  Sources: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
  and https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Public A100 source currently links to PolyU SharePoint; IQON links to Google
  Drive. Added checksum-verifying archive acquisition, with resumable partial
  files and numeric transfer evidence only. Polyvore source identity/checksum
  remains unresolved; do not substitute a similarly named Kaggle dataset.
