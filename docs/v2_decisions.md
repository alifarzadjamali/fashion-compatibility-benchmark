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
- **Owner-approved G3 amendment:** original v1 split files/hashes and prediction
  artifacts are no longer available anywhere; only paper/README numbers survive.
  Owner explicitly approved replacing original-hash identity with two independent
  matching deterministic rebuilds, published outfit-count checks and ResNet50/CLIP
  reproduction against the published v1 values at the handoff tolerances. Label
  this revised G3 throughout; never claim original split identity was verified.
  Disclose this limitation in the v2 protocol and manuscript deviation table.
- HF authentication is present at the DGX default credential path
  `/data/a_jamali2/.cache/huggingface/token`. Access to the pinned gated DINOv3
  config was verified successfully without exposing the credential. The first
  unauthenticated failures are retained; retry only affected models.
- Owner requests **fewer than 15 commits per day**. Batch related code and evidence
  changes; do not create a commit for every improvement. One agent-created commit
  has been pushed so far in this booking.
- Owner can extend the booking. Provisional Phase 1 estimate communicated:
  16–24 hours, recommending 24 hours total initially. This includes remaining
  engineering and validation, not just GPU execution; it is not a G6-derived
  full-study estimate. Reassess after acquisition and revised G3. Continue using
  actual `timeleft` until an extension is reflected by the server. Phase 2 remains
  unauthorized regardless of booking extension.

## 2026-11-18 — revised G3 run status (booking job 34)

- Code on the DGX checkout was byte-compared (rsync checksum dry run) against
  the local workspace before commit: identical for src/scripts/tests/configs.
- Polyvore-D-Clean: two deterministic rebuilds hash-identical; counts
  16,991/2,999/15,142 match published; clean-protocol audit PASS. ResNet50
  PCA-256 reproduction PASS (ΔCP AUC 1.8e-6, ΔFITB 6.5e-4). Polyvore source is
  the Kaggle archive `yiitcandeime/polyvore` (sha256 03f00b12…, checksum first
  established now). Its identity to the v1 copy is supported by count and metric
  reproduction, not by an original hash; disclose this.
- IQON3000-Clean: two rebuilds hash-identical; counts 86,800/12,400/24,800 match
  published; 434,089 verified images extracted.
- Measured G3 extraction rates on GB10: ResNet50 ≈ 464 img/s, CLIP-L/336 ≈
  79 img/s (v1 precision, end-to-end incl. decode). IQON CLIP (~434k images)
  needs ≈ 90 min, which did not fit the remaining booking; owner chose to extend
  the booking rather than defer IQON CLIP.
- A100: the public SharePoint link returned a 59 KB HTML page (sha256
  1a796786…), not the archive. The owner's v1 copy was transferred instead and
  verified on the DGX: `data/a100_audit/A100.zip` sha256 31773883…f829, matching
  the recorded v1 hash. The failed HTML `.part` file is retained as evidence.
  Record that the public link is not a reliable acquisition path.
- Booking job 34 extended by owner; END_TIME 2026-11-18T21:04:43 (server).
  POG go/skip not yet tested. G4 (remaining: POG decision, checksum record),
  G5, G6 and G7 are NOT started. G5/G7 are not expected to fit this booking.

## 2026-11-18 — POG go/skip probe (G4)

- POG metadata downloaded from the README Google Drive folder without manual
  steps (gdown 5.2.0): item_data.txt.zip 426 MB, outfit_data.txt.zip 88 MB,
  user_data.txt.7z 8.71 GB. 1,013,136 outfits; 583,464 items used by outfits;
  75 categories; every used item has a pic_url.
- Probe `scripts/v2/pog_probe.py`: seed 20261118, 1,000 items sampled from items
  used by outfits, 8 req/s cap, generic UA, magic-byte validation, no images
  stored. Target = 124,000 random outfits (IQON-Clean size) → 184,339 items.
- First run (`probe.json`, retained) reported SKIP at 81.4% success. All 186
  failures were a probe defect, not dead links: scheme-relative URLs
  (`//img.alicdn.com/...`) raised ValueError before any request was made. All
  814 absolute URLs succeeded. Fixed by resolving `//` to `http:` (the scheme
  of all other POG URLs) and re-ran with the same seed and sample (`probe-r2.json`).
  The go/skip criterion itself was not changed.
- Some CDN responses are WebP although URLs end in .png; per-image SHA-256 of
  the received bytes is the identity, not the URL suffix.
- Booking job 34 extended again: at 16:21 server time, 14:43:26 left
  (ends ≈ 07:04 on 2026-11-19).
- Corrected probe (`probe-r2.json`): **GO** — success 99.6% (3×HTTP 400,
  1×HTTP 404), 7.95 req/s, 184,339 target items → projected 6.4 h and ≈73 GB.
  3.4% of responses were WebP. Remaining G4 issue: POG images must survive the
  booking wipe (no redistribution via GitHub); a private mirror location is an
  owner decision. Final success on required items must be ≥95% or POG is skipped.
- Owner approved POG image acquisition plus a private byte-identical mirror on
  the owner's laptop (not GitHub). `scripts/v2/pog_download.py` (same seed and
  target as the probe; resumable JSONL manifest with per-image SHA-256; atomic
  writes) started 16:29 server time; projected finish ≈ 23:30. Measured
  laptop↔DGX link: 18.6 MB/s down, 48.7 MB/s up, so a ≈73 GB restore next
  booking takes ≈25 min instead of a ≈6.5 h CDN re-download.
