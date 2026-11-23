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
- **Owner decision (2026-11-18 17:02):** POG final-acquisition skip threshold
  relaxed from the handoff's 95% to **90%** success on required items. This is a
  pre-freeze deviation from HANDOFF §4 and goes in the deviation table. The
  running download was started with the 95% label; its report is re-judged at 90%.
  Observed so far: 0.38% failures (57/15,000). Items lost drop whole outfits, so
  the POG retention / selection-effect audit must report outfits lost.

## 2026-11-18 — POG clean protocol builder (G4) — OWNER DECISIONS NEEDED

`src/repbench/v2/pog.py`, `scripts/v2/build_pog_protocol.py` (two identical builds,
all audits pass on the real metadata; final rebuild with the image manifest pending).
- **Retention:** of the 124,000 target outfits, 78,725 (63.5%) survive the split:
  479 duplicate tuples, 44,704 packing conflicts (one item component holds 95.5%
  of outfits, so whole-component assignment is impossible), 92 in tiny categories.
  Splits: train 56,548 / valid 7,144 / test 15,033 (71.8/9.1/19.1%). Zero cross-split
  item/image/outfit overlap; zero collisions. Selection effect: dropped outfits use
  more popular (hub) items (median reuse 7 vs 3) and are slightly longer.
  Rule kept from IQON (seeded order, discard outfits touching two splits); only
  change: assign to the split furthest below its 70/10/20 share.
  **Decision A:** accept ~79k POG outfits, or enlarge the image target (more downloads).
- **Grouping:** only 12.5% of target outfits are ever clicked in user_data, so
  user groups are ~singletons (largest user group: 6 outfits) — a user-cluster
  bootstrap on POG is not a real dependence correction. Test's largest item
  component holds 47% of test outfits (same issue as v1 IQON, HANDOFF §6.5).
  **Decision B (recommended):** POG primary = two-way user × item-component
  bootstrap (or Louvain sub-communities of the item graph); report user-only as
  sensitivity. Needs owner sign-off before freeze.
- **Decision C:** same-category negatives use raw `cate_id` (74 categories);
  categories with < 10 distinct images inside a split are excluded (92 outfits).
- Seeds: construction 20260912 (as IQON); question seeds 20260912 + 10,000×(1,2,3).

## 2026-11-18 — G6 speed test PASS and full-run projection

- `scripts/v2/g6_speed.py` (via `run_g6.sh`): 5,000 seeded Polyvore-Clean *train*
  images per encoder, v2 settings (bf16, SDPA), threaded CPU preprocessing overlapping
  GPU. All 15 encoders measured; all GPU-bound. End-to-end img/s: ResNet50 712,
  DINOv3-L 228, DINOv3-L@336 103, CLIP-L/336 97, PE-Core-L 80, SigLIP2-So400m 72,
  Qwen3-VL-Emb-2B 50, Qwen3-VL-Emb-8B 21 (others in `research/v2/readiness/2026-11-18-g6/g6/`).
  Concurrent load during the test: POG CDN download (network-bound, ~1 core).
- **Projected extraction (771,213 images per encoder: Polyvore 152,785 + IQON
  434,089 + POG 184,339): 28.9 GPU-hours** for all 15 encoders; Qwen-8B 10.1 h and
  Qwen-2B 4.3 h are half of it.
- **Projected full Phase 2 run:** extraction ≈ 29 h + zero-shot MLLM (≈6,200
  questions; s/question from G5) + OutfitTransformer (2 lr × 5 seeds × 3 inputs ×
  3 datasets) + profiling ≈ 1–2 h; probes are CPU and overlap extraction.
  Planning figure: **≈ 40 GPU-hours, ≈ 2–2.5 days wall clock**, refined after G5.
  Extraction is resumable (`extract_v2.py` state files) but caches do not survive a
  booking wipe, so each booking must finish whole encoder × dataset units.

## 2026-11-18 — G1/G2 re-run (a04) and G5 dry run PASS

- New v2 code synced; G1 (env + pytest incl. all new tests) and G2 (15 encoders +
  MLLM, determinism) re-run as attempt `2026-11-18-a04`: both PASS. G5 binds to a04.
- **G5 PASS** (`research/v2/readiness/2026-11-18-g5/g5/gate.json`): 22 stages, no
  manual steps, ~65 min: 2% train/valid subsample of Polyvore-Clean and
  IQON3000-Clean × {ResNet50, SigLIP2-So400m, Qwen3-VL-Emb-2B} → v2 extraction →
  probes (native/PCA-128/256/512 × logistic/XGBoost/MLP + operator sensitivity) →
  paired cluster bootstrap/Holm/DeLong/McNemar/TOST/rank intervals → temperature
  calibration → Friedman/Nemenyi → generated tables/macros/figures → number check;
  OutfitTransformer v2 (lr grid, 2 seeds, short), zero-shot Qwen3-VL-8B (24 valid
  questions; 0.36 s/question; letter mass > 0.997), GPU profiler (3 encoders),
  Polyvore near-duplicate audit, environment capture. Test split never read for
  scoring. POG joins the dry run after its image download completes.
- Dry-run development numbers (2% valid, not for the paper): native/logistic CP AUC
  Polyvore 0.685 / 0.865 / 0.832, IQON 0.591 / 0.729 / 0.723 (ResNet50 / SigLIP2-So400m
  / Qwen-2B) — tier ordering as expected.
- **Finding (owner decision D):** IQON Qwen-2B PCA-128 logistic kept selecting C at
  the lower grid edge after 4 automatic extensions (recorded in failures.csv, not
  hidden). Native selections sit at C=1e-4–1e-3. Recommend extending the frozen
  grid downward to 1e-6 and re-checking on full train data before freeze.
- **Finding (owner decision E):** Polyvore-Clean near-duplicate audit (pHash
  Hamming ≤ 4): 6,459 cross-split image pairs; **19.1% of test outfits** contain an
  item with a near-duplicate in train/valid. pHash on white-background product shots
  may over-flag similar plain items, so before freeze: visually inspect a seeded
  sample of ~100 pairs to calibrate the threshold, and add the embedding-cosine
  criterion. The exclusion-sensitivity analysis is pre-registered either way.
- MLLM timing implies ≈ 40 min for ~6,200 frozen questions in Phase 2.
- **Decision E evidence (visual check, 2026-11-18 18:40):** in a seeded sample of 10
  Hamming-0 cross-split pairs (385 such pairs in total), 7 are the same product
  photo in train and test (re-encoded/resized copies), 3 are false positives
  (low-texture item vs different item; same blazer in another colour; different
  belts). So Polyvore-Clean still contains genuine cross-split image near-duplicates
  that v1's exact-byte / decoded-pixel disjointness did not catch — a substantive
  new validity finding for v2. Recommended rule for the freeze: near-duplicate =
  pHash Hamming ≤ 2 AND embedding cosine ≥ 0.95 (colour-sensitive); primary results
  on the protocol as built, with the pre-registered exclusion sensitivity, OR rebuild
  splits merging near-duplicate images into one component (owner choice).
  Review sheet with 100 seeded pairs (Hamming 0/2/4) for the owner:
  `~/repbench-mirror/near_dup_review_polyvore.pdf` (contains dataset images; never commit).

## 2026-11-19 — POG acquisition and final split; overnight stall

- The agent session stalled overnight on a permission prompt (~12 h idle DGX GPU
  after 21:40). Lesson: approve tool permissions for unattended runs beforehand.
- POG images: **PASS, 183,806 / 184,339 verified (99.71%)**, 72.98 GB (png 177,921,
  webp 5,750, jpeg 135); failures 322×HTTP 400, 208×HTTP 404, 3×URL-encoding
  (non-ASCII URL; fixable). Manifest sha256 0e4d669e…; target-outfits sha256 e4d40789….
  The report labels the rule 95% (job started before the owner's 90% revision); it
  passes either way.
- POG clean protocol rebuilt on the DGX with the image manifest (exact-image
  identity = SHA-256 of image bytes) and user clicks: two builds hash-identical, all
  audits pass. **train 54,626 / valid 7,276 / test 14,947 (76,849 outfits)**; zero
  cross-split item/image/outfit overlap; largest item component 95.6% of pool.
  Evidence: `research/v2/readiness/2026-11-19-pog/`.
- Full-data development extraction (TRAIN+VALID only) of Polyvore-Clean for all 15
  encoders completed 18:34–21:41 (≈ 3.1 h, consistent with G6).
- **Bug found by the full-data run (not by the 2% dry run):** 9 / 33,982
  Polyvore-Clean train CP outfits list the same item twice; the v2 pair operator
  rejects repeated items (v1 silently included the self-pair). Fix in
  `src/repbench/v2/probe.py`: order-preserving de-duplication within an outfit (a
  repeated item adds no pairwise information). Valid/test contain none. This changes
  code after G2 a04, so G1/G2 must be re-run at the next booking start (standard).
- **Finding (owner decision F, 2026-11-19):** Qwen3-VL-Embedding cost scales with
  image resolution (dynamic resolution, 4–1,800 visual tokens/image). Measured
  rates: Polyvore ≈ 50 img/s, IQON ≈ 124 img/s, **POG ≈ 7 img/s** (large PNGs).
  G6 timed Polyvore only, so the projection under-estimates POG: Qwen-2B ≈ 7 h and
  Qwen-8B ≈ 17 h on POG alone → full run ≈ 60+ GPU-h, not ≈ 40. Options before
  freeze: (1) cap Qwen `max_pixels` uniformly (e.g. 256 visual tokens ≈ the
  384-px budget of the other encoders) — fairer and ~7× cheaper on POG, a
  documented deviation from the provider default; (2) keep the provider default
  and budget the GPU time. Recommendation: (1), plus a provider-default spot check
  on a subsample to show the effect.

## 2026-11-19 — OWNER DECISIONS for the freeze (approved by owner, ~12:10 server time)

- **A** Accept POG-Clean retention (~77k of 124k target outfits); disclose retention
  and the hub-item selection effect.
- **B / grouping principle (all datasets):** cluster on every available dependence
  source. IQON and POG: two-way user × item-component bootstrap (primary); user-only
  as sensitivity. Polyvore (no user data): item-component bootstrap.
- **C** POG same-category = raw `cate_id`; categories with < 10 distinct images in a
  split excluded.
- **D** Logistic C grid extended downward to 1e-6 (upper 1e4), automatic validation-
  only extension retained and reported.
- **E** **Rebuild all three clean protocols as "v2-Clean"** with near-duplicate
  images merged into one indivisible unit (cannot cross splits). Rule is
  model-free (no roster encoder decides splits): pHash Hamming ≤ 2 AND a pixel-level
  colour/shape agreement check, calibrated on a labelled seeded sample. Applied
  identically to Polyvore, IQON and POG. v1-Clean splits are kept only as the
  comparison ("leakage inflation") analysis; G3 remains the v1 reproduction evidence.
- **F** **Uniform input budget: no model sees more than 384×384 pixels.** Fixed-
  resolution encoders use their native 224–384 px processors (unchanged); the three
  dynamic-resolution Qwen models (Emb-2B, Emb-8B, zero-shot 8B) are capped at
  max_pixels = 147,456 (= 384², 144 visual tokens) on all datasets. Provider-default
  spot check on a seeded subsample reported in the supplement.
- **Image decoding (all datasets):** images with alpha/transparency are composited onto
  white before any model or audit (no-op for JPEG; affects POG PNGs, which otherwise
  decode with arbitrary/black backgrounds — a hidden cross-dataset domain shift).
- Per-model preprocessing (crop vs resize) follows each official processor and is
  tabulated in the paper.

## 2026-11-19 — Decision E implemented: v2-Clean near-duplicate equivalence (code ready; DGX build pending)

- **Rule (model-free, identical for all three datasets), `scripts/v2/image_equivalence.py`:**
  two images of the whole pre-split item pool are one unit iff pHash Hamming ≤ 2
  (exact multi-index search, functions imported from `near_duplicate_audit.py`) AND
  foreground-mean CIELAB Δa*b* ≤ 2.0 AND the 95th percentile of per-cell CIELAB ΔE over
  the union foreground of 16×16 thumbnails ≤ 15.0 (white-composited 32×32 BOX
  thumbnails; foreground = min channel < 245). Union-find closure; group id = smallest
  item id. Aspect-ratio check implemented but **off** (all Polyvore images are 300×300,
  so it cannot be calibrated). Note: this pHash's Hamming distances are always even.
- **Calibration** (`research/v2/readiness/2026-11-19-image-equivalence/polyvore_calibration.json`,
  ids + labels + metrics; labelled blind by the agent from side-by-side sheets, owner
  seed included): plain whole-thumbnail MAD could not separate the owner's FALSE pairs
  (navy/black blazer 5.6, belts 6.9) from TRUE re-encoded copies (up to 6.7); the
  colour (Lab) + local-ΔE rule can. Fit set 124 pairs (owner seed, MAD-stratified,
  two boundary rounds): precision 1.00, recall 0.81 (misses are boundary cases: shifted
  crops, one recoloured copy). Owner seed 10/10 correct. **Held-out uniform random 36
  pairs labelled after thresholds were fixed: precision 0.87 (2 FP: two thin gold
  necklaces; pearl bracelet gold vs silver clasp), recall 1.00.** False merges only make
  splits stricter.
- Polyvore pool (152,785 images, from the zip on the laptop): 2,963 candidate pairs →
  1,257 confirmed; 2,359 items in 1,150 groups (largest 6); input sha256 c12591f3….
  Simulation with the v1 component/assignment code: v1-Clean split has **553 equivalence
  groups spanning splits**; v2 has 0 with identical outfit counts 16,991/2,999/15,142.
- **Builders:** optional `image_equivalence` (+ `image_equivalence_info`, `protocol_name`)
  added to `construct_clean_protocol`, `construct_iqon_clean`, `construct_pog_clean`;
  `None` reproduces the pre-change output byte for byte (golden-hash test on fixtures).
  Groups join the exact-image unit (Polyvore components; IQON/POG merged identity
  `near:<smallest>`), and are used for CP/FITB candidate exclusion (new for Polyvore).
- **Build:** `scripts/v2/build_v2_clean.py --dataset {polyvore,iqon3000,pog}_v2_clean`
  builds twice, requires identical hashes, runs existing audits + uniform audit + "no
  group / merged identity spans splits"; manifest records equivalence file sha256,
  report sha256, thresholds, input hash. IQON v2 selects different outfits, so images of
  newly selected items must be extracted (`scripts/v2/extract_iqon_images.py`, resumes).
- Tests: `tests/test_v2_image_equivalence.py` (+ fixtures `tests/v2_clean_fixtures.py`).

## 2026-11-19 — A100 v2 path (implementation decisions)

- A100 prepared with v1 `prepare_a100` unchanged (LAT 100, AAT 100 questions, 5
  candidates each, 1,663 images, all JPEG; LAT archive/majority disagreements at
  questions 92–94, 96, 98–100; no majority ties; label files keep v1 sha256).
- A100 is embedded and scored only after the freeze (guarded). Each corpus-trained
  primary probe (native/logistic) is re-fitted through the identical code path and
  seed and must reproduce the pipeline's chosen C and validation AUC
  (`--verify-pipeline-results`), then scores A100; A100 is never used in fitting.
- Statistics: question bootstrap; LAT-majority primary, LAT-archive and mLAT
  sensitivities, AAT overall + 6 facets; Holm within corpus × setting × learner ×
  task × metric; TOST margins 2/5/10 pp. The zero-shot MLLM is its own family (CIs);
  cross-corpus comparisons of the same encoder are descriptive only.
- A100 MLLM question ids frozen: all 100 LAT + all 100 AAT.

## 2026-11-19 — Sensitivity-analysis technical defaults (agent-level decisions)

- Failure consensus: CP "correct" at probability ≥ 0.5 (v1 rule).
- Construction seeds 42–46: total uncertainty = mean bootstrap variance + between-seed
  variance, normal CI (Rubin's variant also reported).
- Near-duplicate exclusion: primary = drop affected test outfits (`any_method`), item-
  level exclusion as secondary; audits re-run on each v2-Clean split with pHash and
  embedding cosine ≥ 0.95 using DINOv3-L (pre-specified; self-supervised image model).
- LookBench v3 correlation: scores to be transcribed from the LookBench v3 paper into
  `configs/lookbench_v3_scores.json` (exact-match models only) — CPU-only, Phase 3.
- FITB pairwise comparisons now include Cohen's h (metric-name mismatch fixed).

## 2026-11-19 — Independent pre-freeze review: findings and fixes

An independent reviewer read the Phase 2 code and reproduced each finding before
reporting. All confirmed issues were fixed before the freeze:
1. A100 stage could not read Phase 2 split stores (trainvalid/ + test/) → shared loader
   now merges verified parts.
2. A100 probe re-fit used the dry-run carve-out instead of the Phase 2 rule (train on
   TRAIN, select on VALID) and would have aborted verification → same rule as pipeline.
3. Pre-freeze TRAIN+VALID chain existed only on the DGX → `scripts/v2/run_full_trainvalid.sh`.
4. Protocol promised Platt scaling and cluster-adjusted McNemar that the code did not
   compute → both now computed (calibration rows per method; `mcnemar_cluster_p`).
5. Stale carve-out wording → protocol regenerated from code.
Robustness: lanes record failures (`$F/failed/`) instead of stalling or silently
skipping; OutfitTransformer always runs ResNet50 even if selection is incomplete;
prediction files carry provenance (eval split, selection split, store hash) checked
at the stats stage; Polyvore item components merge v2-Clean near-duplicate groups
(verified against the protocol manifest's SHA-256); extraction finalisation edge
cases handled. Calibration ECE CIs use primary clusters only (documented).

## 2026-11-19 — G7 FREEZE (owner sign-off)

- Owner signed off the frozen protocol. `configs/v2_protocol.yaml`
  sha256 34c92beff6a81bbbf06a35e1307968abf916086a07fd865b1e77c9a895ca4bda
  (`configs/v2_protocol_hash.json`); tag `v2-freeze`. No v2 test-split, POG-test or
  A100 outcome had been computed or viewed before this point (test extraction and
  scoring are code-guarded on this hash). Phase 2 authorized by the owner.

## 2026-11-20 — Phase 2 deviation D1 (code/config bug, no science change)

- Found by a downstream rehearsal (stats + all sensitivities on finished IQON
  predictions, 100 replicates, scratch checkout `~/fcb-dev`, scratch output dir) run
  before the live CPU lane reached those stages.
- Bug: `configs/v2_sensitivity_datasets.json` carried a top-level `"_note"` string;
  `run_sensitivities.py near-duplicates` iterates all config entries and crashed on it
  (`AttributeError: 'str' object has no attribute 'get'`). Would have aborted the
  near-duplicate-exclusion and LookBench sensitivities in the live run.
- Fix: removed the `"_note"` key (comment only; no path or value changed). Applied to
  the run checkout `~/fcb-freeze` at 2026-11-20 ~13:45 UTC, before the CPU lane's
  sensitivity stage. No `.py` file changed: the G1/G2 code-hash bindings (a08) remain
  valid (verified with `require_gate`). The frozen protocol and its hash are unchanged.
- Verified: every sensitivity step (label fractions, construction seeds, provenance,
  subgroups, near duplicates, LookBench) plus stats, tables and the Polyvore
  near-duplicate audit completed in the rehearsal with empty failure lists.

## 2026-11-21 — Phase 2 deviation D2 (infrastructure, no science change)

- The Slurm booking ended unexpectedly on 2026-11-20 at ~16:48 (job 34; ssh refused by
  `pam_slurm_adopt`). The node storage was wiped. At that point the run had finished
  all embedding extraction (15 encoders × 3 datasets + A100), the MLLM zero-shot
  scoring and 44 of 45 probe units; stats, sensitivities, OutfitTransformer and
  profiling had not started.
- Restored on a new booking (job 37, 2026-11-21) from the laptop copies, which were
  SHA-256 verified against the DGX before the loss (`SHA256SUMS_emb.txt`): the frozen code
  at commit 240bd85 (protocol hash `34c92bef…a4bda`, unchanged), all embeddings,
  protocols, image-equivalence groups, MLLM outputs and the DGX-origin probe predictions.
- G1/G2 re-run on job 37 with the same code (attempt `2026-11-21-a09`, PASS); the evidence
  is committed under `research/v2/readiness/2026-11-21-a09/`. The run is bound to a09.
- The 14 probe units without restored DGX predictions (13 Polyvore encoders, one of which
  had an incomplete copy, plus IQON Qwen3-VL-Emb-8B) are re-fitted on job 37 with the
  same frozen code and commands (`scripts/v2/run_phase2b_cpu.sh`, the probe + stats part of
  `run_full_cpu.sh`). A laptop re-fit of the same units (same code) is kept as a
  cross-machine check only.
- Orchestration only: sensitivity subcommands run as parallel processes instead of one
  serial `run_sensitivities.py all` call (same subcommands, arguments and outputs).
- No frozen setting, split, model, grid, seed or statistic changed.

### 2026-11-23 — D2 addendum: probe deadlock on job 37 (infrastructure, no science change)

- Sat 2026-11-21 12:17 the IQON3000 Qwen3-VL-Embedding-8B probe unit (`--workers 6`) hung in a
  forked-pool deadlock after 3 of 14 files (the same failure seen on the laptop for Polyvore
  Qwen-8B, 2026-11-20). Processes stayed alive at 0% CPU; nothing was recorded in `failed/`, and
  the monitoring loop did not tick, so it went undetected until Mon 2026-11-23 09:36 (~45 h lost).
- Fix (code-bug-only, no `.py` change): killed the hung processes and restarted
  `run_phase2b_cpu.sh` with `WORKERS=1`; completed cells are skipped by the existing logic.
- The IQON preparation catalog (wiped with the node) is rebuilt with the README protocol-
  construction scripts (`scripts/v2/run_phase2b_iqoncat.sh`); count fields are compared with the
  committed audits in `research/v2/readiness/2026-11-18-g3/g3/`.
- 649 DGX outputs (results, done markers, logs) copied to the laptop and SHA-256 verified.
