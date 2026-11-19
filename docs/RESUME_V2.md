# RepBench v2 — resume here

**Living status file.** Updated at every gate change and pushed to GitHub. A new
agent session reads `HANDOFF_V2.md` (goal and rules), then this file (where we
are, how to restart), then `docs/v2_decisions.md` (why). Do not rely on chat
history.

Last updated: 2026-11-19 12:00 (DGX server time), booking job 34 (ends 2026-11-20 13:04).

## 1. Gate status

| Gate | Status | Evidence |
|---|---|---|
| G1 Environment | PASS (re-run on current code) | `research/v2/readiness/2026-11-18-a04/g1/` |
| G2 Models (15 encoders + MLLM, pinned) | PASS (re-run on current code) | `research/v2/readiness/2026-11-18-a04/g2/gate.json` |
| G3 Reproduce v1 (**revised**, owner-approved) | **PASS** — both datasets: identical rebuilds, published counts; ResNet50 + CLIP within tolerance (max abs Δ 0.00065) | `research/v2/readiness/2026-11-18-g3/g3/gate.json` |
| G4 Data | **PASS** — all 7 archives SHA-256 verified on DGX and laptop mirror; POG GO (99.71% images), clean split 54,626/7,276/14,947, two identical builds, zero cross-split overlap | `research/v2/readiness/2026-11-19-pog/gate.json` |
| G5 Pipeline dry run | **PASS** (Polyvore + IQON, 3 encoders, 22 stages); POG to be added after image download | `research/v2/readiness/2026-11-18-g5/g5/gate.json` |
| G6 Speed test | **PASS** — all 15 encoders; projected extraction 28.9 GPU-h, full run ≈ 40 GPU-h | `research/v2/readiness/2026-11-18-g6/g6/gate.json` |
| G7 Freeze | NOT STARTED — needs owner decisions (§4) | — |

Phase 2 (full run) is **not authorized**. It requires all gates PASS, the
`v2-freeze` tag, and explicit owner approval.

Revised G3 = two identical deterministic rebuilds + published counts + published
metrics within ±0.002 AUC / ±0.005 FITB. Original v1 split identity is
UNVERIFIABLE (original files lost); disclose in the paper.

## 2. Where things live

| What | Where | Durable? |
|---|---|---|
| Code, configs, compact evidence | GitHub branch `v2-readiness` | yes |
| Dataset archives (private mirror) | owner laptop `~/repbench-mirror/` with `SHA256SUMS` | yes, owner laptop only |
| Model weights (53 GB) | re-downloaded from HF at pinned revisions | re-fetch each booking |
| Everything on the DGX (`/data/a_jamali2/repbench`, tmux) | DGX | **wiped at booking expiry** |

Expected archive SHA-256 (verify after every copy):

| Archive | DGX path (under `/data/a_jamali2/repbench/data/`) | SHA-256 |
|---|---|---|
| Polyvore (Kaggle `yiitcandeime/polyvore` v1) | `raw/polyvore_outfits.zip` | `03f00b1253dff35c2e9431e84aad9d9a02969f57e57d05975dee38906141f74e` |
| IQON3000 | `iqon_audit/IQON3000.zip` | `305660cc09bd941a15096b6efeb536013efe0d534d9cd517419a49cd9ee650fd` |
| A100 | `a100_audit/A100.zip` | `31773883ab6fcf9c0de8182b7561f7319a1d928d542906b8c3c4b8674196f829` |
| POG item_data.txt.zip | `raw/pog/` | `e3ead4e42a04eb4bf74786474983e9b9e255cc13a400130756d08ce2e0883aa3` |
| POG outfit_data.txt.zip | `raw/pog/` | `9818caf2a1323d2cf7c32a4be7924121bb83b0c7dbcf5d7854b62df94da13a47` |
| POG user_data.txt.7z | `raw/pog/` | `e4f1d8203b9ea52fea86aa8ca850ff273a77f2255cf6c2050b8a38a271be8237` |
| POG images (tar of `mirror/`, 183,806 images + manifest) | `raw/pog/pog_images.tar` → extract to `raw/pog/` | `8b091ed982d503bec260d1533413eb6292e8e0161eb89dc0dee5aefdc520d2c6` |

The public A100 SharePoint link returns an HTML page, not the archive; use the mirror.

## 3. Fresh booking: restart procedure

Run Claude Code **on the owner laptop** (it holds the mirror and pushes to GitHub;
the DGX has no GitHub write credential by owner decision).

1. Owner: book the DGX; on the DGX run once
   `HF_HOME=/data/$USER/repbench/cache/huggingface uv run --frozen hf auth login`
   (gated DINOv3). Kaggle access is not needed when the mirror is used.
2. Agent: `ssh a_jamali2@spark.salford.cloud`, check `timeleft`, clone branch
   `v2-readiness` into `~/fashion-compatibility-benchmark`.
3. Agent: copy archives from `~/repbench-mirror/` to the DGX paths above
   (≈30 min at 48.7 MB/s) and verify every SHA-256 before use.
4. Agent: rebuild the environment and re-run G1/G2 with a new attempt name
   (`scripts/v2/run_readiness.sh <attempt>`), which re-downloads pinned models.
5. Agent: continue from the first gate that is not PASS in §1.
6. Agent: push evidence from the laptop after each gate; verify with
   `scripts/v2/verify_preservation.py`. Max 14 commits per calendar day.

## 4. Open owner decisions (needed before G7 freeze)

Detailed in `docs/v2_decisions.md` (Decisions A–F): POG retention (A), POG grouping (B), POG categories (C), C-grid lower bound (D), near-duplicate threshold (E), Qwen image-resolution cap (F). Also:

- Is Qwen3-VL-Embedding in the "uncertain provenance" group (HANDOFF §3)?
- POG `cate_id` → "same-category" mapping for negatives (75 categories in use).
- Confirm POG target = 124,000 outfits sampled with seed 20261118 (current
  acquisition), before the split builder drops any outfits.
- Approve Phase 2 after G7.

## 5. Next actions (in order)

2. When POG download finishes: verify ≥95% success, archive, copy to laptop
   mirror with SHA-256.
3. Build and run the G6 speed test (5,000 images × ResNet50, CLIP-L,
   SigLIP2-So400m, Qwen3-VL-Emb); log the projected full-run time.
4. Build the G5 pipeline pieces (HANDOFF §7.8–7.9) and run the 2% dry run.
5. Draft `configs/v2_protocol.yaml` for owner review (G7).

## 6. Why v2 is a stronger paper than v1 (review issue → v2 status)

| v1 weakness (review ID) | v2 fix | Status |
|---|---|---|
| Specialization contrasts not parent-matched (P0-1) | True parents in roster: FashionCLIP2↔OpenCLIP-B/32-LAION2B, Marqo-FashionSigLIP↔SigLIP-B/16, GR-Lite↔DINOv3@336 | roster pinned, G2 PASS |
| Claims fragile to feature width; unreported native runs (P0-2) | Native features primary; PCA-128/256/512 + alternative pair operator as sensitivities; XGBoost **and** MLP on all datasets (P1-4) | `src/repbench/v2/probe.py` |
| Two different "best model" sets (P0-3) | One coded validation-only selection rule | `src/repbench/v2/selection.py` |
| Hand-copied tables, small numeric errors (P0-4, P0-11) | All tables/number macros generated; `check_paper_numbers.py` fails on hand-typed values | `scripts/v2/make_tables_v2.py` |
| CIs mislabelled / "conservative" bootstrap wrong (P0-5, P0-6) | Declared grouping per dataset; two-way user × item-component bootstrap; POG grouping problem found and surfaced (Decision B) | `scripts/v2/pipeline.py` |
| Efficiency table not like-for-like (P0-7) | One device, GPU-only CUDA-event latency median+IQR, FLOPs, allocator peak, energy, separate preprocessing | `scripts/v2/profile_efficiency_v2.py`; G6 measured all 15 encoders |
| C at grid edge (P0-8) | Grid {1e-4..1e4} with automatic validation-only extension; failure recorded, never silent | `src/repbench/v2/logistic.py` |
| Dated roster, no MLLM (P1-1, P1-2) | 15 encoders incl. PE-Core, SigLIP2, DINOv3, Qwen3-VL-Embedding 2B/8B; zero-shot Qwen3-VL-8B by option-letter log-prob | G2 PASS; `scripts/v2/mllm_zeroshot.py` |
| Undertrained OutfitTransformer on ResNet only (P1-3) | lr grid, ≤100 epochs, patience 10, 5 seeds, three input encoders, clean split guard | `scripts/v2/run_outfit_transformer_v2.py` |
| Statistics incomplete/non-uniform (P1-6) | All pairwise Holm, DeLong, exact McNemar everywhere, TOST with pre-set margins, rank intervals, Friedman/Nemenyi CD, Kendall W | pipeline + `rank_statistics.py` |
| Shallow calibration (P1-7) | Temperature/Platt on validation, Brier decomposition, clustered ECE CIs | `calibration_v2.py` |
| No near-duplicate audit (P2-4) | pHash (multi-index) + embedding cosine across splits, exclusion list for sensitivity | `scripts/v2/near_duplicate_audit.py` |
| Single replication dataset | **POG added** as blind third dataset (99.6% image availability, ~79k clean outfits) | G4 in progress |
| v1 split provenance unrecorded | Revised G3: two identical rebuilds + published counts + metric reproduction (max abs Δ 0.00065); all archives SHA-256 pinned and mirrored | G3 PASS |

Not yet done (Phase 2 / manuscript): full-run numbers, deviation table, qualitative figure, citation fixes (HANDOFF §10).
