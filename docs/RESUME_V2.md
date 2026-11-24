# RepBench v2 — resume here

**Living status file.** Updated at every gate change and pushed to GitHub. A new
agent session reads `HANDOFF_V2.md` (goal and rules), then this file (where we
are, how to restart), then `docs/v2_decisions.md` (why). Do not rely on chat
history.

Last updated: 2026-11-24 (post-run sweep; GPU work finished 11:16 and DGX fully copied to the laptop — no DGX needed any more).

## 0. Post-run sweep plan (owner CMD 2026-11-23 20:11/20:15; owner writes the paper)

Three independent audits (code, numbers, docs) found no change to any point estimate or
selection. Work items (tick here as done; DGX outputs → `~/repbench-results/dgx37/final/`
with SHA-256; tmux sessions `fcb-spot`, `fcb-fix-*`; runner `scripts/v2/run_sweep_fixes.sh`):

- [x] A1/A2 GPU spot checks (`scripts/v2/spot_checks_v2.py` → `$F/spot_checks`; copied + verified)
- [x] A3 GPU: IQON v1-Clean images + embeddings for 3 encoders (tmux `fcb-v1clean`,
      `scripts/v2/run_v1clean_iqon_gpu.sh` → `$R/v1clean/emb`, marker `$R/v1clean/IQON_EMB_DONE`).
      Copy `$R/v1clean/` (protocols already copied) to `~/repbench-results/dgx37/final/v2/v1clean/`.
      GPU DONE = A3 marker present and copied → post "GPU done — booking can be cancelled".
      If B3 is still running then, copy nothing more and rerun B3 on the laptop instead.
- [x] B3 user-only grouping, IQON/POG, 20k — run ON THE LAPTOP (DGX run abandoned at booking end):
      stats stage of `scripts/v2/pipeline.py` with grouping `user`, `--replicates 20000`, predictions
      from `~/repbench-results/dgx37/final/v2/full/results/predictions`, protocols and equivalence
      groups from `~/repbench-results/protocols/`; output `.../full/results_useronly_r20k/`
- [x] B4 A100 summary at 20k (`$F/a100_summary_r20k`)
- [x] B5 construction seeds with near-duplicate groups, bug fix D4 (`$F/sens_eqfix`)
- [x] B6 MLLM vs probes paired on the same 2,000 FITB ids
- [x] B7 v1-Clean leakage-inflation comparison; lee-research overlap cross-check (or deviation)
- [x] B8 Holm family disclosure + stricter dataset×metric family variant
- [x] C9 PAPER_GUIDE fixes (POG two-way; near-dup top-model changes; OT > probe on CP-AUC
      only; Qwen = MLLM embedding, not generic; D3 "CIs essentially unchanged"; cross-machine
      max 0.0034 on MLP; emb checksum path; 1,170-number claim scope)
- [x] C10 README.md (v2 status, DGX pipeline), this file, v2_decisions (D4+ rows, owner
      writing-directive amendment, a08 evidence gap, make_figures_v2.py post-freeze, ECE CI
      grouping, Qwen spot result)
- [x] C11 tests: pytest pythonpath; lookbench test; A100 provenance label; D4 unit test
- [x] A4 GPU: Qwen GFLOPs (profile null, SDPA flop-rule assertion) counted with eager attention
      → `dgx37/final/v2/full/profile_qwen_flops/` (2B 516.1, 8B 1743.7 GFLOPs/image); use in the
      efficiency table. kgCO2e = energy × a cited grid intensity (CPU arithmetic).
- [x] D12–15 paper assets (generated only; the owner writes all text): efficiency table
      generator, qualitative FITB figure, feature-width table, macros for all headline numbers
- [x] R independent review of ONLY the new work (git diff 7e5995f..HEAD + new outputs):
      one fresh reviewer agent for code, one for new numbers/docs; fix and disclose findings
- [ ] E regenerate tables+figures, check numbers, final SHA-256 copy, commit, PR to main,
      fresh-clone verify, DONE report on issue #2

## 1. Gate status

| Gate | Status | Evidence |
|---|---|---|
| G1 Environment | PASS (re-run on current code) | `research/v2/readiness/2026-11-18-a04/g1/` |
| G2 Models (15 encoders + MLLM, pinned) | PASS (re-run on current code) | `research/v2/readiness/2026-11-18-a04/g2/gate.json` |
| G3 Reproduce v1 (**revised**, owner-approved) | **PASS** — both datasets: identical rebuilds, published counts; ResNet50 + CLIP within tolerance (max abs Δ 0.00065) | `research/v2/readiness/2026-11-18-g3/g3/gate.json` |
| G4 Data | **PASS** — all 7 archives SHA-256 verified on DGX and laptop mirror; POG GO (99.71% images), clean split 54,626/7,276/14,947, two identical builds, zero cross-split overlap | `research/v2/readiness/2026-11-19-pog/gate.json` |
| G5 Pipeline dry run | **PASS** (Polyvore + IQON, 3 encoders, 22 stages); POG to be added after image download | `research/v2/readiness/2026-11-18-g5/g5/gate.json` |
| G6 Speed test | **PASS** — all 15 encoders; projected extraction 28.9 GPU-h, full run ≈ 40 GPU-h | `research/v2/readiness/2026-11-18-g6/g6/gate.json` |
| G7 Freeze | **PASS** — owner sign-off 2026-11-19; protocol sha256 34c92bef…; tag `v2-freeze` | `configs/v2_protocol.yaml`, `configs/v2_protocol_hash.json` |

Phase 2 (full run) **authorized by the owner at the freeze (2026-11-19)**; running per §3b.

Phase 2 progress (server time; started 2026-11-19 15:57 with G2 a08, run checkout `~/fcb-freeze`):
- 22:01 — train/valid extraction done for 13 encoders × 3 v2-Clean datasets (`TV_CHAIN_DONE`);
  copied to laptop `~/repbench-results/emb/`. Test extraction and CPU probes started 22:02/22:05.
  No failures (`$F/failed/` empty).
- 01:58 Fri — test extraction done (13 encoders × 3). Qwen-2B all splits done ~05:25; Qwen-8B
  IQON 07:55, POG 11:32, Polyvore running. Laptop copy: all base + Qwen-2B + IQON Qwen-8B.
- 13:45 Fri — downstream rehearsal (`$R/v2/rehearsal`, scratch) passed stats, tables, all
  sensitivities, Polyvore near-dup audit; found + fixed config bug D1 (see decisions log).
  Measured: stats 2 min/IQON at 100 reps; label fractions ~9.5 min and construction seeds
  ~20 min per model on IQON (serial). Projected finish: probes ~20:00 Fri, stats ~21:30,
  sensitivities ~11:30 Sat, A100/tables/profiling ~13:30 Sat. Booking ends ~Mon 16:00.
- 13:55 Fri — ALL extraction done (incl. Qwen-8B, A100); 14:27–14:32 MLLM zero-shot done (3 FITB
  sets + A100 LAT/AAT). Laptop: all embeddings (SHA-256 verified, `SHA256SUMS_emb.txt`), MLLM,
  protocols, 433 probe predictions (all IQON except Qwen-8B; all POG; Polyvore resnet50).
  E3: laptop re-fit of Polyvore resnet50 matches DGX (test AUC |Δ| 2e-6, FITB identical).
- **16:48 Fri — DGX ACCESS LOST**: ssh refused "pam_slurm_adopt: you have no active jobs on this
  node" (Slurm job no longer active?); still refused at 17:19. Last seen: 44/45 probe units
  done, IQON Qwen-8B probes running; stats/sensitivities/OT/profiling not started.
  Backup started on the laptop (`~/repbench-results/laptop_backup/`, same frozen code,
  separate output dir): Polyvore 14 models + IQON Qwen-8B probe re-fits.
  If the DGX returns: copy `$F/results` first (it holds the post-14:00 predictions).
- **23:46 Fri — laptop backup**: DGX still refused (hourly retries). Polyvore probes **complete
  on the laptop** (14 models × 14 cells = 196 predictions in `laptop_backup/results/`). The
  2-worker forked pool deadlocked on Qwen-8B native logistic/MLP (no CPU for >1 h); killed and
  re-fitted with `--workers 1` (`run2.sh`, rc=0). IQON Qwen-8B cannot run on the laptop: killed
  by the OOM killer twice (peak RSS 28 GB of 30 GB, `run3.log`). It needs the DGX copy or a new
  GPU/large-RAM booking. Laptop now has 44 of 45 probe units (616 of 630 prediction files; all but IQON Qwen-8B).
  Still needs the DGX or a new booking: IQON Qwen-8B probes, near-dup audits, construction-seed
  and provenance sensitivities (need raw images), OutfitTransformer, profiling.
- **10:07 Sat — new booking, job 37** (ssh port 2222, ends Wed 25 Nov 09:44). Old node wiped
  (deviation D2). 10:30 code restored (frozen commit, protocol hash verified), env installed;
  10:37 G1/G2 a09 PASS (13 min, all 15 encoders + MLLM pinned). 10:55 probes + stats lane
  started in tmux `fcb-cpu` (`run_phase2b_cpu.sh`); upload of embeddings and dataset archives
  from the laptop in progress (~45 MB/s). Next: stats → selection.json → OutfitTransformer (GPU).

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

## 3b. Phase 2 (full run) procedure — after the owner signs the freeze

- DGX checkouts: `~/fcb-freeze` = the **run** checkout (code certified by the latest
  G1/G2 attempt; never sync into it while jobs run — `require_gate` rejects changed
  files); `~/fcb-dev` = scratch checkout for preparation/testing;
  `~/fashion-compatibility-benchmark` = Phase 1 history.
- v2-Clean protocols: `$R/v2/{polyvore,iqon3000,pog}_v2_clean/build_a` (+ `build_report.json`);
  A100: `$R/v2/a100_v2`; frozen MLLM ids: `research/v2/freeze/`.
- Freeze: `scripts/v2/write_protocol.py --final --splits ... --question-ids ... --equivalence ...`
  → `configs/v2_protocol.yaml` + hash → owner sign-off → commit + tag `v2-freeze`.
- Run: `scripts/v2/run_full_gpu.sh <g2 gate>` (tmux `repbench-gpu`) and
  `scripts/v2/run_full_cpu.sh` (tmux `repbench-cpu`); outputs under `$R/v2/full/`
  (emb/, results/, mllm/, outfit_transformer/, a100_*, profile/, generated/).
  Both lanes resume safely; `STOP_GPU` file stops the GPU lane between steps.
- Copy `$R/v2/full/` (minus caches) to the laptop `~/repbench-results/` incrementally.

## 4. Open owner decisions (needed before G7 freeze)

**All decided 2026-11-19** (A–F, see `docs/v2_decisions.md`). Formerly: POG retention (A), POG grouping (B), POG categories (C), C-grid lower bound (D), near-duplicate threshold (E), Qwen image-resolution cap (F). Also:

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

- **09:50 Mon 23 Nov — job 37.** IQON Qwen-8B probe unit deadlocked Sat 12:17 (undetected ~45 h,
  see v2_decisions D2 addendum); restarted with 1 worker. 44/45 probe units done, 619 prediction
  files; 649 outputs copied to laptop `~/repbench-results/dgx37` and SHA-256 verified. IQON catalog
  rebuild running (tmux fcb-iqoncat). Next: stats → selection → OutfitTransformer (GPU).
- **10:35 Mon 23 Nov — probes 45/45.** IQON Qwen-8B finished with 1 worker (10:27); all 630
  prediction files present and copied to laptop (`~/repbench-results/dgx37`, SHA-256 verified).
  Paired statistics started 10:29. IQON catalog rebuilt (counts match committed G3 audits 21/21);
  near-duplicate audits for all 3 datasets done and copied. DGX watchdog (`watchdog_phase2b.sh`) live.
- **11:05 Mon 23 Nov — stats done, OutfitTransformer running.** Stats + selection 10:29–10:43
  (1,170 metric rows, 8,190 pairwise rows). OT moved to 9 parallel streams (GPU 13% → 83%).
  Sensitivities: provenance, subgroups, LookBench done; label-fractions, construction-seeds running.
  800 result files copied to laptop and SHA-256 verified.
- **14:20 Mon 23 Nov — Polyvore OT done; 20k-replicate stats done; all sensitivities done.**
  `results_r20k/` (14:07): point estimates identical to the 2,000-replicate run, selection identical;
  Holm-significant pairs 4,929/8,190 (primary block 230/315; was 0 with 2,000 replicates).
  OutfitTransformer: Polyvore 3/3 streams (train+select+test) done; POG 13/30 and IQON 6/30 runs.
  Construction-seeds, label-fractions, provenance, subgroups, near-duplicates, LookBench all done.
  1,002 more files copied to laptop `~/repbench-results/dgx37` and SHA-256 verified.
- **17:00 Mon 23 Nov — POG OutfitTransformer done (Polyvore + POG = 6/9 streams).** Mean test
  CP-AUC over 5 seeds: Polyvore SigLIP2-So400m 0.942, Marqo-FashionCLIP 0.934, ResNet-50 0.829;
  POG SigLIP2-So400m 0.829, Marqo-FashionSigLIP 0.810, ResNet-50 0.697 (all above the
  corresponding linear probes). IQON 17/30 runs, ETA ~23:00–00:30; profiling follows. 606 OT files
  copied to laptop and SHA-256 verified.
- **19:30 Mon 23 Nov — OutfitTransformer COMPLETE (9/9 streams, 90 training runs + test evals).**
  Mean test CP-AUC (5 seeds): Polyvore SigLIP2-So400m 0.942 / Marqo-FashionCLIP 0.934 / ResNet-50
  0.829; IQON 0.894 / Marqo-FashionSigLIP 0.882 / 0.787; POG 0.829 / 0.810 / 0.697. Profiling
  running (11/15 models at 19:29, CPU lane idle). All OT + profile outputs copied to laptop and
  SHA-256 verified (767 files).
- **20:00 Mon 23 Nov — PHASE 2 COMPLETE.** Profiling 15/15 (19:39). Final copy of all DGX
  outputs to `~/repbench-results/dgx37/final/` (2,232 files, SHA-256 verified); DGX embeddings
  re-verified identical to the laptop (297/297). Final tables/figures generated (1,170 numbers,
  0 violations). Compact evidence committed under `research/v2/results/`. Paper guide complete.
  Deviation D3 (20k replicates) recorded with a correction. Booking job 37 left running for the
  owner to cancel (owner instruction 2026-11-23 18:33).
