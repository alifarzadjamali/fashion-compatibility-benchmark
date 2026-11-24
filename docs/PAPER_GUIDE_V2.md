# RepBench-Fashion v2 — Paper-writing guide

Status: **COMPLETE** (2026-11-23, after the full Phase 2 run on DGX booking job 37; updated
2026-11-24 after the post-run sweep: rows D4–D13 and O1, new evidence in §1, numbers re-checked). Every
number, table and figure comes from a generated file listed here; every provenance claim points
to a hash in a committed file.

---

## ★ Narrative priority (owner directive, 23 Nov 2026 — this overrides everything below)

**The paper presents RepBench-Fashion v2 as a strong, rigorous contribution, and that is the only
priority of the writing.** Lead with what was built and what it shows. Write every section in
confident, active voice. Present design choices as strengths. Do not apologise, hedge or
self-criticise in the Abstract, Introduction, Method, Results or Discussion. Caveats belong in
one place: a short, forward-looking **Limitations** section (§3 row "Limitations"). Run history
(the booking loss, restarts, bug fixes) never appears in the paper body. It lives in the
repository and, where a reviewer needs it, in the supplementary protocol record (§6).

**Contributions to headline (Intro bullet list, in this order):**

1. **A pre-registered, leakage-controlled compatibility benchmark** (claim "first"/"most
   rigorous" only after the related-work check confirms it). The protocol was frozen and
   hashed before any test evaluation (`v2-freeze`, sha256 `34c92bef…a4bda`). The pipeline
   follows fit on TRAIN, select on VALID, and touches TEST once. Splits are separated by item and
   by exact/near-duplicate image (pHash ≤ 2 with pixel confirmation; merged groups never cross
   splits), and they are published with hashes. Every reported number is machine-generated:
   1,170 probe metrics plus 108 headline macros, checked by `check_paper_numbers.py`.
   On the older, leakier v1-Clean splits every one of 15 encoders posts a higher Polyvore
   CP-AUC (mean +0.0023; IQON about +0.0075) with the same ranking, so the v2-Clean numbers are
   the stricter ones (descriptive comparison: the test sets differ; FITB differences are mixed).
2. **A broad, modern encoder roster.** 15 frozen encoders spanning supervised CNNs,
   self-supervised models, generic VLMs, fashion VLMs, fashion retrieval and MLLM embeddings, all
   on 3 datasets (Polyvore, IQON3000 and POG, the last new to this line of work). Matched
   parent/child pairs isolate the effect of fashion fine-tuning.
3. **A clear answer to "does fashion-specific pretraining help?"** Large general-purpose
   encoders lead on every dataset: generic VLMs SigLIP2-So400m and PE-Core, plus the Qwen3-VL
   MLLM embedding, which tops IQON CP-AUC. SigLIP2-So400m is the validation-selected generic encoder on
   all three datasets. Fashion VLMs do not beat them, and pure-vision models trail. Holm-corrected
   paired cluster-bootstrap tests back this up: 230 of 315 primary CP-AUC comparisons and 172 of
   315 primary FITB comparisons are significant (4,929 of 8,190 across all settings).
4. **Compatibility is its own capability.** Fashion-retrieval leaderboard rank (LookBench v3)
   does not predict compatibility rank (ρ between −0.14 and +0.14). Compatibility therefore needs
   its own benchmark, which is this work's reason to exist.
5. **The findings carry over to a trained compatibility model.** OutfitTransformer on the
   selected encoders reaches a mean test CP-AUC of 0.942 / 0.894 / 0.829 on Polyvore / IQON / POG,
   with the generic encoder > fashion encoder > ResNet-50 ordering on all three. The ranking is
   stable across probe settings, label fractions, construction seeds and provenance groups, and
   stays highly concordant when every test outfit touched by a looser near-duplicate audit is
   removed (Kendall τ 0.83–0.98; SigLIP2-So400m leads CP-AUC on all three datasets after exclusion).
6. **Reproducibility as a first-class result.** Laptop (x86) and DGX (ARM) runs agree to a
   median |ΔAUC| of 1.5e-8 on Polyvore (196 probe runs; primary probes ≤ 3.9e-8 with identical selections; largest gap
   0.0034, on an MLP sensitivity probe). bf16 inference matches fp32 (mean embedding cosine
   ≥ 0.998 on all checked encoders and datasets). The v1 results are reproduced within 0.001 AUC. Code, splits, hashes
   and all small result files are released.

**Wording rules:**
- Use positive, factual framing: "stricter near-duplicate-aware splits", "cluster-aware
  statistics give honest intervals", "20,000 bootstrap replicates for fine-grained Holm
  p-values".
- Each result sentence states a finding, then its evidence. Avoid "unfortunately", "however we
  could not", "a limitation is" outside the Limitations section.
- Limitations section: one paragraph, written as scope and future work. Cover proxy labels;
  image-only input; frozen features by design; POG click coverage; LookBench overlap n = 6.
  End on future directions.
- Statistics sentence for Methods: "Holm-adjusted p-values use 20,000 paired cluster-bootstrap
  replicates for p-value resolution. The pre-registered 2,000-replicate analysis gives identical
  estimates and model selection and is released alongside." Use this sentence as written.
  Add the Holm family in one clause: "Holm is applied to the 105 model pairs within each dataset,
  metric and probe configuration; pooling all configurations per dataset and metric with DeLong
  and exact McNemar p-values (which ignore clustering) leaves 489 of 630 primary comparisons
  significant." These are the
  only protocol notes the body needs.

## 0. Ground rules for writing

1. **Never type a number.** Use `\input{numbers.tex}` macros (generated by
   `scripts/v2/make_tables_v2.py`); `scripts/v2/check_paper_numbers.py` fails the build on
   hand-typed values. Headline values outside the probe tables (OutfitTransformer, Holm counts,
   LookBench, cross-machine, MLLM, A100) are in `numbers_extra.tex` from
   `scripts/v2/make_assets_v2.py`; run the checker with `--extra numbers_extra.json`
   (24 Nov: 1,278 generated numbers, 0 violations). The current `paper_elsevier_ivc/` sources
   are still the v1 text; the checker flags their v1 numbers until the v2 rewrite replaces them.
2. **Figures and tables are generated** (`make_figures_v2.py`, `make_tables_v2.py`). If a
   figure needs a change, change the script and regenerate (§7).
3. **Model choices quoted in the text** (best generic / best fashion) come from
   `selection.json` — validation-only, never test.
4. **Protocol record** (§6) goes in the supplementary material / repository only, neutral and
   brief. The paper body carries only the one statistics sentence above.
5. **Statistics wording.** CIs, ranks and effect sizes: either result set (identical point
   estimates). Holm-adjusted bootstrap p-values: quote from `results_r20k/` and label as D3.
6. Venue: Elsevier IVC (mirror to Springer MVA). Manuscript sources:
   `paper_elsevier_ivc/`, `paper_springer_mva/`. HANDOFF_V2.md §10 lists the editorial
   changes, citation fixes and the submission package.

---

## 1. Where everything lives

Laptop root: `~/repbench-results/` (index: `~/repbench-results/README.md`).
**Canonical run outputs:** `~/repbench-results/dgx37/final/v2/full/` (written `$FF` below),
verified by `~/repbench-results/dgx37/final/SHA256SUMS_final.txt` (2,653 files after the 24 Nov sweep).
Compact copies of every small result file are committed under `research/v2/results/`.

| What | Laptop path | Repo path (GitHub `v2-readiness`) |
|---|---|---|
| Frozen protocol | — | `configs/v2_protocol.yaml` (+ `.txt` twin) |
| Protocol hash | — | `configs/v2_protocol_hash.json` (sha256 `34c92bef…a4bda`), git tag `v2-freeze` |
| Owner decisions + deviations | — | `docs/v2_decisions.md` |
| Run log / recovery | — | `docs/RESUME_V2.md` |
| Gate evidence G1–G7 | — | `research/v2/readiness/<date>-<attempt>/`; the Phase 2 run is bound to G2 **a09** (`research/v2/readiness/2026-11-21-a09/g2/gate.json`) |
| Split build / equivalence reports, MLLM ids, A100 manifest | — | `research/v2/freeze/` |
| Embeddings (15 encoders × 3 datasets + A100) | `~/repbench-results/emb/` (`cd ~/repbench-results/emb && sha256sum -c ../SHA256SUMS_emb.txt`; DGX copy re-verified identical 297/297) | not in repo (size) |
| Probe predictions (630 files) | `$FF/results/predictions/` | not in repo (size) |
| Main stats, 2,000 replicates (pre-registered) | `$FF/results/*.csv`, `friedman.json` | `research/v2/results/results/` |
| Main stats, 20,000 replicates (D3) | `$FF/results_r20k/` | `research/v2/results/results_r20k/` |
| Combined view used for tables/figures | `$FF/results_final/` (symlinks: r20k stats + predictions + sensitivities) | — |
| Validation-only selection | `$FF/results_r20k/selection.json` (identical in `results/`) | `research/v2/results/results_r20k/selection.json` |
| Sensitivities (label fractions, construction seeds, provenance, subgroups, near-duplicates, LookBench) | `$FF/results/sensitivities/<name>/` | `research/v2/results/results/sensitivities/` (csv/json) |
| MLLM zero-shot | `$FF/mllm/<set>/summary.json`, `results.jsonl` | `research/v2/results/mllm/*/summary.json` |
| OutfitTransformer (9 streams × 2 lr × 5 seeds + test eval) | `$FF/outfit_transformer/<dataset>/<encoder>/` (weights incl.) | `research/v2/results/outfit_transformer/` (metrics, histories, selections; no weights) |
| A100 scores / summary | `$FF/a100_scores/`, `$FF/a100_summary/` | `research/v2/results/a100_summary/` |
| Efficiency profile (15 encoders) | `$FF/profile/<model>.json` | `research/v2/results/profile/` |
| Near-duplicate audits | `$FF/near_duplicates/<dataset>/` | `research/v2/results/near_duplicates/` |
| Generated tables + macros (final, 20k) | `$FF/generated_final/` | `research/v2/results/generated_final/` |
| Generated tables (pre-registered 2,000) | `$FF/generated_r2000_prereg/` | `research/v2/results/generated_r2000_prereg/` |
| Figures (8, PDF + PNG) | `$FF/figures_final/` | `research/v2/results/figures_final/` (PDF + manifest) |
| Cross-machine check | `~/repbench-results/dgx37/cross_machine_polyvore.json` | `research/v2/readiness/2026-11-21-a09/cross_machine/` |
| IQON preparation audits (rebuilt) | `~/repbench-results/dgx37/final/v2/iqon_preparation/` | counts compared with `research/v2/readiness/2026-11-18-g3/g3/iqon_*` |
| Split protocols (v2-Clean, A100) | `~/repbench-results/protocols/` | build reports in `research/v2/freeze/` |
| Dataset archives (private) | `~/repbench-mirror/` + `SHA256SUMS` | never committed |
| **Post-run sweep (24 Nov)** | | |
| bf16/fp32 and Qwen pixel-cap spot checks | `$FF/spot_checks/` | `research/v2/results/spot_checks/` |
| Qwen GFLOPs (eager attention) | `$FF/profile_qwen_flops/` | `research/v2/results/profile_qwen_flops/` |
| Construction seeds with near-duplicate groups (D4) | `$FF/sens_eqfix/construction_seeds/` | `research/v2/results/sens_eqfix/` |
| User-only grouping sensitivity, IQON/POG (D8) | `$FF/results_useronly_r20k/` | `research/v2/results/results_useronly_r20k/` |
| A100 summary at 20,000 replicates (D9) | `$FF/a100_summary_r20k/` | `research/v2/results/a100_summary_r20k/` |
| Paired MLLM vs probes on the same 2,000 FITB ids (D10) | `$FF/mllm_paired/` | `research/v2/results/mllm_paired/` |
| v1-Clean vs v2-Clean leakage inflation (D11) | `$FF/leakage_inflation/`; v1-Clean splits + IQON stores `dgx37/final/v2/v1clean/` | `research/v2/results/leakage_inflation/` |
| Holm family variant (D12) | `$FF/holm_family_variant/` | `research/v2/results/holm_family_variant/` |
| Paper assets: efficiency table, feature-width table, `numbers_extra.tex`, qualitative FITB figure | `$FF/generated_assets/` | `research/v2/results/generated_assets/` |

## 2. Hashes to quote (provenance paragraph)

| Item | Where to read it |
|---|---|
| Protocol | `configs/v2_protocol_hash.json` |
| Code at run time | G2 a09 `gate.json` → `input_sha256` (every `.py`, `pyproject.toml`, `uv.lock`) |
| Model checkpoints | `configs/v2_protocol.yaml` → `roster.encoders.*.revision` (HF commit) |
| Dataset archives | `configs/v2_protocol.yaml` → `datasets.archives_sha256`; `~/repbench-mirror/SHA256SUMS` |
| Splits | `research/v2/freeze/*_build_report.json` (`file_sha256`) |
| Near-duplicate groups | `research/v2/freeze/*_equivalence_report.json` |
| MLLM question sets | `configs/v2_protocol.yaml` → `baselines.zero_shot_mllm.frozen_question_ids` |
| Embedding stores | `metadata.json` in each store + `SHA256SUMS_emb.txt` |
| Each prediction file | provenance block inside the JSON (eval/selection split, items_sha256) |
| Each figure | `figures_manifest.json` (input SHA-256 per figure) |
| All run outputs | `~/repbench-results/dgx37/final/SHA256SUMS_final.txt` |

## 3. Section → evidence map

| Paper section | Content | Evidence / generator |
|---|---|---|
| Abstract | problem, method, 3 numeric findings, implication | `numbers.tex` |
| Intro / RQ1–4 | RQ list + 4 findings | main table, matched pairs, rank intervals |
| Related work | HANDOFF §10 citation list; LookBench v3 (arXiv:2601.14706) | — |
| Benchmark & data | 3 v2-Clean datasets + A100; counts; grouping; near-dup merging | build reports; equivalence reports |
| Protocol | frozen before test; fit TRAIN / select VALID / test once | `v2_protocol.yaml`; decisions log |
| Models | 15 encoders, families, matched parents, provenance group, feature widths | protocol roster; `registry.py`; `table_feature_width.tex` |
| Probe | native primary; PCA/operator/learner sensitivities | `table_settings.tex`, `fig_settings` |
| Statistics | cluster bootstrap, Holm, DeLong, McNemar, TOST, ranks, Friedman | `pairwise.csv`, `equivalence.csv`, `rank_intervals.csv`, `friedman.json` |
| Results RQ1 (ranking) | | `table_main.tex`, `fig_ranking`, `fig_rank_intervals` |
| Results RQ2 (fashion vs generic) | | `fig_matched_pairs`, pairwise rows |
| Results RQ3 (robustness) | settings, label fractions, construction seeds, provenance, near-dup exclusion, user-only clusters, v1-Clean leakage inflation | `fig_settings`, `fig_label_fractions`, `fig_construction_seeds` (CIs from `sens_eqfix/`), sensitivities CSVs, `results_useronly_r20k/`, `leakage_inflation/` |
| Results RQ4 (external validity) | A100 LAT/AAT; MLLM zero-shot (paired with probes on the same questions); OutfitTransformer; LookBench | `a100_summary_r20k/`, `mllm_paired/`, `fig_mllm_vs_probes` (descriptive), OT outputs, `sensitivities/lookbench/` |
| Calibration | temperature/Platt on VALID, ECE, Brier decomposition | `calibration.csv`, `fig_calibration` |
| Efficiency | params, GFLOPs, latency, memory, energy, kgCO2e | `table_efficiency.tex` (`generated_assets/`) |
| Qualitative | FITB cases + unanimous failures | `fig_qualitative_fitb.pdf` + manifest (`generated_assets/`) |
| Limitations (the only place for caveats; one forward-looking paragraph) | scope: proxy labels; image-only; frozen features by design; POG click coverage 12.5%; LookBench overlap n = 6 → future work | decisions B; protocol |
| Reproducibility | hashes (§2); laptop-vs-DGX check; v1 reproduction (G3) | §5, `cross_machine/` |

Key results already visible (for orientation; cite the macros, not these):
- Linear-probe CP-AUC leaders: Polyvore SigLIP2-So400m / PE-Core; IQON Qwen3-VL-Emb-8B /
  SigLIP2-So400m; POG PE-Core / SigLIP2-So400m. ResNet-50 lowest everywhere; pure-vision models
  (DINOv3, GR-Lite) trail vision-language models; fashion VLMs do not beat strong generic VLMs.
- OutfitTransformer (5-seed mean test CP-AUC) exceeds the linear probe on the same encoder in
  CP-AUC on every dataset (on Polyvore FITB the probe is ahead for the generic and fashion encoder):
  Polyvore 0.942 / 0.934 / 0.829, IQON 0.894 / 0.882 / 0.787, POG 0.829 / 0.810 / 0.697
  (generic / fashion / ResNet-50).
- LookBench v3 retrieval rank vs compatibility rank: Spearman ρ −0.14…+0.14 (n = 6): no
  evidence that retrieval benchmarks predict compatibility.
- Near-duplicate exclusion (every test outfit flagged by the looser pHash ≤ 4 / cosine audit
  removed) keeps rankings highly concordant (Kendall τ 0.83–0.98); SigLIP2-So400m becomes the
  CP-AUC leader on IQON and POG as well (PE-Core still leads POG FITB).
- Zero-shot Qwen3-VL-8B vs probes on the same 2,000 FITB questions: on Polyvore 11 of 15 probes
  are Holm-significantly better and 1 is worse; on IQON 4; on POG none differ (`mllm_paired/`).

## 4. Frozen-protocol summary (for the Methods section)

- **Roster.** 15 frozen image encoders at pinned HF revisions (supervised CNN, self-supervised,
  generic VLM, fashion VLM, fashion retrieval, MLLM embedding), with matched parent pairs
  (FashionCLIP2↔OpenCLIP-B/32-LAION2B, Marqo-FashionSigLIP↔SigLIP-B/16, GR-Lite↔DINOv3@336) and a
  provenance-uncertain group; Qwen3-VL-8B-Instruct (bf16) as a zero-shot baseline, not a probe.
- **Inputs.** Alpha composited onto white; at most 384×384 pixels per image (official processors
  for fixed-resolution encoders; Qwen min/max pixels 4,096/147,456); uniform bf16 + SDPA; seed
  20260912, deterministic cuBLAS, no cuDNN benchmark.
- **Datasets.** Polyvore-D, IQON3000 and POG v2-Clean (item- and exact/near-duplicate-image
  separated splits; near-duplicates merged into indivisible groups by pHash ≤ 2 + pixel check),
  plus A100 (evaluation only). Archive SHA-256 values in the protocol. Clusters: Polyvore item
  component; IQON and POG two-way user × item component (user-only as sensitivity).
- **Probe.** Native features primary: train-item StandardScaler → L2 → mean pair [|zᵢ−zⱼ|; zᵢ⊙zⱼ]
  → train-outfit StandardScaler → logistic regression, C grid 1e-6…1e4 (validation-only edge
  extension). Sensitivities: PCA 128/256/512, operator with zᵢ+zⱼ, XGBoost and MLP grids, label
  fractions 10/25/50/100%, construction seeds 42–46. FITB = argmax probability, ties → lowest index.
- **Selection.** Fit on TRAIN; C, hyperparameters and post-hoc calibration on VALID; TEST once.
  Best generic/fashion encoder = highest VALID native CP-AUC per dataset (code, no hard-coding).
- **Statistics.** Paired cluster bootstrap (2,000 replicates pre-registered; 20,000 for Holm
  p-values, D3), seed 20260912; all pairwise comparisons Holm within dataset × metric; DeLong for
  CP; exact and cluster sign-flip McNemar for FITB; TOST margins (CP-AUC 0.002/0.005/0.01; FITB
  0.5/1/2 pp; A100 2/5/10 pp); bootstrap rank intervals, Friedman + Nemenyi (descriptive), Kendall W;
  deltas with CIs and Cohen h.
- **Calibration.** Brier, log-loss, ECE (10/15/20 bins); temperature and Platt fitted on VALID;
  Murphy decomposition; ECE CIs by cluster bootstrap.
- **Baselines.** OutfitTransformer on ResNet-50 + selected generic + selected fashion encoder
  (lr 1e-4/3e-4, ≤100 epochs, patience 10, seeds 42–46); zero-shot MLLM by option-letter
  log-probability with seeded option order on all A100 + 2,000 FITB per clean test set.
- **Efficiency.** `profile_efficiency_v2.py` on the DGX GB10 with nothing else running: params,
  GFLOPs, GPU-only latency (median + IQR, batch 1/64), allocator peak, separate CPU
  preprocessing, energy, storage.

## 5. v1 → v2 improvements (for the Introduction / Discussion)

Source table: `docs/RESUME_V2.md` §6 (review issue → v2 fix). Headline items: matched
parent pairs; native features primary; one coded validation-only selection rule; all
numbers generated; correct clustered bootstrap; like-for-like efficiency; modern roster
incl. MLLM embeddings and zero-shot MLLM; tuned OutfitTransformer; full statistics and
calibration; near-duplicate merging; POG as a third dataset; hashed, published splits;
cross-machine reproducibility check (laptop x86 vs DGX ARM: median |ΔAUC| 1.5e-8).

**Reproducibility of v1 (G3), one line:** "v2 splits are rebuilt deterministically and
published with SHA-256 hashes; rebuilding v1 with the same procedure reproduces its
published metrics within 0.001 AUC (v1's original split files were not retained)."
Keep it to that — a strength, with one neutral clause.

## 6. Protocol record (supplementary material / repository — not the paper body)

A neutral record for reviewers who want it. In the body, H1–H5 appear only as design choices
in Methods (stricter splits, uniform pixel budget, cluster-aware grouping). D1–D2d have no
effect on any number and stay in the repo; D4–D13 are post-run sweep items (24 Nov). D3 is covered by the one statistics sentence in
★ above.

| ID | Deviation | Type | Impact |
|---|---|---|---|
| H1 | G3 revised (v1 split identity unverifiable) | pre-freeze, owner | none on v2 |
| H2 | POG acquisition threshold 90% (handoff 95%) | pre-freeze, owner | achieved 99.71% |
| H3 | Qwen pixel cap 384×384 | pre-freeze, owner (decision F) | uniform budget |
| H4 | v2-Clean merges near-duplicate images | pre-freeze, owner (decision E) | stricter splits |
| H5 | IQON grouping two-way user × item-component | pre-freeze, owner (decision B) | wider, honest CIs |
| D1 | Sensitivity config `_note` key removed (crash fix) | Phase 2 bug fix | none |
| D2 | Booking lost 2026-11-20; node wiped; restored on job 37 from SHA-verified copies; G1/G2 re-run (a09 PASS); 14 probe units recomputed (laptop re-fit kept as cross-check) | infrastructure | none (cross-machine median \|ΔAUC\| 1.5e-8) |
| D2a | IQON Qwen-8B probe forked-pool deadlock; re-run with 1 worker | infrastructure | none |
| D2b | IQON preparation catalog rebuilt with the README scripts | infrastructure | counts match original audits 21/21 |
| D2c | LookBench CLI parse bug; same library function called directly | code bug | none (identical to `all` path) |
| D2d | OutfitTransformer run as 9 parallel streams (identical arguments; OMP threads 2) | scheduling | none |
| D3 | Main pairwise bootstrap at 20,000 replicates (protocol: 2,000), owner-approved after the freeze because the 2,000-replicate Holm floor (≈0.0525) exceeded α | statistics, post-freeze | p-value resolution only; estimates and selection identical; CIs essentially unchanged (max \|Δ\| 0.0027 for metric CIs, 0.0031 for pairwise CIs); 2,000-replicate results kept and reported |
| D4 | Construction-seed protocols did not carry the near-duplicate groups, so that sensitivity's bootstrap ignored the merge; fixed in `sensitivities.py`, rerun with byte-identical questions and reused predictions | code bug, post-run sweep | point estimates and seed SDs unchanged; bootstrap variance ratio new/old median 1.01 (0.77–1.45) |
| D5 | bf16 vs fp32 spot check (HANDOFF §5; GR-Lite SDPA substitution), 3 encoders × 3 datasets, 512 seeded TRAIN/VALID items | promised check, run post-run | mean cosine ≥ 0.998 (0.999 for all but GR-Lite on IQON); re-embedding reproduces stored vectors (cosine 1.0000) |
| D6 | Qwen 384×384 cap vs provider default (decision F), 2 Qwen encoders × 3 datasets, 256 items | promised check, run post-run | Polyvore/IQON images are below the cap (identical tokens; cosine 1.0 / ≥0.999); POG 99% of images capped, embedding cosine 0.94 — reported as a disclosed design choice |
| D7 | Qwen GFLOPs: FlopCounterMode SDPA rule rejects GQA shapes; counted with the equivalent eager attention | measurement fix | 2B 516.1, 8B 1,743.7 GFLOPs/image |
| D8 | User-only grouping sensitivity for IQON/POG (protocol `sensitivity: user_only`) was not run in Phase 2; run post-run at 20,000 replicates | promised analysis | estimates and selection identical; user-only CIs ≈0.44× as wide (median, metric CIs; 0.48× pairwise), so the two-way primary analysis is the conservative one |
| D9 | A100 summary had the same 2,000-replicate Holm floor as D3; recomputed at 20,000 | statistics | 195 of 3,150 A100 comparisons Holm-significant (0 at 2,000) |
| D10 | MLLM vs probe figure compared different question sets; paired comparison on the frozen 2,000 ids added | analysis fix | figure kept as descriptive; claims use `mllm_paired/` |
| D11 | v1-Clean leakage-inflation analysis (protocol) run post-run: v1-Clean splits rebuilt (counts match G3), IQON items missing from v2 stores embedded on the GPU before the booking ended (3 encoders); the rebuild called the library functions directly because the G2 gate refuses after the D4 code change; lee-research Polyvore-D cross-check: see row | promised analysis | Polyvore CP-AUC inflation +0.0023 (15/15 encoders up), IQON ≈ +0.0075 (2 encoders; ResNet-50 out of memory on the laptop); rankings agree (Spearman 0.97); lee-research data unavailable, count-level overlap matches (3,781/84/34) |
| D12 | Holm family: protocol says "within dataset × metric"; run applied it per dataset × metric × probe configuration (105 pairs). Pooling all configurations: bootstrap p-values cannot reach α (floor 5e-5 × 1,365), DeLong/exact-McNemar Holm (unclustered) keeps 489/630 primary comparisons significant | interpretation, disclosed | reported counts stay the clustered as-run ones |
| D13 | Record items: G2 a08 (job 34) evidence not retained (a09 input hashes = v2-freeze + make_figures_v2.py, verified); `make_figures_v2.py` added after the freeze (reporting only); ECE CIs use user clusters for IQON/POG (protocol `:375`); A100 manifest label said "carve-out selection" (Phase 2 used VALID; label fixed) | record | none |
| O1 | Owner directive 23 Nov: the deviation table lives in the supplement / repository, the paper body carries the statistics sentence (amends HANDOFF hard line 4) | owner | — |
| P3 | LookBench v3 scores transcribed from arXiv:2601.14706 Table 3 (6 exact matches) | planned Phase 3 input (not in protocol hash) | LookBench sensitivity now populated |

## 7. Regenerating everything (laptop, no DGX needed)

```bash
FF=~/repbench-results/dgx37/final/v2/full
.venv/bin/python scripts/v2/make_tables_v2.py --results $FF/results_final --out $FF/generated_final
.venv/bin/python scripts/v2/make_tables_v2.py --results $FF/results_r2000_view --out $FF/generated_r2000_prereg
.venv/bin/python scripts/v2/make_figures_v2.py --results $FF/results_final --out $FF/figures_final \
  --mllm $FF/mllm/*/
.venv/bin/python scripts/v2/make_assets_v2.py --ff $FF --out $FF/generated_assets
.venv/bin/python scripts/v2/check_paper_numbers.py --generated $FF/generated_final \
  --extra $FF/generated_assets/numbers_extra.json
sha256sum -c ~/repbench-results/dgx37/final/SHA256SUMS_final.txt   # run from dgx37/final
```
Probes/statistics can be re-run on the laptop from `~/repbench-results/emb` and
`~/repbench-results/protocols` (verified: E3, |ΔAUC| 2e-6; cross-machine check above).

## 8. What remains (owner)

All analyses, tables, figures and macros are generated. Remaining work is writing only, by the
owner: the v2 manuscript text in `paper_elsevier_ivc/` (currently v1 text), citations
(HANDOFF §10, incl. LookBench arXiv:2601.14706), highlights, graphical abstract, cover letter.
`v2-readiness` is merged into `main` at the end of the sweep (v1 preserved at tag `v1-final`).
Macro note: `resHolmSigPrimary` / `resHolmComparisonsPrimary` are the CP-AUC primary block (230/315);
the FITB block has its own `...Fitb...` macros.
