# RepBench-Fashion v2 — Handoff

**For:** an AI coding agent (Claude Code, Codex, or similar) running the v2 study on an **NVIDIA DGX Spark** (GB10, 128 GB unified memory, aarch64, CUDA 13).
**From:** pre-submission review of the v1 IVC manuscript (2026-11-17).
**Repo:** https://github.com/alifarzadjamali/fashion-compatibility-benchmark. The local clone was in sync with `origin/main` at commit `7f5c62b`, and `pytest` passed on Windows.

---

## How to read this

This file sets **the goal, the guardrails, and what we already know**. It is not a script.

- **Use your judgment.** If a library breaks, a URL moves, a model loads differently than described, or there is a cleaner way to do a step, fix it and carry on. Prefer the robust, simple solution.
- **Facts here can be stale.** They were checked in Nov 2026. Verify versions, model cards and URLs before relying on them.
- **Record what you decide.** Keep a short running log in `docs/v2_decisions.md`: date, decision, why. Anything that changes the science goes in the paper's deviation table.
- **Ask the owner** only for decisions that are genuinely theirs. For everything else, decide and log.

### The few hard lines (scientific integrity)

1. **Do not look at test outcomes before the v2 protocol is frozen and hashed.** That covers POG test results, and Polyvore/IQON test results of new models or settings. Development uses train and validation splits only.
2. **Never delete or overwrite v1 artifacts.** Archive them; v1 is disclosed in the paper as a pilot.
3. **A100 is evaluation-only.** No fitting, tuning, calibration or selection on A100.
4. **Every deviation from the frozen v2 protocol is disclosed** in the paper's deviation table, however small.
5. **Paper numbers come from scripts, never hand-typed.**

Everything else is your call.

---

## 1. Goal

Produce a clean, frozen, reproducible **v2** of the study that:

- **keeps every strength of v1** (§2);
- **fixes every issue found in review** (§10 maps each issue to where it is handled);
- uses a **modern roster** with **true parent checkpoints**, **native features as the primary setting**, **uniform statistics**, and **like-for-like efficiency** measured on one device (the DGX);
- adds **POG** as a blind replication dataset if it can be obtained (§4). If not, the study runs without it.

**End state:** v2 results, regenerated tables and figures, an updated IVC manuscript (`paper_elsevier_ivc/`, mirrored to `paper_springer_mva/`), and a tagged repo release.

### Framing of the paper (for context)

The contribution is **evaluation methodology plus a controlled measurement**:
- Leakage-audited protocols.
- Question construction changes scores.
- Under a fixed probe, the **tier ordering** (VLMs > self-supervised ViT > supervised CNN) is robust, but **specialization effects and within-tier ranks are not**.
- Retrieval leaderboards mislead compatibility model selection.

It is **not** a SOTA claim.

---

## 2. Keep from v1 (strengths reviewers will credit)

- Item, exact-image and decoded-pixel disjoint splits built from connected components. The **IQON leakage audit is novel**.
- Collision audit of generated CP negatives and FITB distractors.
- Historical-to-clean **question bridge**, which separates split cleaning from question regeneration.
- IQON retention and selection-effect audit; dependence-aware bootstrap.
- Five construction seeds, label-fraction curves, provenance-restricted sensitivity.
- A100 evaluated unchanged and never trained on; both LAT label interpretations; mLAT.
- Calibration, failure-consensus and outfit-length analyses; LookBench exact-checkpoint comparison.
- Protocol hashing and scripts that refuse to overwrite completed outputs.

---

## 3. Roster v2

**15 frozen encoders and 1 zero-shot MLLM baseline.**
- Pin an exact HF revision (commit hash) for every new entry when you first load it.
- Put the revisions in the registry and the protocol YAML.
- All licences below were checked: none blocks research use.

| # | key (suggested) | Checkpoint | Load via | Res / dim | Licence / access | Role |
|---|---|---|---|---|---|---|
| 1 | `resnet50` | torchvision `IMAGENET1K_V2` | torchvision | 224 / 2048 | BSD | supervised CNN anchor |
| 2 | `dinov3_vitl16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | HF | 224 / 1024 | DINOv3 licence, **gated** (HF token) | self-supervised |
| 3 | `dinov3_vitl16_336` | same checkpoint, run at 336 | HF | 336 / 1024 | same | **true parent of GR-Lite** |
| 4 | `clip_vitl14_336` | `openai/clip-vit-large-patch14-336` | HF | 336 / 768 | MIT | generic VLM (legacy) |
| 5 | `pe_core_l14_336` | `hf-hub:timm/PE-Core-L-14-336` | open_clip (no Meta repo needed) | 336 / 1024 | Apache-2.0 | modern generic VLM |
| 6 | `siglip2_b16_384` | `google/siglip2-base-patch16-384` | HF | 384 / 768 | Apache-2.0 | generic VLM |
| 7 | `siglip2_so400m_384` | `google/siglip2-so400m-patch16-384` (or `-patch14-384`) | HF | 384 / 1152 | Apache-2.0 | scale within family |
| 8 | `siglip_b16_224` | open_clip `ViT-B-16-SigLIP` / `webli` (= `google/siglip-base-patch16-224`) | open_clip | 224 / 768 | Apache-2.0 | **true parent of Marqo-FashionSigLIP** |
| 9 | `openclip_b32_laion2b` | open_clip `ViT-B-32` / `laion2b_s34b_b79k` | open_clip | 224 / 512 | MIT | **true parent of FashionCLIP 2.0** |
| 10 | `fashionclip2` | `patrickjohncyh/fashion-clip` | HF | 224 / 512 | MIT | fashion VLM |
| 11 | `marqo_fashionsiglip` | `Marqo/marqo-fashionSigLIP` | open_clip | 224 / 768 | Apache-2.0 | fashion VLM |
| 12 | `marqo_fashionclip` | `Marqo/marqo-fashionCLIP` (base: ViT-B-16 `laion2b_s34b_b88k`) | open_clip | 224 / 512 | Apache-2.0 | 2nd fashion recipe |
| 13 | `gr_lite` | `srpone/gr-lite` | HF, `trust_remote_code` | 336 / 1024 | Apache-2.0 on HF card vs "DINOv3 licence" in LookBench repo: note both | fashion retrieval |
| 14 | `qwen3_vl_emb_2b` | `Qwen/Qwen3-VL-Embedding-2B` | transformers ≥ 4.57 + `qwen-vl-utils` | 2048 max | Apache-2.0 | MLLM-derived embedder |
| 15 | *(optional)* `qwen3_vl_emb_8b` | `Qwen/Qwen3-VL-Embedding-8B` | same | — | Apache-2.0 | scale; fits on DGX |
| B | `qwen3_vl_8b_zeroshot` | `Qwen/Qwen3-VL-8B-Instruct`, **bf16, no quantization** | transformers | — | Apache-2.0 | zero-shot MLLM baseline (not a probe) |

**Specialization contrasts in v2.** These replace v1's mismatched pairs:
- **Matched-parent pairs:**
  - FashionCLIP2 − OpenCLIP-B/32-LAION2B
  - Marqo-FashionSigLIP − SigLIP-B/16-224
  - GR-Lite − DINOv3@336
  - Marqo-FashionCLIP − its ViT-B/16 LAION parent (optional extra extraction)
- **Practical pairs:** best fashion vs best generic, and best overall vs ResNet50. Both are chosen by a coded validation rule (§6.4).

**Provenance group "uncertain"** (used in the restricted sensitivity analysis): Marqo-FashionSigLIP, Marqo-FashionCLIP and GR-Lite.
- Marqo-FashionCLIP's training data is undisclosed, and Marqo **evaluates on Polyvore**, which is a contamination risk.
- Decide before the freeze whether Qwen3-VL-Embedding belongs in this group. Its training data is web-scale and undisclosed.

**Optional, if time allows:** MetaCLIP 2, C-RADIOv4. Otherwise justify their exclusion in the limitations.

---

## 4. Datasets

| Dataset | Role in v2 | Source | Have it? | Checksum |
|---|---|---|---|---|
| Polyvore Outfits (disjoint) | development | Kaggle archive or original release (≈6 GB, includes images); HF `mvasil/polyvore-outfits` (cc-by-4.0, request access) | ✅ on Windows: `data/raw/polyvore_outfits/` (152,785 images) | **none recorded — add one** |
| IQON3000 | development | Google Drive (README link) | ✅ `.venv/iqon_audit/IQON3000.zip` | `305660cc…650fd` |
| A100 | external human/expert test | AiDLab GitHub (README link) | ✅ `.venv/a100_audit/A100.zip` | `31773883…6f829` |
| **POG (Alibaba iFashion)** | **blind replication (new)** | Metadata on Google Drive via https://github.com/wenyuer/POG; images are **URLs only** (`pic_url`, alicdn) | ❌ | add after download |

**Moving data to the DGX.** Copy the three existing sources from the Windows machine (rsync or a drive) and verify the SHA-256 values on arrival. Also copy v1 `artifacts/` and `reports/` into an archive folder, needed for the v1-vs-v2 check in Gate 3.

### POG: go / skip rule

POG is **valuable but optional**.
- **No licence is stated**, so treat it as research-only and **never redistribute images**. Share item IDs and split files only.
- A 2026 paper (AesRec, arXiv 2602.03416) found only 37 dead image URLs in its subset, so the links mostly work.

1. Download the metadata and fetch a random 1,000 `pic_url`s, rate-limited with a generic user-agent.
2. **Go** if ≥ 97% succeed and the projected full download fits in about 48 h.
3. Choose a target size similar to IQON-Clean (≈124k positive outfits) and download only the items those outfits need. Store SHA-256 per image, and keep a local mirror because the CDN may disappear.
4. **Skip** if the gate fails, or if the final success rate on required items is below 95%. Then v2 runs on Polyvore + IQON + A100, and the paper presents IQON as replication, disclosing that its design was informed by the v1 pilot.

**POG protocol notes:**
- Build the split with the IQON-style connected-component / hypergraph packing builder: items and exact images indivisible, plus the same collision audits.
- `cate_id` is fine-grained; decide (pre-freeze) how categories map to "same-category" negatives.
- Run the retention and selection-effect audit as for IQON.
- **Do not run any model on POG test until the v2 freeze.**

---

## 5. DGX environment

**Target:** one locked, rebuildable environment, recorded in the paper. Facts checked in Nov 2026:
- The DGX Spark is **aarch64, GB10 = compute capability sm_121, CUDA 13 driver**.
- PyTorch's **cu130 aarch64 wheels** support Spark; an sm_121 warning is typically harmless because sm_120 kernels are binary-compatible.
- NGC PyTorch containers ≥ 25.10 support GB10.
- There are **no prebuilt flash-attn wheels for aarch64**. Use PyTorch SDPA everywhere.

**Suggested approach (adapt as needed):**

1. `uv` with Python 3.12 on the host or inside a minimal `nvidia/cuda:13.x-…-ubuntu24.04` (arm64) container. Switch `pyproject.toml` from the `cu128` index to **`cu130`**: change the torch/torchvision pins and the `[[tool.uv.index]]` URL, then relock. The current lock already contains aarch64 `cu128` wheels; if cu130 misbehaves, try those as a fallback.
2. Fallback: an NGC PyTorch container, with project dependencies installed by uv on top (`--system-site-packages`-style), freezing the container digest.
3. Qwen3-VL-Embedding's card pins `torch==2.8.0`. Try the main env first. If outputs differ or it fails, give it a **separate locked env** (`envs/qwen/`) and record that in the paper.
4. XGBoost: aarch64 wheels exist. Use `tree_method="hist"` on **CPU** with fixed `nthread` and seed for determinism.
5. Set `HF_HOME`, `TORCH_HOME` and the data root **outside the venv** (§7, item 1). Store `HF_TOKEN` in the environment (DINOv3 is gated); never commit it.

**Record per run** (`artifacts/v2/env/`):
- `uv.lock` hash and container digest
- `nvidia-smi -q`, driver/CUDA, `torch.__config__.show()`
- `pip list`
- model revisions
- dataset checksums

**Determinism:**
- Set seeds everywhere, plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Turn off cuDNN benchmark.
- Use one uniform inference precision (bf16 suggested) for all encoders. Do an fp32 spot check on 2–3 models to show the effect is negligible.

**Known DGX quirk:**
- With unified memory, `nvidia-smi` may report memory as "Not Supported". Report **peak allocator memory** (`torch.cuda.max_memory_allocated`) and label it as such.
- NVML energy counters may also be unsupported. Fall back to sampling `power.draw` at about 10 Hz and integrating, or to a wall-power meter. Say which method was used.

---

## 6. v2 protocol: what changes from v1 (the fixes)

Write `configs/v2_protocol.yaml` (plus a readable `.txt` twin and a hash file, as in v1). It must specify everything below **before** any test evaluation.

### 6.1 Probe
- **Primary setting: native features.** Train-only StandardScaler, then ℓ2, then pair features `[|zi−zj|; zi⊙zj]`, mean-pooled over pairs, then StandardScaler, then logistic regression.
- **Sensitivities:**
  - PCA-128, PCA-256 and PCA-512, with PCA-256 = the v1 setting kept for continuity;
  - one alternative pair operator (e.g. add `[zi+zj]` or a low-rank bilinear term);
  - XGBoost **and** MLP learners on **both** datasets. v1's protocol named an MLP but the paper reported only XGBoost.
- **C grid** `{1e-4, 1e-3, …, 1e4}`, currently `SHARED_C_GRID` in `src/repbench/models/logistic.py`. Add a check that warns or fails when the selected C sits at a grid edge, and extend the grid if it does. v1 had 6 of 7 IQON models at the lower edge and native runs still improving there.
- Randomized PCA needs a fixed seed. Cite Halko et al. 2011.

### 6.2 Baselines
- **OutfitTransformer (image-only adaptation).** v1 was undertrained: best epoch was 27–30 in all 10 runs and early stopping never fired.
  - Tune on validation only: lr ∈ {1e-4, 3e-4}; ≤ 100 epochs; patience 10; 5 seeds.
  - Train on **three inputs**: ResNet50, best generic and best fashion, the last two chosen by the §6.4 rule. This answers whether contextual learning adds anything on top of good frozen features.
  - Verify it reads the **clean** split mapping. v1's Polyvore config pointed at the historical cache.
- **Zero-shot MLLM** (Qwen3-VL-8B, bf16):
  - Question sets: all 200 A100 questions plus a fixed random 2,000-question FITB subset per clean test set. Freeze the IDs pre-freeze.
  - Score options by next-token log-probability of the option letter instead of parsing free text. Fix and randomize the option order by seed.
  - Report accuracy with bootstrap CIs and cost (s/question, memory).
  - Compare in discussion to TATTOO (GPT-4o + FashionCLIP: A100 LAT 79%, AAT 75%), which is a different, training-free system.

### 6.3 Statistics (one script, uniform across datasets)
- **All pairwise model comparisons** within dataset × metric, Holm-corrected, with paired cluster bootstrap (2,000+ replicates). Clusters: Polyvore = item component, IQON/POG = user (§6.5), A100 = question. Present as a significance matrix.
- **CP:** DeLong test alongside the bootstrap. If they disagree, explain why: DeLong ignores clustering.
- **FITB/A100:** exact McNemar on **every** dataset, plus a cluster-adjusted variant where users cluster. v1 used McNemar on Polyvore only.
- **TOST equivalence** for every "no difference" claim, using the pre-specified practical margins (CP 0.002/0.005/0.010; FITB 0.5/1/2 pp; A100 2/5/10 pp). Never write "no difference" without it.
- **Bootstrap rank intervals** per model.
- **Friedman + Nemenyi critical-difference diagram** across settings, plus Kendall's W, labelled descriptive.
- Effect sizes: Δ with CI, Cohen's h for accuracies.
- Report total uncertainty alongside the construction-seed SD.
- Report p-values with their floor stated, e.g. "< 5e-4 at 2,000 reps".

### 6.4 Model-selection rule (one rule, in code)
- "Best overall / best generic / best fashion" = **highest validation CP AUC on the clean validation split of the training corpus**, computed by code from results files.
- **No hard-coded model keys.** v1's `scripts/paper_statistics.py` and `scripts/cluster_bootstrap.py` hard-coded Marqo, chosen on *historical* validation, while the A100 analysis chose CLIP on *clean* validation, so the same contrast meant two different models.

### 6.5 Dependence (IQON and POG)
- Primary: user-cluster bootstrap.
- Conservative sensitivity: **two-way user × item-component** cluster bootstrap. Alternatively, split the giant item component into sub-communities (e.g. Louvain) for block resampling.
- v1's item-component bootstrap was called "deliberately conservative", but its intervals had about the same width (ratio 0.97–1.24). One component holds about 47% of test outfits, so roughly 37% of replicates miss it. Do not call it conservative unless it is.
- Declare the grouping in the protocol and follow it. v1's YAML said item-component while the paper used user clusters.

### 6.6 Efficiency (one device: the DGX)
Measure every encoder with a new profiler (keep the v1 profiler for history):

| Metric | Method |
|---|---|
| Params (M), image tower only | count parameters |
| GFLOPs per image (MFLOPs for the probe) | `FlopCounterMode` at native res; 1 MAC = 2 FLOPs |
| **GPU-only latency** | preprocessed tensors already on GPU; CUDA events; 10 warm-up + 50 timed; **median + IQR**; batch 1 and 64; uniform bf16 + SDPA |
| Throughput | from batch-64 latency |
| Peak memory | `max_memory_allocated` (label as allocator peak) |
| Preprocessing cost | CPU decode/resize/normalize, reported **separately** (note fast vs slow processors) |
| End-to-end extraction | wall clock, labelled as including I/O, workers and image sizes |
| Energy | Wh / 1k images (NVML or power sampling; state the method) + kgCO₂e (state the grid intensity) |
| Storage | MB per 100k items, native vs PCA-256 |
| Probe cost | analytic MFLOPs per outfit + measured fit/predict time |
| Total study compute | GPU-hours |

v1's table timed decode, preprocessing and `.cpu()` at mixed precision (ResNet fp32, others fp16/bf16) with a mean of 5, while the protocol asked for a median. This inverted the FLOPs-to-latency order: Marqo appeared faster than ResNet50.

### 6.7 Calibration
- Keep Brier, log loss and ECE (10/15/20 bins).
- Add **post-hoc temperature/Platt scaling fitted on validation**, with before/after values.
- Add the **Brier decomposition** (reliability/resolution) and bootstrap CIs on ECE.

### 6.8 Validity additions
- **Near-duplicate audit across splits:** pHash plus embedding cosine (e.g. > 0.95). Report counts, and run a sensitivity analysis that excludes affected test outfits.
- *(Optional)* Check a random sample of Polyvore/IQON images against a public LAION index for upstream contamination. Skip it if no index is accessible, and say so.
- Cross-check historical Polyvore-D overlap counts against https://github.com/lee-research/outfit-completion-retrieval (train∩valid 3,781; train∩test 84; valid∩test 34 items).

### 6.9 What stays from v1
Construction seeds 42–46, label fractions (10/25/50/100%), bridge, collision audit, A100 handling (majority-label LAT primary, archive-gt sensitivity, mLAT, facets), failure consensus, outfit-length subgroup, provenance-restricted sensitivity (now 3 models), and the LookBench comparison. Cite LookBench **v3 (13 Apr 2026)** and note that it is a live benchmark. With PE-Core and Marqo-FashionCLIP added, the overlap grows (Marqo-FashionCLIP = 63.24).

---

## 7. Repo gaps to close before the run

1. **Data paths live inside `.venv/`.** There are 28 hard-coded `.venv/...` defaults, e.g. `scripts/run_final_iqon.py`, `scripts/extract_a100_embeddings.py`, `src/repbench/encoders/*.py` (which set `HF_HOME`/`TORCH_HOME`). Replace them with one config or env var (e.g. `REPBENCH_DATA_ROOT`) so recreating the env cannot delete data.
2. **Hard-coded model choices** in `scripts/paper_statistics.py` and `scripts/cluster_bootstrap.py` (§6.4).
3. **Torch pinned to `cu128`** in `pyproject.toml`; move to cu130 for the DGX (§5).
4. **No container or env-capture script.** Add a `Dockerfile` (or document the exact host setup) and `scripts/capture_env.py`.
5. **No single pipeline entry point.** The README is PowerShell-only. Add a Linux runner (Makefile, `run_v2.sh`, or a Python CLI) with resumable stages and refuse-to-overwrite behaviour.
6. **Encoder registry** (`src/repbench/encoders/registry.py`) has 7 models and three backends. Add the §3 models and generalize the open_clip backend to `hf-hub:` IDs. Add a Qwen3-VL-Embedding backend and a resolution override (DINOv3@336).
7. **No POG loader or protocol builder** (§4).
8. **Missing scripts:**
   - GPU-only efficiency profiler (§6.6)
   - unified statistics (§6.3)
   - MLLM zero-shot runner (§6.2)
   - near-duplicate audit (§6.8)
   - calibration additions (§6.7)
9. **Paper tables are hand-copied.** Generate every `.tex` table from CSVs, and add `scripts/check_paper_numbers.py` to assert in-text numbers against the artifacts.
10. **The OutfitTransformer lr is hard-coded** (`lr=1e-5` in `scripts/run_outfit_transformer_baseline.py`). Expose it, and fix the cache mapping.
11. **No Polyvore checksum** in the protocol. Record one.
12. **Stale artifact:** `artifacts/lookbench_correlations.csv` was computed on historical Polyvore. Regenerate it in v2 from the clean data.
13. **Unit tests:** extend for the new backends, the POG builder, the selection rule, the C-edge check and the stats functions, and run them on the DGX.

Refactor freely if it makes the pipeline simpler or more robust. Keep v1 behaviour reproducible behind its old settings where practical.

---

## 8. Readiness gates: all must pass before the full run

| Gate | Pass when |
|---|---|
| **G1 Environment** | Clean env builds from lock on the DGX; `pytest` green; GPU visible; bf16 + SDPA work; env capture written |
| **G2 Models** | All 15 encoders and the MLLM load at pinned revisions and embed a 64-image smoke batch with no NaNs and the expected dim; deterministic across two runs (≤ 1e-3 cosine drift in bf16) |
| **G3 Reproduce v1** | Rebuilt Polyvore-D-Clean and IQON3000-Clean splits on the DGX **hash-identical to v1**. Re-extract ResNet50 + CLIP and reproduce v1 PCA-256 test results within ±0.002 AUC / ±0.005 FITB. Investigate any larger gap before continuing |
| **G4 Data** | All checksums match; POG go/skip decided and logged; if go, the POG split is built and audited (zero cross-split item/exact-image overlap, zero collisions) |
| **G5 Pipeline dry run** | Full pipeline end-to-end on a ~2% subsample of every dataset × 3 models (one small, one large, Qwen embedder), through stats, tables and figures, with no manual steps |
| **G6 Speed test** | Time 5,000 images for ResNet50, CLIP-L, SigLIP2-So400m and Qwen3-VL-Emb on the DGX; write the projected full-run time to the decision log |
| **G7 Freeze** | `configs/v2_protocol.yaml` complete (roster + revisions, datasets + checksums, splits, probe, grids, selection rule, stats, margins, efficiency method, MLLM question IDs, baselines); hashed; repo committed and tagged `v2-freeze`; **no test outcomes viewed for new settings or POG** |

Only after G7: launch the full run.

---

## 9. Full run (after the freeze)

Order is a suggestion; parallelize CPU and GPU work where it helps.

1. Extract all encoders × all datasets (GPU, ~22–37 h estimated; G6 refines this).
2. Probe runs: primary native, then the sensitivities (PCA widths, operator, learners, label fractions, construction seeds). CPU-heavy; overlap with step 1.
3. OutfitTransformer runs (≈ 5–15 h) and MLLM zero-shot (≈ 4–10 h).
4. Efficiency profiling (≈ 1–2 h; nothing else running on the machine).
5. Statistics, calibration, audits, failure analyses.
6. Generate tables and figures from CSVs, then run `check_paper_numbers.py`.

**Rough total:** 35–65 GPU-hours, about 3–5 days wall clock. If a job fails mid-run, fix and resume. If the fix changes anything scientific, log it as a deviation.

---

## 10. Manuscript update (after results)

Edit `paper_elsevier_ivc/` and mirror to `paper_springer_mva/`. No TeX is installed on the Windows machine; install TeX Live on the DGX or elsewhere.

**Content:**
- **Abstract:** problem, method, **3 numeric findings**, implication.
- Collapse G1–4/O1–4/C1–4/RQ1–4 into **RQ1–4 + a 4-bullet list of findings**.
- Cut repeated hedges; keep caveats in Limitations.
- Rename the "Qualification" column to "Caveats".
- Rename §7.5 to "Practical implications".
- Main text ≈ 10–12k words; move audit detail to the supplement.
- Add a **qualitative figure**: FITB cases VLMs get right and ResNet gets wrong, and unanimous failures grouped by cause.
- Add a **feature-width table** for all datasets, with the sign of each contrast per setting.
- Add a **deviation table** (v2 protocol vs. what ran) and a **v1-pilot disclosure paragraph**. The public repo history shows v1 numbers, so be explicit.
- Reword "No initial Git commit existed at freeze" (v1) to "Provenance at freeze is established by SHA-256 hashes; the repository was initialised after the freeze".
- Justify image-only probing (text is out of scope) in the Limitations.

**Citations:**
- **Fix:**
  - IQON3000 is **Song et al. ACM MM 2019 (`song2019gpbpr`)**, not `nakamura2018iqon` (main.tex lines 63, 77, §4.2). Fix Nakamura's title or drop it.
  - HON = **Hierarchical Outfit Network** (lines 79, 107).
  - `simeoni2025dinov3`: now TMLR 2026; fix author spellings "Yi, Seung Eun" and "Ramamonjisoa, Michael".
  - Drop `feng2025mllmbenchmark` (padding) or replace it with Kolesnikov et al. CVPR 2019.
  - Make `dosovitskiy2021vit` and `dehghani2022efficiency` `@inproceedings` (ICLR).
- **Add:**
  - Methods: McNemar 1947, Halko et al. 2011, DeLong et al. 1988, Schuirmann 1987 (TOST), Friedman 1937, Cherti et al. CVPR 2023 (OpenCLIP), Bolya et al. NeurIPS 2025 (Perception Encoder), SigLIP v1 parent, Qwen3-VL-Embedding (arXiv 2601.04720), Marqo-FashionCLIP.
  - Positioning: **TATTOO** (arXiv 2509.23242; most important), Battle of the Backbones (NeurIPS 2023), Yang et al. tuple compatibility (MM 2020), GPT-4V fashion aesthetics (arXiv 2410.23730), FashionLLM (2502.15696), *Decoding Style* (CIKM 2024), CFALR (2606.13001), POG (KDD 2019), AesRec (2602.03416).
  - Classic models: NGNN (WWW 2019), HFGN (SIGIR 2020), CSA-Net (CVPR 2020), SCE-Net (ICCV 2019), Singhal et al. (WACV 2020), SHIFT15M (CVPRW 2023).
  - Keep the energy and carbon citations; v2 now reports energy.
- Six entries are still preprints (`tschannen2025siglip2`, `zhai2019vtab`, `czerwinska2025ecommerce`, `gao2026lookbench`, `feng2026fashionstylist`, `xue2026zooclaw`). Re-check them for published versions.

**IVC submission package** (re-check the live author guide; it blocked automated fetches):
- **Highlights**: separate file, 3–5 bullets, ≤ 85 characters each.
- **Graphical abstract**: made without generative AI (Elsevier policy, Aug 2026).
- Separate generative-AI declaration form, plus the in-paper statement.
- Cover letter (evaluation methodology, IVC's "quantitative comparison and performance evaluation" scope); 3–5 suggested reviewers.
- Review model is single-anonymized, so the public repo is fine. Tag the release (`v2.0-ivc`) and cite the tag.
- Build with `latexmk`; zero undefined refs; check figures for colour-blind-safe palettes and legibility at column width.

---

## 11. Coverage map (review issue → where handled)

| Review ID | Issue | Here |
|---|---|---|
| P0-1 | Specialization contrasts not parent-matched | §3 |
| P0-2 | Claims not robust to feature width; unreported native/PCA runs | §6.1, §10 |
| P0-3 | Two different "best model" selection sets | §6.4, §7.2 |
| P0-4 | Unsourced IQON CI in Table S3 | §7.9 (generated tables) |
| P0-5 | Polyvore CIs mislabelled "group-aware" | §6.3 |
| P0-6 | Item-component bootstrap called conservative | §6.5 |
| P0-7 | Efficiency table not like-for-like | §6.6 |
| P0-8 | C grid edge selections | §6.1 |
| P0-9 | Undisclosed protocol deviations (MLP, Marqo-only, median, grouping) | §6.1, §6.5, §6.6, §10 deviation table |
| P0-10 | Citation errors (IQON3000, HON, DINOv3, padding, missing method refs) | §10 |
| P0-11 | Small numeric errors and stale LookBench CSV | §7.9, §7.12 |
| P1-1 | Dated roster | §3 |
| P1-2 | No zero-shot MLLM | §6.2 |
| P1-3 | Undertrained OutfitTransformer on ResNet only | §6.2 |
| P1-4 | Unreported Polyvore MLP/XGBoost | §6.1 |
| P1-5 | Missing related work; LookBench live; GR-Lite licence; lee-research overlap | §3, §6.8, §6.9, §10 |
| P1-6 | Statistics incomplete or non-uniform | §6.3 |
| P1-7 | Shallow calibration | §6.7 |
| P1-8 | Writing and framing | §1, §10 |
| P2-1 | Qualitative figure | §10 |
| P2-2 | Probe operator sensitivity | §6.1 |
| P2-3 | Text modality | §10 (justify in limitations) |
| P2-4 | Near-duplicate audit | §6.8 |
| P2-5 | Upstream contamination probe | §6.8 (optional) |
| P2-6 | "No Git commit at freeze" wording | §10 |
| P2-7 | Section title | §10 |
| P2-8 | Figure styling | §10 |
| §5/§6 of review | Statistics plan / resource plan | §6.3 / §6.6 |
| §8 of review | IVC package checklist | §10 |

---

## 12. Done means

- [ ] Gates G1–G7 passed and logged; `v2-freeze` tag exists.
- [ ] Full run complete; every result file has a manifest entry and hash.
- [ ] All tables and figures generated by script; `check_paper_numbers.py` passes.
- [ ] Manuscript updated (both journal folders), builds cleanly; deviation table and v1 disclosure present.
- [ ] IVC package ready (highlights, graphical abstract, declarations, cover letter).
- [ ] README updated for the Linux/DGX pipeline; release tagged; `docs/v2_decisions.md` complete.
