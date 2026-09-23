# Decision log

## 2026-09-12 — Pilot protocol lock

- The official Polyvore Outfits disjoint files are consumed without regenerating CP negatives
  or FITB candidates.
- Item identity is `item_id`; official question keys (`set_id_index`) are resolved through each
  split's outfit JSON. Any unresolved or ambiguous key is a hard failure.
- Train, validation, and test item sets must be pairwise disjoint. Missing or corrupt images are
  hard failures and are never silently skipped.
- Learned item features are L2-normalized, standardized using training items only, reduced with
  randomized PCA fit on training items only, then L2-normalized again.
- The primary outfit feature is only mean-pooled `[abs(a-b), a*b]` over unordered item pairs.
  Optional scalar statistics are excluded for the pilot to preserve a 512-dimensional head.
- Logistic regression uses the fixed `C={0.01,0.1,1,10}` grid selected by validation ROC-AUC.
- FITB uses the same learned CP scorer by inserting each official candidate and taking argmax.
- Pilot seed is 20260912; encoder weights remain frozen and official checkpoint preprocessing is used.

## 2026-09-12 — Phase 0 dataset audit

- The authors' current Hugging Face distribution is gated and the authenticated local account does
  not have access. A public Kaggle mirror (`yiitcandeime/polyvore`, version 1) was used only to audit
  the packaged official JSON/question files and images. This provenance limitation is recorded.
- The packaged `disjoint` split reproduces the standard outfit/task counts and has no missing images,
  but it is not item-disjoint: 3,781 train–validation, 84 train–test, and 34 validation–test item IDs
  overlap. This exactly reproduces the previously reported 3,781 train–validation issue and adds
  small train–test/validation–test overlaps observed directly here.
- Because zero cross-split item overlap is a mandatory Phase 0 condition, the primary protocol fails
  closed. No repaired split is generated automatically because filtering outfits/questions would no
  longer be the official evaluation and requires an explicit protocol decision.
- FashionCLIP 2.0 replaces Marqo-FashionSigLIP in the intended clean pilot: its documented domain
  fine-tuning data is the Farfetch catalog, while Marqo does not disclose the provenance of its
  million-product fine-tuning corpus. DINOv3 is currently inaccessible to the authenticated account.

## 2026-09-12 — Historical-protocol pilot authorization

- The pilot is authorized on the unchanged packaged Polyvore-D files and is labeled
  `historical_polyvore_d`, never as leakage-audited primary evidence.
- DINOv2 ViT-B/14 (`facebook/dinov2-base`) replaces unavailable DINOv3 for Gate 1. DINOv3 remains
  planned for the full experiment after access is granted.
- The pilot roster is locked to ResNet50, DINOv2 ViT-B/14, CLIP ViT-L/14, and FashionCLIP 2.0.
- The full representation sweep remains prohibited without explicit permission after Gate 1.

## 2026-09-12 — DINOv2 post-pilot sanity audit

- The DINOv2 result was audited before any full sweep. Pooling is the final normalized CLS token;
  Hugging Face `pooler_output` exactly equals `last_hidden_state[:, 0]` for this architecture.
- Full cached features have correct counts, finite values, unit norms, and deterministic repeats.
- Two diagnostic tracks were run without changing the pilot: PCA-256 without standardization and
  native 768-D L2 features, using a diagnostic common grid extending to `C=100000`.
- The original DINOv2 PCA-256 score was materially under-tuned: validation selected the original
  upper boundary `C=10`, while the extended diagnostic reached 0.6908 CP AUC and 40.01% FITB.
- Native 768-D features still select `C=10` and score 0.6671/37.83%, so extraction and PCA information
  loss do not explain the original weakness. The narrow shared grid interacts with representation
  geometry after PCA and is the main identified confound.
- The original four-model pilot comparison is now provisional until all four models are rerun with
  one sufficiently broad, identical grid. No full sweep is authorized.

## 2026-09-12 — Shared logistic grid expansion

- The pilot Logistic Regression search grid is expanded identically for every representation to
  `C={0.01,0.1,1,10,100,1000}`, with selection still based only on validation ROC-AUC.
- Prior pilot results and DINOv2 audit artifacts remain historical records and are not relabeled as
  results from the expanded grid.

## 2026-09-13 — Full-sweep outfit-feature scaling correction

- An initial seven-model PCA-256 run selected the common upper boundary `C=1000` for every model.
- A scale audit found that the median standard deviation of the 256 absolute-difference coordinates
  was 23–32 times that of the 256 product coordinates. L2 regularization was therefore suppressing
  the product block according to arbitrary units rather than controlled predictive evidence.
- The locked primary pipeline is corrected by fitting one `StandardScaler` to training CP outfit
  features for each representation and applying it unchanged to validation, CP test, and all FITB
  candidates. This is the same leakage-safe operation and search grid for all seven models.
- The unscaled run is preserved under `artifacts/provisional_unscaled_primary/`; it is not used for
  primary conclusions. The correction was made from scale and boundary diagnostics, not test-score
  optimization.

## 2026-09-13 — Full benchmark authorization and checkpoint audit

- Gate 1 was declared passed and the seven-model full sweep was explicitly authorized without a
  further go/no-go gate. DINOv2 is absent from the primary roster and retained only in historical
  pilot/audit artifacts.
- All checkpoints were accessible and pinned before extraction. DINOv3 uses ViT-L/16 at the
  checkpoint's official 224-pixel preprocessing; the H+ checkpoint was not used.
- The checkpoint metadata corrected two stale draft assumptions: DINOv3 ViT-L/16 uses 224 rather
  than 256 pixels, and Marqo-FashionSigLIP uses 224 rather than 384 pixels.
- Marqo's inspected remote Transformers code was not executed; the pinned safetensors and official
  OpenCLIP configuration were loaded directly. GR-Lite's two small remote-code files were inspected,
  hashed, revision-pinned, and then executed because the release requires its custom architecture.
- FP16 caused non-finite DINOv3 and GR-Lite smoke-test features. Both DINO-family encoders therefore
  use BF16 inference; the other Hugging Face/OpenCLIP encoders use FP16. Stored embeddings are finite,
  L2-normalized FP32 for downstream numerical stability.
- GR-Lite's released 1024-dimensional CLS implementation differs from the paper text describing a
  512-dimensional projection, although the paper's result table and model card both say 1024. The
  exact requested release is included, but DINOv3-vs-GR-Lite is not interpreted as a pure controlled
  fine-tuning ablation.

## 2026-09-13 - Reviewer-readiness protocol correction

- The unchanged packaged split remains `historical_polyvore_d` for literature comparability. It is
  not relabeled or overwritten.
- A new `polyvore_d_clean` protocol pools all historical positives, removes eight exact duplicate
  outfits, constructs the full outfit-item graph, and assigns complete connected components. The
  largest component is reserved for training and deterministic best-fit assignment exactly reaches
  the proportional outfit targets. The resulting split has 0/0/0 pairwise item overlaps and no
  missing images.
- A post-construction adversarial audit found 60 cross-split groups of byte-identical images under
  different item IDs. The clean constructor was strengthened before final analysis to union all 120
  exact-image duplicate groups before component assignment. The final split has zero cross-split
  encoded-file or decoded-RGB duplicates. Resized or cropped near-duplicates remain a limitation.
- Clean CP uses one unique full-corruption negative per positive with length and semantic-category
  composition fixed. Clean FITB uses one blank and three within-split, same-category distractors;
  answer positions are balanced. Seed and SHA-256 digests are locked in the protocol manifest.
- A secondary `historical_polyvore_d_regenerated` bridge was added after the reviewer audit identified
  a confound: official-to-clean score changes mix unseen-item repartitioning with new negative and
  candidate generation. The bridge uses unchanged historical positive splits and the exact clean
  task generator, so it can separate most of these effects. It retains known leakage and is labeled
  secondary throughout.

## 2026-09-13 - Statistical and robustness lock

- The six requested pairwise comparisons are reported in full for both protocols. Holm correction is
  applied within each protocol and metric family; no comparison is dropped after seeing results.
- In addition to paired stratified bootstrap, a 2,000-resample connected-component cluster bootstrap
  is reported. This matters most for historical FITB, where one test-only item component contains
  2,058 outfits; the clean test maximum is two outfits.
- PCA-128, PCA-512, and native-width tracks were run without changing other pipeline stages. Fixed
  MLP and XGBoost heads and nested 10/25/50/100% compatibility-label subsets use identical settings
  for every representation.
- XGBoost and MLP are robustness checks, not replacements for the locked Logistic Regression primary.
  Their lower absolute accuracy is reported rather than tuned away.
- Marqo-FashionSigLIP is excluded in the provenance-restricted sensitivity table because it remains
  `possibly_exposed`; it remains in the scientifically useful full benchmark with a clear flag.

## 2026-09-13 - Efficiency profile

- All encoders were timed at batch size 64 on the same RTX 5070 Ti after two warm-up batches and over
  five repeats. Full-dataset extraction throughput is retained alongside repeated warmed latency.
- Per-image FLOPs use one PyTorch `FlopCounterMode` definition for all encoders. They are labeled
  estimates. Both whole-checkpoint and active image-forward parameter counts are reported so VLM
  text-tower parameters are not silently treated as image inference cost.

## 2026-09-13 - IQON3000 source-graph correction before evaluation

- The original archive was acquired and hashed before construction. A full metadata-only audit,
  performed before any IQON model score was computed, found 308,747 JSON records, 297,916 outfits
  with at least two distinct items, 610 duplicate outfits, and complete item-ID image coverage.
- Whole-component assignment specified in final protocol v1.0 is structurally impossible on IQON:
  292,197/297,916 valid outfits (98.08%) belong to one item-connected component. Exact-image links
  can only merge components further. Keeping that rule would leave no useful validation or test set.
- Protocol v1.1 therefore uses deterministic balanced greedy hypergraph packing and discards
  cross-owned boundary outfits. A feasibility grid, using only source structure and no representation
  or outcome values, showed exact 70/10/20 capacities through 125,000 outfits but not 150,000; the
  former is locked as 87,500/12,500/25,000. Verified encoded and decoded-image groups replace the
  CRC/size screen in final construction. The revision is prospective with respect to every IQON and
  A100 model result and will be reported as a dataset-construction limitation, not concealed.
- The subsequent full decode audit found 36 corrupt canonical files, three of which had a valid
  repeated archive variant, and additional decoded-pixel equivalences beyond byte-identical files.
  With those verified groups, 125,000 no longer filled exactly (validation/test missed capacity),
  while 124,000 did. Protocol v1.2 therefore locks 86,800/12,400/24,800 before any model evaluation.

## 2026-09-13 - Final external-validation and provenance policy

- IQON uncertainty is now recorded separately from Polyvore uncertainty. Marqo-FashionSigLIP and
  GR-Lite are treated as `possibly_exposed` for the IQON provenance-sensitivity analysis because
  their fashion adaptation corpora are not enumerated sufficiently to exclude IQON images. Generic
  web-pretrained models retain an incidental-web-exposure caveat but are not represented as known
  or targeted IQON exposure.
- A100 is kept external: neither its images nor labels can fit PCA, scalers, classifiers,
  calibration, thresholds, hyperparameters, or model choices. LAT majority labels and mLAT are
  primary; the archive `gt` field is mandatory sensitivity because seven questions disagree with
  the vote-distribution argmax. AAT remains separate and is reported by its six predefined facets.
- Construction-seed stability reuses each source-trained PCA because item partitions are fixed, but
  refits the outfit-feature scaler, validation-selected LR, and task examples when negative/FITB
  construction changes. This varies only the components genuinely affected by construction.

## 2026-09-13 - Final task-specific baseline and reporting lock

- The single task-specific baseline is explicitly an image-only OutfitTransformer adaptation, not
  an exact reproduction. It consumes the already cached frozen ResNet50 features so its modalities
  and item-disjoint protocols match the seven controlled probes. Its architecture, optimizer,
  validation-only early stopping, and seeds 42-46 were fixed before test evaluation.
- The official-context GNN alternative was rejected because test-graph message passing is not
  compatible with prospective item-disjoint evaluation. The released Bi-LSTM alternative was
  rejected for unlicensed, obsolete software and a dataset-specific ordered-sequence interface.
- Five stochastic training seeds are retained on both datasets despite the added runtime. Frozen
  encoder features are never re-extracted. The baseline remains secondary context and cannot turn
  this representation benchmark into an architectural SOTA claim.
- IQON split users are intentionally not disjoint: user identifiers are unavailable to the image-
  only methods and forcing user separation would discard substantially more data. Therefore the
  paper may claim unseen-item and exact-image separation, but not unseen-user generalization.
