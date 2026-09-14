# Methodology

The Gate 1 pilot uses the historical packaged Polyvore-D protocol. It is not described as truly
item-disjoint because direct auditing found cross-split item-ID overlap. The historical files remain
unchanged to preserve comparability with prior work.

The benchmark holds all downstream components fixed. Each encoder produces L2-normalized
image-only features. A scaler and PCA-256 are fitted on training items only and applied unchanged
to validation and test items. The transformed features are L2-normalized.

For each outfit, every unordered item pair contributes the concatenation of absolute difference
and elementwise product. Mean pooling across pairs creates a fixed 512-dimensional, order-invariant
feature. A logistic-regression classifier is selected from the identical shared regularization grid
`C={0.01,0.1,1,10,100,1000}` for every representation using validation ROC-AUC. CP reports test
ROC-AUC and PR-AUC. FITB completes each official question with each official candidate and selects
the candidate receiving the highest CP probability.

Before Logistic Regression, each of the 512 outfit-feature coordinates is standardized using only
training CP examples. This common step was added before the full sweep because the L2-normalized
pair construction produced a 23–32× scale imbalance between the absolute-difference and product
blocks, making the common `C` search depend on arbitrary coordinate units. The fitted scaler is
applied unchanged to validation, CP test, and FITB candidate outfits.

The pilot comprises ResNet50 ImageNet V2, DINOv2 ViT-B/14, OpenAI CLIP ViT-L/14, and FashionCLIP
2.0. It is retained only as historical Gate 1 material.

## Full representation benchmark

The approved full sweep replaces DINOv2 and the 224-pixel CLIP pilot checkpoint with the locked
seven-model roster in `configs/experiments/primary.yaml`: ResNet50, DINOv3 ViT-L/16, CLIP
ViT-L/14@336, SigLIP2-B/16@384, FashionCLIP 2.0, Marqo-FashionSigLIP, and GR-Lite. All checkpoints
are revision-pinned and all seven use the same packaged split, CP examples, FITB questions,
candidates, negatives, train-only transformations, and evaluation code.

The controlled primary track uses PCA-256 and the fixed 512-dimensional pairwise outfit feature.
The native-dimensional track is secondary and changes only the item width (and consequently the
pairwise outfit width); it does not replace the controlled primary conclusions.

## Leakage-free Polyvore-D-Clean protocol

`polyvore_d_clean` is a separate derived protocol; it never overwrites the packaged historical
files. All 35,140 historical positive outfits are pooled, eight exact duplicate item sets are
removed, and the full positive-outfit/item bipartite graph is decomposed into connected components.
Distinct item IDs with byte-identical images are linked before assignment. Whole components are
assigned to one split only. The largest component (14,679 outfits and 57,481 items) is assigned to
training, then deterministic best-fit assignment meets largest-remainder targets of
16,991/2,999/15,142 outfits. This gives zero item-ID and zero exact-image overlap among all three
splits. The exact-image audit does not exclude resized or cropped near-duplicates.

One CP negative is generated for each positive by replacing every item with a different within-split
item of the same semantic category. Negative outfits preserve positive length and category
composition, are unique, and cannot equal a known positive. One FITB question is generated per
positive; the held-out answer and three unique distractors share a semantic category, all candidates
belong to the same split, and no distractor completes a known positive. Answer positions are balanced
to within one question. Seed `20260912` and per-file SHA-256 digests are in the protocol manifest.

Because the historical and clean protocols use different CP negatives and FITB candidates, their
absolute score difference is a composite of repartitioning and task-generation effects. The
secondary `historical_polyvore_d_regenerated` bridge retains the historical positive splits but uses
the identical clean question generator. Historical-regenerated versus clean-regenerated is the more
controlled split sensitivity; the bridge remains leakage-prone and is never primary evidence.

## Paper-readiness robustness and inference

All headline intervals use 2,000 paired resamples shared across representations. Stratified
example/question bootstrap is reported for direct comparison; an additional graph-component cluster
bootstrap accounts for repeated garments. Predefined pairwise effects report bootstrap intervals,
exact McNemar tests for FITB, and Holm correction within each protocol/metric family.

PCA-128, PCA-512, and native-width tracks change only the common representation bottleneck. Learner
sensitivity uses one fixed 128-unit MLP and one fixed 300-tree depth-4 XGBoost configuration for all
representations. The 10/25/50/100% curves are compatibility-label-efficiency experiments: PCA may use
all unlabeled training-split garments, while the outfit-feature scaler is refit only on each labeled
subset. They must not be described as end-to-end image-data efficiency.
