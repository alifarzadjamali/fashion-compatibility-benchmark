# RepBench-Fashion final experimental protocol

Status: **re-frozen as v1.2 before IQON3000 or A100 model evaluation**. This document and its machine-readable
counterpart define the final experimental phase. Prior results remain immutable and are never
overwritten. Any necessary deviation must be applied fairly, entered in `docs/decision_log.md`, and
identified as a deviation in the final reports.

## Scientific scope and claims boundary

RepBench-Fashion measures how seven exact frozen image representations transfer to outfit
compatibility when the downstream pipeline is controlled. It does not introduce an encoder,
compatibility architecture, loss, or human annotation study. Polyvore-D-Clean is the completed
development corpus; IQON3000-Clean is the prospective independent replication; A100 is a strictly
external evaluation using only its previously collected public/expert annotations. Historical
Polyvore-D is retained only for comparison and does not satisfy the leakage-audited split controls. Because Polyvore influenced
development, its test set is not called a prospectively untouched blind test.

No result may trigger a change to preprocessing, labels, candidates, hyperparameters, metrics, or
the primary method. Test and A100 records cannot fit or select any learned object. All seven exact
checkpoints and preprocessing definitions are those pinned in `artifacts/model_audit.csv` and
enumerated in `configs/final_experimental_protocol.yaml`.

## Dataset rules

Polyvore-D-Clean retains its existing manifest and construction. Its train split alone fits item
standardization, PCA, outfit standardization, and the classifier. Validation alone selects `C`.
Test is evaluation-only. Existing zero item-ID, exact encoded-image, decoded-pixel, and outfit
overlap checks remain mandatory.

IQON3000 is obtained from the original authors' release (archive SHA-256
`305660cc09bd941a15096b6efeb536013efe0d534d9cd517419a49cd9ee650fd`). Before constructing
questions or exposing model test results, audit access and usage terms, missing/corrupt images,
duplicate items/outfits, encoded and decoded-pixel duplicates, category/outfit-length distributions,
users, and the complete outfit-item graph.

The source audit found that naive whole-component assignment is impossible: 292,197 of 297,916
valid source outfits (98.08%) lie in one item-connected component. This was discovered before any
IQON model evaluation. Version 1.1 therefore replaces the infeasible component allocator with a
deterministic balanced greedy hypergraph packing. After excluding outfits with fewer than two
distinct items and retaining one canonical copy of exact outfit duplicates, outfits are traversed
in the fixed seed-20260912 permutation. Item IDs and verified exact encoded/decoded-image duplicate
groups are indivisible leakage groups. An outfit whose groups are unowned goes to the split with the
largest normalized deficit; an outfit touching exactly one existing split goes there if capacity
remains; an outfit touching multiple splits or a full split is discarded. The largest feasible
prespecified verified-image feasibility-grid target is 124,000 outfits: exactly
86,800/12,400/24,800. Failure to
fill those capacities after verified pixel grouping fails closed. User identity is audited but is
not a model input or split constraint because the experiment evaluates item-level, not personalized,
generalization. Selection/retention distributions and discarded boundary outfits are reported.

There must be zero item, exact-image, outfit, candidate, and negative leakage. IQON preprocessing
objects and classifiers are fitted only on IQON training, with `C` selected only on IQON validation.
Polyvore and IQON are never pooled in the primary experiment.

A100 remains unchanged. Polyvore-trained and IQON-trained final scorers are applied separately.
There is no A100 split, fitting, tuning, calibration, negative generation, candidate replacement, or
checkpoint selection. LAT majority accuracy uses `argmax(gt_distribution)` as prescribed by the
paper, mLAT uses the probability assigned to the model-selected option, and archive-`gt` accuracy is
a required sensitivity result. Questions 92, 93, 94, 96, 98, 99, and 100 have known archive-`gt`
versus majority disagreement and are never silently repaired. AAT reports overall accuracy plus
Color (1-20), Style (21-52), Occasion (53-67), Season (68-79), Material (80-91), and Balance
(92-100). LAT and AAT remain separate.

## Locked primary pipeline

Official image preprocessing feeds a frozen image-only encoder. The existing representation-
appropriate raw normalization is retained. An item scaler and randomized PCA-256 are fitted on
training items only, followed by L2 normalization. Every unordered item pair contributes
`[abs(zi-zj), zi*zj]`; pair features are averaged to a 512-dimensional outfit vector. A second
standardizer is fitted on training CP outfits only. Logistic Regression uses `lbfgs`, at most 2,000
iterations, no class weighting, and the identical validation grid
`C={0.01,0.1,1,10,100,1000}`. The CP decision threshold is fixed at 0.5.

CP reports ROC-AUC, PR-AUC, accuracy, balanced accuracy, Brier score, log loss, and equal-width ECE.
Fifteen bins are primary; 10 and 20 bins are sensitivity checks. FITB reports accuracy and empirical
chance. Candidate scores are ranking scores, not calibrated probabilities. Optional calibration is
not part of the primary analysis; if later required, it must be validation-fitted and separately
labeled.

## Stability, inference, and practical magnitude

Frozen embeddings are extracted once, validated, hashed, and reused. The primary construction seed
is 20260912. With item partitions fixed, seeds 42-46 regenerate only CP negatives and FITB
candidates. This is construction stability, not five fictitious Logistic Regression training runs.
Report mean, SD, range, rank agreement, and whether conclusions cross the construction variability.

Use 2,000 paired bootstrap resamples and component-aware grouping wherever dependence exists.
Report all six predefined contrasts listed in the YAML and apply Holm correction within each
dataset/metric family. No all-pairs post-hoc testing is allowed. Rank analyses use Spearman and
Kendall descriptively because there are seven or fewer model-level observations.

No universal practical threshold is asserted. Instead report whether absolute effects exceed the
prespecified sensitivity levels: CP AUC 0.002/0.005/0.010; FITB 0.5/1/2 percentage points; A100
2/5/10 percentage points. Numerical difference, statistical uncertainty, and practical magnitude
are separate conclusions.

## Task-specific secondary baseline

The single baseline is an image-only OutfitTransformer secondary system. The published method is
peer reviewed and directly targets outfit CP/FITB, but no official implementation was found; an
MIT-licensed third-party implementation was audited only as a cross-check. To make the task and
modalities identical across Polyvore, IQON, and A100, an independent PyTorch implementation follows
the published set model while consuming cached frozen ResNet50 `IMAGENET1K_V2` 2,048-dimensional
features. This avoids a new encoder, text availability differences, and encoder fine-tuning.

The fixed configuration is a learned 64-dimensional image projection and outfit token, six
Transformer layers, 16 heads, 256-dimensional feed-forward blocks, dropout 0.1, no positional
encoding, a two-layer binary MLP, focal BCE (`gamma=2`, `alpha=0.25`), AdamW at 1e-5 with weight
decay 1e-4, batch size 50, 30 epochs maximum, and validation-AUC early stopping with patience five.
Seeds are 42-46. It is reported as an image-only adaptation, not an exact reproduction or new method.

The official context-aware GNN candidate was rejected because its evaluation obtains message-
passing context from the test graph; under an item-disjoint prospective protocol this would either
use held-out relationships or collapse to its context-free variant. The canonical Bi-LSTM candidate
was rejected because its public release targets obsolete TensorFlow, lacks explicit software
licensing, imposes dataset-specific item ordering, and does not provide a clean common set interface.

## Efficiency and auditability

Existing same-GPU extraction profiling remains primary: RTX 5070 Ti 16 GB, batch 64, two warm-up
batches, five timed repeats, CUDA synchronization, and PyTorch `FlopCounterMode`. Final reporting
adds CPU/RAM, OS, CUDA, package versions, precision, attention backend, median and variability,
checkpoint/cache storage, PCA/training/scoring time, and dataset-specific extraction time. Parameter
count is never equated with efficiency.

Every run records dataset/version, manifest and configuration hashes, construction/training seed,
checkpoint revision, preprocessing, learned-transform configuration, classifier parameters,
selected `C`, environment, hardware, precision/backend, git state, timestamps, and artifact paths.
Runs fail on non-finite embeddings/losses/probabilities, constants, wrong counts/candidates,
missing/duplicate IDs, checkpoint mismatch, unregistered datasets, or duplicate run IDs. Raw
probabilities, scores, labels, IDs, grouping IDs, models, logs, and hashes are preserved.

After the locked primary IQON result, only native-vs-PCA-256, one fixed nonlinear learner,
provenance sensitivity excluding Marqo, and construction stability are repeated. Existing Polyvore
robustness is preserved rather than rerun. The phase ends after final tables, figures, reports,
tests, lint/static checks, and the hostile final red-team review unless a genuine scientific defect
is found.
