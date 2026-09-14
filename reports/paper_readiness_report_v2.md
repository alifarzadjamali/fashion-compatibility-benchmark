# RepBench-Fashion paper-readiness verdict v2

## Verdict

- **Conference-ready:** yes.
- **Strong-conference-ready:** yes for a controlled applied benchmark at RecSys, WSDM, SIGIR, or
  ACM MM; less naturally positioned for CVPR without a stronger vision-method contribution.
- **Journal-ready:** yes for a focused empirical journal submission such as JVCIR, ESWA, or
  Neurocomputing; IEEE TMM remains a higher-risk fit because the work deliberately contributes no
  new multimodal architecture and A100 is small.

## Conclusions that survive the full validation stack

1. Frozen modern VLM features substantially outperform ResNet50 and generic DINOv3 for outfit
   compatibility across Polyvore-D-Clean, IQON3000-Clean, and broadly on A100.
2. Dataset transfer preserves the broad ordering but not exact top-model positions; Polyvore/IQON
   CP rank agreement is .964 and FITB agreement .857.
3. Fashion specialization is model-dependent, not uniformly beneficial. Marqo is strong, but
   FashionCLIP trails CLIP; generic models lead the conservative provenance-only subset.
4. GR-Lite improves over DINOv3, but this is an empirical family comparison rather than a causal
   fine-tuning ablation.
5. Retrieval rank does not reliably predict compatibility rank in the five-model exact LookBench
   overlap.
6. Conclusions are stable to construction seeds, PCA/native width, nonlinear learners, label
   fractions, calibration reporting, and exclusion of uncertain-provenance models.
7. The task-specific baseline wins IQON CP but not IQON FITB, and transfers worse to A100; frozen
   probes are competitive rather than universally superior.

## Claims the paper can safely make

- This is a controlled comparison of seven exact frozen checkpoints under identical downstream
  methodology on two exact-item/image-disjoint protocols plus unchanged external human/expert data.
- The broad modern-VLM advantage replicates across datasets and is larger than construction-seed
  variability.
- Retrieval performance and compatibility performance answer meaningfully different questions.
- Accuracy, calibration, compute efficiency, and provenance yield different deployment choices.

## Claims to avoid

- “Polyvore-D is inherently leakage-free,” “all near-duplicate leakage is eliminated,” or “IQON is
  user-disjoint.”
- “Marqo/fashion specialization is universally superior,” “GR-Lite isolates the causal effect of
  fashion tuning,” or “web-pretraining contamination is impossible.”
- “RepBench-Fashion establishes architectural SOTA,” “A100 proves human preference,” or “rank
  correlations with five models are statistically definitive.”
- “FITB scores are calibrated probabilities” or “parameter count alone measures efficiency.”

## Recommended narrative and package

Lead with methodology: frozen representation quality for relational compatibility is not implied by
retrieval quality, and conclusions require leakage-safe cross-dataset and human-grounded validation.
Present historical Polyvore only as a bridge; make Polyvore-D-Clean and prospective IQON-Clean the
main tables; use A100 to constrain human-validity claims. Put the baseline, native width, learners,
low-label curves, provenance sensitivity, and construction seeds as robustness evidence rather than
competing contributions.

No further experiment is likely to improve readiness enough to justify expanding the project. The
remaining work is manuscript writing and, where legally required, confirming redistribution terms;
neither calls for another model, dataset, architecture, or human study.

