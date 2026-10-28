# Revision report

Audit completed: 14 September 2026. No encoder extraction or expensive model training was rerun. New analyses were limited to existing metadata, manifests, generated questions, and frozen predictions.

## 1. Final value of X

**X = 4** core research questions, with exactly four gaps, objectives, and contributions.

The prior five-RQ structure separated replication from the main checkpoint comparison even though prospective replication is the central inferential test of that comparison, and it split robustness from the validity claims it supports. Four questions give a cleaner scientific hierarchy:

1. controlled frozen transfer and prospective replication;
2. construct/modelling boundaries;
3. operational model-selection axes;
4. validity and robustness.

This choice eliminates the earlier causal wording about whether gains are “explained” by scale or specialization. The intervention is now consistently the released checkpoint package.

## 2. Gap → Objective → RQ → Contribution → Evidence mapping

| Gap | Objective | RQ | Contribution | Evidence | Explicit answer |
|---|---|---|---|---|---|
| **G1:** no systematic common-probe comparison of contemporary frozen checkpoints across leakage-audited relational corpora | **O1:** measure and prospectively replicate the hierarchy | **RQ1:** checkpoint comparison on clean Polyvore and replication on IQON | **C1:** controlled seven-checkpoint benchmark | main Table 5, Fig. 1, predefined contrasts/rank correlations | evaluated contrastive VLMs lead ResNet50/DINOv3; broad hierarchy replicates, with lower IQON absolute performance |
| **G2:** external, specialization, contextual, and retrieval boundaries insufficiently triangulated | **O2:** test boundary conditions without conflating heterogeneous protocols | **RQ2:** extension to A100, contextual baseline, specialization, and retrieval | **C2:** construct-boundary triangulation | Figs. 2–3, A100 tables, baseline tables, exact LookBench v3 overlap | A100 corroborates family-level gaps; baseline is competitive not dominant; specialization is recipe-dependent; retrieval ranks reverse |
| **G3:** accuracy-only selection omits calibration, compute, and provenance | **O3:** compare operational criteria | **RQ3:** whether axes favour the same representation | **C3:** discrimination/calibration/efficiency/provenance audit | Table 6, Figs. 4–5, calibration and efficiency tables | axes do not agree; calibration is benchmark-conditional |
| **G4:** leakage, question validity, IQON selection/dependence, and probe choices threaten inference | **O4:** stress-test conclusions | **RQ4:** which conclusions survive the audit suite | **C4:** exact-collision, bridge, retention/dependence, seed, width, learner, compatibility-label, failure, and provenance checks | Figs. 6–8, supplement, four audit artifacts, frozen robustness tables | broad hierarchy survives, with bounded IQON selection/dependence and contamination limitations |

The same mapping appears in manuscript Table 2, `research_framework.md`, Results subsection labels, four Discussion answers, and the four-part Conclusion.

## 3. Major reviewer concerns addressed

- **Novelty:** added a defensive closest-work section and comparison table covering VICTOR, OutfitTransformer, HAT, MEDAL, Text2Outfit, LMLMO/BEiT3, HON/A100, the 2025 e-commerce embedding benchmark, FashionStylist, ZooClaw-FashionSigLIP2, LookBench v3, and bundle recommendation. The claim is comparative transfer measurement, not first use of modern representations.
- **Leakage wording:** changed the title and narrative from “Leakage-Safe” to “Leakage-Audited” or the exact “item- and exact-image-disjoint” guarantee.
- **Causal overclaiming:** removed explanations attributing effects to language supervision, scale, fashion adaptation, objective, or architecture. Checkpoints are packages.
- **IQON selection bias:** added retained-versus-excluded metadata analysis. The manuscript reports nearly preserved user/category coverage but substantial shifts toward shorter, less-repeated outfits.
- **IQON dependence:** added a conservative item-component cluster bootstrap from frozen predictions. Predefined contrast directions survive; the 46.56% largest component remains an explicit limitation.
- **False negatives/FITB validity:** audited all clean manifests. Generated CP negatives do not duplicate known positives; FITB candidates are unique and distractor completions do not reproduce known positives by item or exact-image set.
- **Historical bridge:** restored the historical-positive/regenerated-question bridge. The paper now explains why scores can rise after cleaning without attributing the rise to leakage removal.
- **A100:** corrected LAT/AAT names and implemented semantics; preserved majority-label, mLAT, archive-gt, seven disagreements, facets, and zero adaptation; corrected image accounting to 1,663 referenced paths and 1,592 globally unique exact images.
- **LookBench:** pinned version 3/revision date, exact five-checkpoint matches, and every Fine Recall@1 value. Added IQON clean ranks; correlations are descriptive and rank reversals carry the interpretation.
- **Compatibility-label efficiency:** states exactly that labelled outfits vary while PCA retains access to all train-split item embeddings.
- **Calibration:** consistently described as benchmark/protocol calibration under balanced synthetic CP; no FITB or population-prevalence calibration claim.
- **Task-specific baseline:** explicitly an image-only contextual adaptation, neither canonical reproduction nor SOTA attempt; heterogeneous published scores are not treated as comparable.
- **Figures:** regenerated nine vector/PDF figures from frozen publication tables with grayscale-safe encodings, shorter consistent labels, uncluttered legends, a five-domain rank plot, two-panel IQON Pareto view, all four seed-stability panels, explicit label-fraction wording, and a non-overlapping AAT colorbar.
- **Bibliography:** expanded and audited to 81 relevant records. New records are primary publisher/CVF/arXiv sources; no placeholder metadata remains.

## 4. Changes made

- Rewrote title, abstract, introduction contribution logic, RQs, framework table, Results navigation, Discussion, Conclusion, and limitations around X=4.
- Added the closest-contemporary-work table and `closest_prior_work.md`.
- Added `research_framework.md` and rebuilt `claim_evidence_matrix.md` against exact paths.
- Expanded `supplementary.tex` with construction validation, historical bridge, IQON selection/dependence, A100 accounting, LookBench IQON ranks, compatibility-label details, and calibration scope.
- Added read-only audit scripts and JSON/CSV evidence for IQON selection, IQON dependence, A100 internal image counts, and generated-question validity.
- Added `scripts/make_paper_revision_figures.py` as reproducible figure source; all figure outputs are based only on frozen tables.
- Retargeted the final package to *Machine Vision and Applications* after checking its current scope and author guidance. The manuscript uses the Springer Nature two-column numeric style, a 203-word abstract, six keywords, complete declarations, data/code availability statements, and a separate Online Resource 1.
- Recompiled both PDFs with Tectonic and reran the focused protocol/A100/FITB/statistics test set; all 17 selected tests passed.

## 5. Concerns intentionally left as bounded limitations

- CP negatives and FITB distractors remain synthetic; exact collision safeguards cannot exclude semantically valid alternative outfits.
- Transformed/perceptual near-duplicates cannot be exhaustively eliminated.
- Web-scale pretraining exposure cannot be proved absent; Marqo and GR-Lite have additional source-membership uncertainty.
- IQON retains 41.62% of valid outfits, skews shorter/less repeated, is not user-disjoint, and contains one large test item component.
- A100 contains only 100 LAT and 100 AAT questions and does not represent universal aesthetic preference.
- Only one task-specific contextual adaptation is implemented.
- LookBench exact overlap contains five checkpoints.
- Hardware efficiency is conditional on one GPU/software/batch configuration.
- The post-audit IQON and collision diagnostics were not preregistered.
- No architectural SOTA claim is made.

## 6. Suggestions rejected and why

- **Forcing five RQs or four for symmetry:** rejected as a formal objective. Four was chosen because it best matches the inferential structure, not because four was requested as a preferred number.
- **Running every recent architecture:** rejected because differing splits, modalities, negatives, personalization, and training objectives would not create a controlled comparison and would change the frozen study.
- **User-disjoint IQON reconstruction:** rejected because it would redefine the prospective protocol and likely require a new experimental campaign; the limitation and target population are now explicit.
- **Treating component bootstrap as the new primary IQON interval:** rejected because the huge component yields a conservative but low-resolution sensitivity and does not replace the predeclared user-cluster analysis.
- **Calling the label-fraction study few-shot:** rejected because PCA observes the full train-item embedding pool.
- **Claiming retrieval and compatibility are statistically independent:** rejected; five exact matches support rank-reversal evidence only.
- **Adding a decorative research-framework figure:** rejected. The compact vector-safe manuscript table is more readable and avoids redundant visual content.

## 7. Final hostile-review verdict

**Verdict: defensible for submission after author-side identity/release checks; no publication-threatening inconsistency found.**

Hostile Reviewer 2 tests:

- **Insufficient novelty / missed work:** mitigated by the narrowed transfer-measurement claim and explicit 2023–2026 closest-work matrix. Remaining novelty is empirical and should be judged as such, not architectural.
- **Leakage / false questions:** exact item/file/pixel/outfit disjointness and zero construction collisions are directly auditable. Residual transformed duplicates and pretraining contamination are acknowledged.
- **IQON selection / pseudoreplication:** materially strengthened, not hidden. Selection differences and graph dependence constrain external validity; the broad contrasts survive component sensitivity.
- **Causal overclaiming:** removed. Language supervision and specialization are not presented as isolated causes.
- **A100 misuse:** corrected semantics, counts, label policy, and scope; framed as low-powered corroboration.
- **Weak statistics:** paired/group-aware bootstrap, predeclared contrasts, Holm correction, effect intervals, construction seeds, and the new dependence sensitivity support the main family-level statements. Close/null results remain inconclusive.
- **Calibration misuse:** limited to balanced benchmark probabilities; FITB and population calibration are excluded.
- **Unfair baselines / cherry-picking:** roster rationale and package-level intervention are explicit; the contextual baseline is diagnostic; recent architectures are discussed without invalid numerical comparisons.
- **Post-hoc analysis:** the new audits are labelled post-audit diagnostics and used to scope validity, not generate a new performance claim.
- **Reproducibility:** exact revisions, protocol hashes, manifests, predictions, audit scripts, and figure source are mapped. A tagged archival release is still required before submission.
- **Contribution structure / figures / bibliography:** four one-to-one paths are navigable; figures are regenerated as vector PDFs; 81 references compile with no undefined citations or placeholders.

The strongest remaining reviewer attack is the 41.62% IQON retention coupled with its large item component. The manuscript now gives that concern enough quantitative evidence and scope that it is a transparent limitation rather than a hidden validity failure. A reviewer may still prefer a future fully user-disjoint corpus, but that is a new study rather than a correction to this frozen benchmark.
