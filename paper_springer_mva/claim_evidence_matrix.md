# Claim–evidence matrix

Paths are repository-relative. Frozen experiment outputs remain the numerical ground truth. The four post-audit diagnostics use existing metadata, manifests, questions, and predictions; they do not rerun encoders or fit models.

| ID / RQ | Major manuscript claim | Exact evidence | Qualification in manuscript |
|---|---|---|---|
| C01 / method | Seven released frozen checkpoints receive the same PCA-256 unordered-pair mean-pooled logistic probe. | `configs/final_experimental_protocol.yaml`; `reports/tables/final/representation_families.csv`; `src/repbench/features/pca.py`; `src/repbench/features/outfit_features.py`; `src/repbench/models/logistic.py` | Checkpoints are packages differing in architecture, data, scale, loss, and resolution; no causal component attribution. |
| C02 / RQ4 | Polyvore-D-Clean has 16,991/2,999/15,142 positives, removed eight duplicate outfits, links 120 exact-image groups, and has zero cross-split item/exact-image/outfit overlap. | `reports/tables/final/dataset_protocol_comparison.csv`; `reports/final_experiment_report.md`; manifests in `artifacts/final_artifact_manifest.csv` | Item and exact-image disjoint, not guaranteed free of transformed/perceptual duplicates or pretraining exposure. |
| C03 / RQ4 | IQON has 308,747 records, 297,916 valid outfits, and a 292,197-outfit (98.08%) giant component; the clean retained split is 86,800/12,400/24,800 (41.62%). | `reports/final_cross_dataset_report.md`; `reports/tables/final/dataset_protocol_comparison.csv`; `src/repbench/data/iqon.py` | Retained subset is item/exact-image-disjoint but not user-disjoint or representative of all IQON. |
| C04 / RQ4 | IQON retention preserves 3,523/3,568 users and category mix but skews to shorter and less-repeated outfits. | `artifacts/iqon3000_selection_audit.json`; `artifacts/iqon3000_selection_outfit_length_distribution.csv`; `scripts/audit_iqon_selection.py` | Post-audit descriptive diagnostic; selection bias is reported, not dismissed. |
| C05 / RQ4 | IQON user-cluster results are directionally stable under an item-component cluster bootstrap. | `artifacts/iqon_item_component_dependence_audit.json`; `artifacts/iqon_item_component_bootstrap_sensitivity.csv`; `artifacts/iqon_item_component_pairwise_sensitivity.csv`; `scripts/audit_iqon_dependence.py` | One component contains 46.56% of test positives; sensitivity complements but does not eliminate dependence uncertainty. |
| C06 / RQ4 | CP construction has no generated negative equal to a known positive and FITB has no duplicate candidate or known-positive distractor completion. | `artifacts/question_construction_audit.json`; `scripts/audit_question_construction.py`; builders in `src/repbench/data/` | Exact item/pixel collision audit; semantic false negatives and transformed copies remain possible. |
| C07 / RQ1 | Main Polyvore and IQON point estimates and 95% intervals for all seven checkpoints. | `reports/tables/final/main_polyvore_iqon_results.csv` | Bold values are point maxima, not all-pairs significance claims. |
| C08 / RQ1 | The evaluated contrastive VLMs substantially exceed ResNet50 and generic DINOv3 on both clean datasets. | `reports/tables/final/main_polyvore_iqon_results.csv`; `reports/tables/final/predefined_statistical_effects.csv`; `reports/final_statistics_report.md` | Family-level package comparison; no universal claim about all VLMs. |
| C09 / RQ1 | Cross-dataset ranks replicate: CP Spearman/Kendall .964/.905 and FITB .857/.714. | `reports/tables/final/rank_correlations.csv`; `reports/final_cross_dataset_report.md` | Seven-model descriptive ranks; Polyvore is development-influenced, IQON prospective. |
| C10 / RQ2 | Fashion specialization is recipe-dependent: FashionCLIP trails CLIP on both corpora; Marqo improves over SigLIP2 on IQON but not Polyvore. | `reports/tables/final/predefined_statistical_effects.csv`; `reports/final_statistics_report.md` | Released checkpoint differences cannot isolate specialization causally. |
| C11 / RQ4 | Historical-to-clean score changes are substantially affected by regenerated questions, not leakage removal alone. | `reports/tables/final/historical_polyvore_results.csv`; `artifacts/historical_regenerated_results.csv`; `reports/tables/final/main_polyvore_iqon_results.csv` | Bridge is not a perfect causal ablation; clean-minus-bridge ranges −.0017 to .0061 CP and −.0023 to .0106 FITB. |
| C12 / RQ2 | A100 provides unchanged five-choice human/expert-grounded corroboration with no fitting/tuning/calibration. | `reports/tables/final/a100_headline.csv`; `src/repbench/data/a100.py`; `scripts/evaluate_a100.py`; `reports/final_a100_report.md` | 100 LAT + 100 AAT questions; close rankings are low-powered. |
| C13 / RQ2 | A100 image accounting is 1,663 referenced files and 1,592 globally unique exact images, with 65 LAT/AAT exact-image overlap. | `artifacts/a100_internal_image_audit.json`; `scripts/audit_a100_internal_images.py` | Counts distinguish question references, paths, and exact global identities. |
| C14 / RQ2 | LAT primary labels use `argmax(gt_distribution)`; archive `gt` disagrees on questions 92, 93, 94, 96, 98, 99, 100. | `reports/final_a100_report.md`; `reports/tables/final/a100_headline.csv` | mLAT and archive-gt are sensitivities. |
| C15 / RQ2 | The OutfitTransformer adaptation is competitive on IQON CP but does not dominate all task/dataset/external metrics. | `reports/tables/final/task_specific_baseline.csv`; `reports/tables/final/task_specific_baseline_a100_aggregate.csv`; `src/repbench/models/outfit_transformer.py` | Contextual image-only adaptation, not the canonical published implementation. |
| C16 / RQ2 | LookBench v3 exact overlap contains five checkpoints and shows descriptive rank reversals; Polyvore rho .20/−.20, IQON rho 0/0. | `reports/tables/final/lookbench_overlap_by_protocol.csv`; frozen main results; LookBench v3 primary table; Fig. 3 source `scripts/make_paper_revision_figures.py` | No population independence or significance claim. |
| C17 / RQ3 | Marqo has the lowest Brier score and ResNet50 the lowest ECE15 on both clean protocols. | `reports/tables/final/calibration_metrics.csv`; `artifacts/final/calibration/reliability_bins_15.csv` | Calibration is conditional on balanced constructed CP; ECE is bin-dependent; no FITB calibration. |
| C18 / RQ3 | Efficiency values use one RTX 5070 Ti and extraction dominates PCA/classifier time. | `reports/tables/final/model_characteristics_and_efficiency.csv`; `reports/tables/final/efficiency_full.csv`; `reports/final_efficiency_report.md` | Hardware/software/batch-specific; GFLOPs are estimates. |
| C19 / RQ3 | Predictive accuracy, calibration, cost, and provenance can favour different checkpoints. | calibration/efficiency tables above; `reports/tables/final/provenance_exposure.csv`; `reports/tables/final/efficiency_pareto.csv` | Pareto status is conditional on chosen axes; provenance is not numerically collapsed. |
| C20 / RQ4 | Construction-seed ranks remain stable over seeds 42–46. | `reports/tables/final/construction_seed_rank_agreement.csv`; `reports/tables/final/construction_seed_stability.csv` | Construction variation is distinct from test-sample/training-seed uncertainty. |
| C21 / RQ4 | Native-width and XGBoost checks broadly preserve the hierarchy. | `reports/tables/final/iqon_pca256_vs_native.csv`; `reports/tables/final/iqon_downstream_learner_robustness.csv` | Native width improves absolute results; nonlinear learning is not uniformly better. |
| C22 / RQ4 | Compatibility-label gaps persist at 10/25/50/100% labelled outfits. | `reports/tables/final/low_data_results.csv`; Fig. 8 source | PCA retains all train-split item embeddings; not generic few-shot or encoder adaptation. |
| C23 / RQ4 | Removing uncertain-provenance Marqo and GR-Lite preserves the broad VLM and FashionCLIP-vs-CLIP conclusions. | `reports/tables/final/clean_provenance_sensitivity.csv`; `reports/tables/final/provenance_exposure.csv` | Remaining web contamination cannot be certified absent; Marqo-best claim is conditional. |
| C24 / RQ4 | Outfit-length trends and disagreement/unanimous-error counts are descriptive. | `reports/tables/final/subgroup_performance_summary.csv`; `reports/tables/final/failure_consensus_audit.csv` | No causal explanation of subgroup difficulty or failure cues. |
| C25 / reproducibility | Protocol and IQON manifests are SHA-256 identified and final outputs are traceable. | `reports/tables/final/reproducibility_inventory.csv`; `artifacts/final_artifact_manifest.csv`; `artifacts/final_report_manifest.csv`; `reports/paper_readiness_report_v2.md` | No initial Git commit existed at freeze; tagged archival release remains required before submission. |

## Framework-to-evidence map

- **G1 → O1 → RQ1 → C1:** C07–C09, plus Table 3 and Fig. 1.
- **G2 → O2 → RQ2 → C2:** C10 and C12–C16, plus Figs. 2–3.
- **G3 → O3 → RQ3 → C3:** C17–C19, plus Table 4 and Figs. 4–5.
- **G4 → O4 → RQ4 → C4:** C02–C06, C11, and C20–C24, plus Figs. 6–8 and the supplement.

## Method-to-code map

| Method | Source |
|---|---|
| Encoder identity/preprocessing | `src/repbench/encoders/registry.py`; encoder modules under `src/repbench/encoders/` |
| Train-only PCA and normalization | `src/repbench/features/pca.py` |
| Unordered pair features and pooling | `src/repbench/features/outfit_features.py` |
| Logistic fitting/selection | `src/repbench/models/logistic.py`; `scripts/run_benchmark.py`; `scripts/run_final_iqon.py` |
| CP/FITB/calibration metrics | `src/repbench/eval/compatibility.py`; `src/repbench/eval/fitb.py`; `src/repbench/eval/calibration.py` |
| Clean builders | `src/repbench/data/clean_protocol.py`; `src/repbench/data/iqon.py` |
| Paired statistics | `scripts/final_phase_statistics.py`; `scripts/paper_statistics.py` |
| Revision-only metadata/prediction audits | `scripts/audit_iqon_selection.py`; `scripts/audit_iqon_dependence.py`; `scripts/audit_question_construction.py`; `scripts/audit_a100_internal_images.py` |
| Revised vector figures | `scripts/make_paper_revision_figures.py` |
