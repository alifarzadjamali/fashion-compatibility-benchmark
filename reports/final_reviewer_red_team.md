# Final hostile-review audit

| Reviewer concern | Severity | Evidence | Fix/status | Residual limitation | Safe paper wording |
|---|---|---|---|---|---|
| Historical Polyvore leakage | Critical | Packaged split has cross-split items | Retained only as explicitly historical; two clean protocols are headline evidence | Historical scores remain non-generalization comparisons | “Historical Polyvore-D is reported for comparability, not as leakage-free evidence.” |
| Exact duplicate-image leakage | Critical | Distinct IDs can share bytes/pixels | Both clean constructors group encoded and decoded duplicates; automated zero-overlap tests pass | Transformed/cropped near-duplicates are not exhaustively excluded | “Zero exact-file and decoded-pixel overlap; perceptual near-duplicates remain possible.” |
| Test-set tuning | Critical | Polyvore influenced project development | Protocol v1.2 was hashed before IQON/A100 scoring; all learned components use train, C uses validation | Polyvore test is not prospectively blind | “IQON is the prospective replication; Polyvore is development-influenced.” |
| IQON giant graph component | Major | 98.08% of valid outfits form one item component | Deterministic item ownership discards boundary outfits; 124k outfits retain exact 70/10/20 ratios | Only 41.62% of valid outfits retained; selection may shift the population | “IQON-Clean prioritizes leakage safety over full source coverage.” |
| IQON user overlap | Major | Users occur across splits | Models receive no user IDs; inference is user-cluster bootstrapped | No unseen-user generalization claim | “The protocol is item/image-disjoint, not user-disjoint.” |
| Synthetic-negative realism | Major | CP negatives are deterministic corruptions | Five construction seeds plus unchanged A100 human/expert candidates and FITB are reported | Synthetic CP can reward corruption detection | “CP measures performance under the stated corruption process, triangulated with FITB and A100.” |
| Outfit-length dependence | Major | CP rises sharply with outfit length on both clean datasets | Length/category subgroup tables and sample sizes are reported | More pair evidence and easier corruption detection cannot be separated causally | “Aggregate results are accompanied by descriptive length strata; longer outfits are easier under this protocol.” |
| A100 underpowered | Major | 100 LAT and 100 AAT questions | Full CIs, mLAT, archive-gt sensitivity, facets, and no tuning | Small effects cannot be ranked reliably | “A100 supports broad trends; close differences are inconclusive.” |
| Pretraining contamination | Major | Marqo/GR-Lite fashion corpora are incompletely enumerated | Exact revisions audited; both flagged possibly exposed; clean-provenance sensitivity excludes them | Incidental web exposure cannot be disproved for web-scale models | “Results are shown with and without uncertain-provenance releases.” |
| Baseline fairness/reproducibility | Major | No official OutfitTransformer implementation was found | Fixed image-only adaptation, train/valid/test separation, five seeds, raw predictions | Not an exact published reproduction; one specialized baseline only | “The baseline provides task-specific context, not a comprehensive SOTA comparison.” |
| A100 label ambiguity | Major | Seven LAT archive labels differ from vote majority | Majority/mLAT primary and archive-gt mandatory sensitivity | Vote counts and archive policy remain imperfect | “Both label policies are reported without choosing the favorable one.” |
| Retrieval comparison overclaim | Major | Only five exact LookBench checkpoint matches | Similar variants excluded; Spearman/Kendall and reversals reported descriptively | n=5 has little statistical power | “Retrieval rank was not predictive in the small exact-match subset.” |
| GR-Lite as controlled child of DINOv3 | Major | Release differs in resolution/checkpoint head and data | Exact release documented; no pure-ablation language | Adaptation ingredients cannot be isolated | “GR-Lite is family-related contextual evidence, not a causal specialization ablation.” |
| PCA-256 confound | Major | Compression changes information and native capacity differs | 128/256/512/native tracks plus fixed nonlinear learners | Native track changes dimensionality and regularization geometry | “Primary claims use controlled PCA-256; native results are secondary.” |
| Calibration misuse | Minor | High AUC need not mean calibrated probabilities | Brier/log loss/ECE sensitivity and reliability plots; no FITB probability claim | ECE depends on binning | “Calibration is reported separately from discrimination.” |
| Efficiency confounds | Minor | Precision and feasible batch size can affect timings | Same GPU/profiler, warm-up, synchronization, exact precision/backend/batch recorded | Results remain hardware-specific | “Measured latency is implementation- and hardware-specific.” |
| Licensing | Major | A100 lacks a formal license file; GR-Lite derivative terms are nuanced | Terms and uncertainty recorded; no A100 redistribution claim | Downstream users must verify rights | “Access/academic-use statements are reported, not interpreted as blanket relicensing.” |
| No repository commit identifier | Minor | Workspace has no initial Git commit, so runs cannot name a commit | Frozen protocol, exact revisions, raw outputs, and SHA-256 artifact/report manifests are preserved | Code state is hash-audited but not commit-addressable | “The release manifest identifies every artifact; the archived release should be committed/tagged before submission.” |

## Hostile verdict

No unresolved issue invalidates the controlled comparison. The remaining risks constrain scope:
synthetic CP tasks, incomplete pretraining provenance, small A100, IQON selection, transformed
near-duplicates, and a contextual rather than canonical task baseline. A reviewer should reject
claims of universal SOTA, causal benefit from specialization, perfect contamination freedom, or
unseen-user generalization. With those claims removed, the evidence supports a rigorous benchmark
paper.
