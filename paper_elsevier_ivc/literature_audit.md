# Literature and journal-fit audit

Audit updated: 22 September 2026. The bibliography contains 81 records. Metadata were checked against an original publisher proceedings/article page, PMLR/JMLR/OpenReview, the official arXiv record, or (for a released checkpoint without a paper) the provider model card. DOI, venue, year, volume, issue, and page fields were included only when those records exposed them. arXiv-issued DOIs are identified as such. No discovery page or secondary blog is used as bibliographic authority.

## Re-audit of the seven arXiv-designated records

The seven entries formatted as arXiv preprints were checked again against official publisher and conference records. VTAB, SigLIP2, DINOv3, LookBench, FashionStylist, and ZooClaw-FashionSigLIP2 still have no confirmed authoritative peer-reviewed version of the same work. The e-commerce benchmark's official arXiv record states acceptance at FTC 2025, and the Springer FTC 2025 proceedings volumes were checked, but no matching authoritative chapter page or DOI was found. It therefore remains cited by its verified arXiv record with the acceptance note rather than speculative proceedings metadata. No arXiv entry was replaced in this revision. All DOI and explicit URL targets in the bibliography were re-resolved; none produced a mismatched destination or unresolved 404.

## Image and Vision Computing and Elsevier requirements

- **Scope and instructions checked:** the current [Image and Vision Computing journal page and scope](https://www.sciencedirect.com/journal/image-and-vision-computing/about/aims-and-scope), [Guide for Authors](https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors), and [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions). IVC explicitly values image interpretation and computer vision research supported by quantitative comparison and performance evaluation. The manuscript presents controlled representation transfer, relational visual reasoning, benchmark validity, and model-choice trade-offs within that scope.
- **Template and references checked:** the package uses the current Elsevier `elsarticle` class (version 3.5, 9 January 2026) in its standard single-column `preprint,12pt` initial-submission configuration. IVC uses sequential bracketed citations, implemented with `elsarticle-num.bst`. Journal-specific configuration is isolated at the top of `main.tex` for later Elsevier retargeting.
- **Highlights and graphical abstract:** five highlights are supplied in a separate editable file; every bullet is at most 85 characters, consistent with the current [Elsevier highlights instructions](https://www.elsevier.com/researcher/author/tools-and-resources/highlights). Elsevier describes a graphical abstract as optional but encouraged unless a journal guide makes it mandatory. No graphical abstract was added because it is not required for scientific completeness and would duplicate the paper's existing evidence figures.
- **Declarations and supplementary material:** the manuscript and separate submission files provide a CRediT contribution statement, declaration of competing interest, funding statement, data/code availability, ethics statement, and acknowledgements. Supplementary tables and figures are labelled with `S` prefixes in a separately compiled PDF.
- **Benchmarking precedent retained:** Feng et al., “Benchmarking large and small MLLMs,” *Machine Vision and Applications* 36, 137 (2025), [DOI 10.1007/s00138-025-01762-0](https://doi.org/10.1007/s00138-025-01762-0), remains relevant literature evidence that technically motivated empirical benchmarking can constitute a substantive contribution.

## Bibliography audit

| Key | Verified primary record | Why it is used |
|---|---|---|
| `ding2024fashion` | [ACM, DOI 10.1145/3627100](https://doi.org/10.1145/3627100) | Current survey framing computational fashion recommendation and evaluation fragmentation. |
| `deldjoo2024review` | [ACM, DOI 10.1145/3624733](https://doi.org/10.1145/3624733) | Current survey of multimodal, personalized, and modern fashion recommenders. |
| `selwon2024explainable` | [ACM, DOI 10.1145/3664614](https://doi.org/10.1145/3664614) | Recent review of explainable fashion-compatibility methods. |
| `han2017bilstm` | [ACM MM, DOI 10.1145/3123266.3123394](https://doi.org/10.1145/3123266.3123394) | Foundational Polyvore CP/FITB sequence model. |
| `vasileva2018typeaware` | [Springer ECCV, DOI 10.1007/978-3-030-01270-0_24](https://doi.org/10.1007/978-3-030-01270-0_24) | Foundational type-aware compatibility embeddings and Polyvore protocol. |
| `cucurull2019context` | [IEEE/CVF CVPR, DOI 10.1109/CVPR.2019.01290](https://doi.org/10.1109/CVPR.2019.01290) | Context-aware relational compatibility architecture. |
| `sarkar2023outfittransformer` | [IEEE/CVF WACV, DOI 10.1109/WACV56688.2023.00359](https://doi.org/10.1109/WACV56688.2023.00359) | Source design for the task-specific contextual baseline. |
| `zou2022a100` | [IEEE/CVF CVPR, DOI 10.1109/CVPR52688.2022.02052](https://doi.org/10.1109/CVPR52688.2022.02052) | Original A100 human/expert-grounded LAT and AAT benchmark. |
| `nakamura2018iqon` | [Official arXiv:1807.03133](https://arxiv.org/abs/1807.03133) | Original IQON3000 dataset and outfit/style task description. |
| `song2018attentive` | [ACM SIGIR, DOI 10.1145/3209978.3209996](https://doi.org/10.1145/3209978.3209996) | Attentive knowledge-distillation compatibility model. |
| `mcauley2015styles` | [ACM SIGIR, DOI 10.1145/2766462.2767755](https://doi.org/10.1145/2766462.2767755) | Foundational image-based style and substitute recommendation. |
| `veit2015style` | [IEEE ICCV, DOI 10.1109/ICCV.2015.527](https://doi.org/10.1109/ICCV.2015.527) | Foundational visual clothing-style representation from co-occurrence. |
| `chen2018dress` | [AAAI, DOI 10.1609/aaai.v32i1.11895](https://doi.org/10.1609/aaai.v32i1.11895) | Mixed-category metric learning for fashion collocation. |
| `hsiao2018capsule` | [IEEE/CVF CVPR, DOI 10.1109/CVPR.2018.00748](https://doi.org/10.1109/CVPR.2018.00748) | Set-level capsule wardrobe construction. |
| `li2017set` | [IEEE TMM, DOI 10.1109/TMM.2017.2690144](https://doi.org/10.1109/TMM.2017.2690144) | End-to-end outfit composition over sets. |
| `yang2019attributes` | [ACM SIGIR, DOI 10.1145/3331184.3331242](https://doi.org/10.1145/3331184.3331242) | Rich-attribute interpretable fashion matching. |
| `polania2019categories` | [IEEE ICIP, DOI 10.1109/ICIP.2019.8803587](https://doi.org/10.1109/ICIP.2019.8803587) | Cross-apparel-category compatibility learning. |
| `kim2021sval` | [IEEE ICCVW, DOI 10.1109/ICCVW54120.2021.00123](https://doi.org/10.1109/ICCVW54120.2021.00123) | Self-supervised visual attributes for compatibility. |
| `chen2019pog` | [ACM KDD, DOI 10.1145/3292500.3330652](https://doi.org/10.1145/3292500.3330652) | Large-scale personalized outfit generation. |
| `song2019gpbpr` | [ACM MM, DOI 10.1145/3343031.3350956](https://doi.org/10.1145/3343031.3350956) | Personalized compatibility ranking. |
| `chen2019visualexplanations` | [ACM SIGIR, DOI 10.1145/3331184.3331254](https://doi.org/10.1145/3331184.3331254) | Multimodal visual explanations for personalized fashion recommendation. |
| `lin2020explainable` | [IEEE TKDE, DOI 10.1109/TKDE.2019.2906190](https://doi.org/10.1109/TKDE.2019.2906190) | Joint compatibility and comment-generation explanation. |
| `wang2019mcn` | [ACM MM, DOI 10.1145/3343031.3350909](https://doi.org/10.1145/3343031.3350909) | Compatibility prediction plus item-level diagnosis. |
| `balim2023diagnosing` | [Elsevier ESWA, DOI 10.1016/j.eswa.2022.119305](https://doi.org/10.1016/j.eswa.2022.119305) | Recent deep-learning diagnosis study. |
| `wang2024pfe` | [AAAI, DOI 10.1609/aaai.v38i8.28764](https://doi.org/10.1609/aaai.v38i8.28764) | 2024 textual extraction/explanation of compatibility relations. |
| `li2024attribute` | [Elsevier ECRA, DOI 10.1016/j.elerap.2024.101451](https://doi.org/10.1016/j.elerap.2024.101451) | 2024 attribute-augmented explainable compatibility framework. |
| `liu2016deepfashion` | [IEEE CVPR, DOI 10.1109/CVPR.2016.124](https://doi.org/10.1109/CVPR.2016.124) | Major fashion recognition/retrieval dataset, used to distinguish item tasks from outfit relations. |
| `han2017spatial` | [IEEE ICCV, DOI 10.1109/ICCV.2017.163](https://doi.org/10.1109/ICCV.2017.163) | Fashion200K and language-aware fashion concept discovery context. |
| `kiapour2015street` | [IEEE ICCV, DOI 10.1109/ICCV.2015.382](https://doi.org/10.1109/ICCV.2015.382) | Street-to-shop retrieval, illustrating a non-compatibility fashion target. |
| `gao2026lookbench` | [Official arXiv:2601.14706 v3](https://arxiv.org/abs/2601.14706) | 2026 retrieval benchmark, source of GR-Lite and exact-overlap retrieval scores. |
| `deng2009imagenet` | [IEEE CVPR, DOI 10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848) | ImageNet provenance for the supervised baseline. |
| `he2016resnet` | [IEEE CVPR, DOI 10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90) | ResNet architecture provenance. |
| `dosovitskiy2021vit` | [OpenReview ICLR](https://openreview.net/forum?id=YicbFdNTTy) | Vision Transformer foundation. |
| `radford2021clip` | [PMLR ICML](https://proceedings.mlr.press/v139/radford21a.html) | Original CLIP objective and web-supervised transfer. |
| `jia2021align` | [PMLR ICML](https://proceedings.mlr.press/v139/jia21b.html) | Independent evidence for web-scale noisy image--text pretraining. |
| `zhai2023siglip` | [IEEE ICCV, DOI 10.1109/ICCV51070.2023.01100](https://doi.org/10.1109/ICCV51070.2023.01100) | Original sigmoid-loss image--text pretraining. |
| `tschannen2025siglip2` | [Official arXiv:2502.14786](https://arxiv.org/abs/2502.14786) | 2025 SigLIP2 training recipe and released family. |
| `caron2021dino` | [IEEE ICCV, DOI 10.1109/ICCV48922.2021.00951](https://doi.org/10.1109/ICCV48922.2021.00951) | Original DINO self-distillation lineage. |
| `oquab2024dinov2` | [OpenReview TMLR](https://openreview.net/forum?id=a68SUt6zFt) | DINOv2 scaling and robust feature lineage. |
| `simeoni2025dinov3` | [Official arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | Exact DINOv3 family used in the benchmark. |
| `chia2022fashionclip` | [Nature Scientific Reports, DOI 10.1038/s41598-022-23052-9](https://doi.org/10.1038/s41598-022-23052-9) | FashionCLIP training and domain-specialized transfer claims. |
| `zhu2025gcl` | [ACM Web Conference, DOI 10.1145/3701716.3715227](https://doi.org/10.1145/3701716.3715227) | Generalized contrastive ranking objective behind the Marqo ecosystem. |
| `marqo2024fashionsiglip` | [Official model card](https://huggingface.co/Marqo/marqo-fashionSigLIP) | Primary release record for the exact Marqo-FashionSigLIP checkpoint; no standalone model paper was found. |
| `schuhmann2022laion` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html) | Open web-scale image--text data context and contamination limitation. |
| `kornblith2019transfer` | [IEEE CVPR, DOI 10.1109/CVPR.2019.00277](https://doi.org/10.1109/CVPR.2019.00277) | Foundational evidence on source accuracy versus downstream transfer. |
| `ericsson2021transfer` | [IEEE CVPR, DOI 10.1109/CVPR46437.2021.00537](https://doi.org/10.1109/CVPR46437.2021.00537) | Systematic self-supervised transfer comparison. |
| `zhai2019vtab` | [Official arXiv:1910.04867](https://arxiv.org/abs/1910.04867) | Multi-task visual representation benchmark design. |
| `li2022elevater` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html) | Benchmarking language-augmented visual models under common protocols. |
| `koh2021wilds` | [PMLR ICML](https://proceedings.mlr.press/v139/koh21a.html) | Distribution-shift benchmark methodology and cross-domain caution. |
| `brier1950` | [AMS, DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2) | Original Brier probability score. |
| `niculescu2005probabilities` | [ACM ICML, DOI 10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430) | Empirical probability calibration of supervised learners. |
| `guo2017calibration` | [PMLR ICML](https://proceedings.mlr.press/v70/guo17a.html) | Modern neural calibration and temperature scaling. |
| `naeini2015calibration` | [AAAI, DOI 10.1609/aaai.v29i1.9602](https://doi.org/10.1609/aaai.v29i1.9602) | Histogram/Bayesian binning calibration method context. |
| `vaicenavicius2019calibration` | [PMLR AISTATS](https://proceedings.mlr.press/v89/vaicenavicius19a.html) | Formal cautions about calibration evaluation. |
| `minderer2021revisiting` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html) | Reassessment of calibration across modern architectures and scale. |
| `ovadia2019uncertainty` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) | Predictive uncertainty under dataset shift. |
| `schwartz2020green` | [ACM CACM, DOI 10.1145/3381831](https://doi.org/10.1145/3381831) | Motivation for accuracy--compute reporting. |
| `henderson2020reporting` | [JMLR](https://jmlr.org/papers/v21/20-312.html) | Systematic energy and compute reporting guidance. |
| `dehghani2022efficiency` | [OpenReview ICLR](https://openreview.net/forum?id=iulEMLYh1uR) | Why efficiency requires multiple declared measures. |
| `hooker2021hardware` | [ACM CACM, DOI 10.1145/3467017](https://doi.org/10.1145/3467017) | Hardware dependence of practical model performance. |
| `dodge2022carbon` | [ACM FAccT, DOI 10.1145/3531146.3533234](https://doi.org/10.1145/3531146.3533234) | Location/hardware-sensitive carbon intensity context. |
| `holm1979` | [JSTOR original journal record](https://www.jstor.org/stable/4615733) | Sequential multiplicity correction used for predefined contrasts. |
| `efron1994bootstrap` | [CRC, DOI 10.1201/9780429246593](https://doi.org/10.1201/9780429246593) | Bootstrap interval and resampling foundation. |
| `demsar2006statistical` | [JMLR](https://jmlr.org/papers/v7/demsar06a.html) | Guidance for statistical comparison over multiple data sets/classifiers. |
| `nadeau2003inference` | [Springer, DOI 10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366) | Dependence and uncertainty in generalization-error comparison. |
| `bouthillier2021variance` | [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html) | Construction/training variance in ML benchmarks. |
| `pineau2021reproducibility` | [JMLR](https://jmlr.org/papers/v22/20-303.html) | Reproducibility artifacts, checklists, and executable evidence. |
| `kapoor2023leakage` | [Cell Press Patterns, DOI 10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804) | Recent leakage taxonomy and reproducibility implications. |
| `kaufman2011leakage` | [ACM KDD, DOI 10.1145/2020408.2020496](https://doi.org/10.1145/2020408.2020496) | Foundational data-mining leakage analysis. |
| `papadopoulos2023victor` | [Elsevier, DOI 10.1016/j.jvcir.2022.103741](https://doi.org/10.1016/j.jvcir.2022.103741) | Closest JVCIR task-specific Transformer with fashion contrastive pretraining and compute reporting. |
| `jung2025hat` | [ACM WSDM, DOI 10.1145/3701551.3703545](https://doi.org/10.1145/3701551.3703545) | Recent history-aware personalized outfit recommendation on Polyvore and IQON. |
| `sanny2026medal` | [CVF WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Sanny_MEDAL_multi-modal_MEta-space_Distillation_and_ALignment_for_Visual_Compatibility_Learning_WACV_2026_paper.html) | Recent task-specific meta-space distillation and alignment model for visual compatibility. |
| `zhai2025text2outfit` | [IEEE ICCV, DOI 10.1109/ICCV51701.2025.01500](https://doi.org/10.1109/ICCV51701.2025.01500) | Contemporary MLLM text-driven outfit generation and compatibility context. |
| `czerwinska2025ecommerce` | [Official arXiv:2504.07567](https://arxiv.org/abs/2504.07567) | Closest frozen image-embedding benchmark for e-commerce classification/retrieval and efficiency; the record notes acceptance at FTC 2025. |
| `feng2025mllmbenchmark` | [Springer MVA, DOI 10.1007/s00138-025-01762-0](https://doi.org/10.1007/s00138-025-01762-0) | Contemporary machine-vision precedent for systematic capability and application benchmarking across released multimodal models. |
| `chang2025large` | [Elsevier DSS, DOI 10.1016/j.dss.2025.114457](https://doi.org/10.1016/j.dss.2025.114457) | Recent Gemini-cue/BEiT3 outfit-compatibility system; constrains the novelty claim. |
| `pang2026crossselling` | [Elsevier ESWA, DOI 10.1016/j.eswa.2025.129686](https://doi.org/10.1016/j.eswa.2025.129686) | Recent higher-order compatibility and A100-grounded cross-selling system. |
| `feng2026fashionstylist` | [Official arXiv:2604.09249](https://arxiv.org/abs/2604.09249) | Expert-knowledge-enhanced styling data and MLLM evaluation. |
| `xue2026zooclaw` | [Official arXiv:2606.27708](https://arxiv.org/abs/2606.27708) | Distilled FashionSigLIP2 retrieval specialization and retrieval-bias audit. |
| `ma2024cirp` | [ACM MM, DOI 10.1145/3664647.3681349](https://doi.org/10.1145/3664647.3681349) | Cross-item relational pretraining for multimodal product bundling. |
| `liu2025bundlemllm` | [ACM KDD, DOI 10.1145/3690624.3709255](https://doi.org/10.1145/3690624.3709255) | Fine-tuned multimodal models for product bundling. |

## Coverage check

- Fashion compatibility, Polyvore, FITB, IQON3000, and A100: 38 records.
- Generic/fashion representations and representation benchmarks: 22 records, including CLIP, SigLIP/SigLIP2, DINO/DINOv3, FashionCLIP, Marqo-FashionSigLIP, GR-Lite, and LookBench.
- Calibration and uncertainty: 7 records.
- Computational efficiency: 5 records.
- Reproducibility, leakage, and statistical comparison: 8 records.
- Recent 2023--2026 items are included where directly relevant, while foundational dataset, architecture, calibration, bootstrap, and leakage sources are retained. The four category totals are overlapping thematic counts rather than a partition of the 81 records.
