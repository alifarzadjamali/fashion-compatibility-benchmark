# Closest contemporary prior work audit

Verified through primary paper, publisher, conference, or arXiv records on 14 September 2026. “Frozen comparison” means that comparing released image representations is a central experimental intervention; an open circle denotes partial backbone or adjacent embedding analysis. Absence of a dimension means it is not a central reported contribution, not that the authors ignored validity.

| Study | Status / primary record | Frozen comparison | Task-specific training | Polyvore | IQON | External human/expert task | Leakage audit | Retrieval vs compatibility | Calibration | Efficiency | Why it matters here |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VICTOR | JVCIR 90 (2023), 103741, [DOI](https://doi.org/10.1016/j.jvcir.2022.103741) | ○ | ● | ● | — | — | — | — | — | ● | Closest task-specific Transformer plus fashion contrastive-pretraining study; it does not provide our cross-corpus frozen-probe audit. |
| OutfitTransformer | WACV 2023, [IEEE DOI](https://doi.org/10.1109/WACV56688.2023.00359) | — | ● | ● | — | — | — | — | — | — | Canonical contextual-set baseline. Our implementation is explicitly an image-only adaptation, not a reproduction or SOTA comparison. |
| HAT / History-Aware Transformers | WSDM 2025, [ACM DOI](https://doi.org/10.1145/3701551.3703545) | — | ● | ● | ● | — | — | — | — | — | Closest recent Polyvore/IQON personalized system; personalization and protocols prevent direct numerical comparison. |
| MEDAL | WACV 2026, [CVF paper](https://openaccess.thecvf.com/content/WACV2026/html/Sanny_MEDAL_multi-modal_MEta-space_Distillation_and_ALignment_for_Visual_Compatibility_Learning_WACV_2026_paper.html) | ○ | ● | ● | — | — | — | — | — | — | Contemporary meta-space distillation and alignment; task-specific rather than a common frozen checkpoint probe. |
| Text2Outfit | ICCV 2025, [IEEE DOI](https://doi.org/10.1109/ICCV51701.2025.01500) | — | ● | ● | — | — | — | — | — | — | MLLM text-driven outfit generation; relevant modern multimodal compatibility work but a different task and input. |
| LMLMO / BEiT3 compatibility | Decision Support Systems 194 (2025), 114457, [DOI](https://doi.org/10.1016/j.dss.2025.114457) | ○ | ● | — | — | — | — | — | — | — | Uses generated fashion cues and BEiT3 for FashionVC/EVALUATION3 pair compatibility; shows modern representations already enter systems, narrowing our novelty wording. |
| HON journal extension / A100 | Expert Systems with Applications 298 (2026), 129686, [DOI](https://doi.org/10.1016/j.eswa.2025.129686) | — | ● | ● | — | ● | — | — | — | — | Closest recent use of A100 in a higher-order task-specific compatibility/cross-selling system. |
| Benchmarking Image Embeddings for E-Commerce | arXiv:2504.07567; accepted FTC 2025, [record](https://arxiv.org/abs/2504.07567) | ● | ○ | — | — | — | — | ● | — | ● | Closest general frozen-embedding benchmark for e-commerce classification/retrieval; not relational outfit compatibility. |
| LookBench v3 | arXiv:2601.14706, revised 13 Apr 2026, [record](https://arxiv.org/abs/2601.14706) | ● | ○ | — | — | — | ○ | ● | — | ● | Supplies the exact five-checkpoint retrieval overlap; its construct is retrieval and overlap is too small for population inference. |
| ZooClaw-FashionSigLIP2 | arXiv:2606.27708, [record](https://arxiv.org/abs/2606.27708) | ○ | ● | — | — | — | ● | ● | — | ○ | Recent distilled fashion-retrieval model and bias audit; relevant to model/data recipe dependence. |
| FashionStylist | arXiv:2604.09249, [record](https://arxiv.org/abs/2604.09249) | — | ● | — | — | ● | ○ | ○ | — | — | Expert-knowledge-enhanced styling data and MLLM evaluation broaden external validity but postdate and differ from our fixed A100 evidence. |
| Multimodal product bundling | KDD 2025, [ACM DOI](https://doi.org/10.1145/3690624.3709255) | — | ● | — | — | — | — | ○ | — | — | Fine-tuned multimodal product bundling; different objective and behavioural setting. |
| RepBench-Fashion | This manuscript | ● | one contextual baseline | ● | ● | ● | ● | ● | ● | ● | Controlled transfer measurement plus validity, operational, and provenance triangulation. |

## Defensible novelty position

Prior work has incorporated, adapted, or fine-tuned modern visual and multimodal representations within compatibility, styling, bundle-recommendation, and fashion-retrieval systems. The defensible gap is not “first use of modern representations.” It is the insufficient characterization of comparative transfer among contemporary released frozen checkpoints under one relational probe across item- and exact-image-disjoint corpora, a prospectively designated replication, unchanged external human/expert-grounded questions, question/leakage/selection audits, protocol calibration, efficiency, provenance, and construction robustness.

## Why published scores are not copied into a leaderboard

The closest systems change multiple factors simultaneously: historical versus disjoint splits; pair, set, generation, or personalized objectives; text and behavioural modalities; numbers and sources of distractors; synthetic-negative rules; candidate counts; end-to-end training; and model-selection procedures. A numeric side-by-side table would imply exchangeability that the protocols do not support. The manuscript instead compares study dimensions honestly and uses the one implemented contextual adaptation only as diagnostic evidence.

## Version notes

- LookBench values used in the manuscript are version 3 Fine Recall@1: DINOv3 43.97, CLIP 39.79, SigLIP2 59.44, Marqo-FashionSigLIP 62.77, and GR-Lite 65.71. Checkpoint identity was matched against the released model names; only these five are exact overlaps.
- MEDAL and the 2026 arXiv papers are identified as 2026 work, not retroactively described as available when the experiment roster was frozen.
- DINOv3 and SigLIP2 remain cited through their primary technical records; the manuscript does not invent journal publication metadata.
