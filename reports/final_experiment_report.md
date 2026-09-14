# RepBench-Fashion final experiment report

## Scope and protocol

The completed study compares exactly seven frozen image representations under one locked PCA-256,
pair-feature, and Logistic Regression probe. Historical Polyvore-D is retained only for literature
comparability. Polyvore-D-Clean and IQON3000-Clean enforce zero item-ID, exact encoded-image,
decoded-pixel, and outfit overlap. A100 is strictly external: no fitting, tuning, calibration,
candidate replacement, or negative generation used its data. The signed protocol and configuration
are in `docs/final_experimental_protocol.md`, `configs/final_experimental_protocol.yaml`, and
`artifacts/final_experimental_protocol_hash.json`.

## Headline leakage-safe results

Values are test CP ROC-AUC / FITB accuracy; brackets are 95% group-aware bootstrap intervals.

| Representation | Polyvore-D-Clean CP | Polyvore FITB | IQON3000-Clean CP | IQON FITB |
|---|---:|---:|---:|---:|
| ResNet50 | .6938 [.6881,.6997] | .4063 [.3981,.4145] | .5852 [.5798,.5909] | .3166 [.3097,.3233] |
| DINOv3 ViT-L/16 | .7822 [.7772,.7873] | .5048 [.4972,.5134] | .6419 [.6362,.6475] | .3660 [.3587,.3739] |
| CLIP ViT-L/14@336 | .8883 [.8843,.8919] | .6500 [.6424,.6576] | .7042 [.6977,.7108] | .4364 [.4281,.4444] |
| SigLIP2-B/16@384 | .8892 [.8857,.8927] | **.6528** [.6454,.6602] | .7031 [.6970,.7093] | .4330 [.4254,.4413] |
| FashionCLIP 2.0 | .8690 [.8648,.8730] | .6175 [.6099,.6250] | .6981 [.6920,.7042] | .4275 [.4193,.4358] |
| Marqo-FashionSigLIP | **.8897** [.8859,.8931] | .6479 [.6408,.6557] | **.7221** [.7158,.7282] | **.4451** [.4371,.4539] |
| GR-Lite | .8277 [.8232,.8323] | .5573 [.5498,.5657] | .6511 [.6440,.6580] | .3832 [.3757,.3911] |

The two clean protocols strongly agree: CP rank Spearman rho=.964/Kendall tau=.905 and FITB
rho=.857/tau=.714. Absolute scores fall substantially on IQON, but the broad ordering survives.
All selected C values and PR-AUC intervals are in
`reports/tables/final/main_polyvore_iqon_results.csv`; unchanged historical results are in
`reports/tables/final/historical_polyvore_results.csv`.

## External and task-specific validation

On 100-question A100 LAT, Polyvore-trained Marqo scored .62 [.52,.72] and IQON-trained Marqo .58
[.48,.68]; corresponding mLAT values were .501 and .474. On 100-question AAT, the strongest scores
were .67 [.58,.76] for Polyvore-trained CLIP and .63 [.54,.72] for IQON-trained SigLIP2/Marqo.
Intervals are wide, so small A100 differences are not decisive. Archive-`gt` sensitivity and all six
AAT facets are retained in `reports/tables/final/a100_headline.csv` and
`reports/tables/final/a100_aat_dimensions.csv`.

The fixed image-only OutfitTransformer adaptation averaged .8095±.0020 CP AUC and .4655±.0051 FITB
on Polyvore, versus .7723±.0013 and .4354±.0012 on IQON across five training seeds. It beats every
frozen probe on IQON CP, but not IQON FITB, and trails the leading VLM probes on Polyvore and A100.
This is contextual evidence, not an exact reproduction or SOTA claim.

## Subgroups and failure signal

All seven models find two-item outfits hardest. Mean CP AUC across models rises from .721 to .888
between length two and length six-plus on Polyvore, and from .602 to .758 on IQON. FITB shows a
smaller but similar pattern, especially on IQON (.355 to .442). This is descriptive rather than
causal: longer outfits provide more pair evidence and may make deterministic corruption easier to
detect. Category and length subgroup sample sizes and results are preserved in
`reports/tables/final/subgroup_performance.csv`; no tiny subgroup is used for a headline claim.
Model disagreement is substantial (for example, 20,200/24,800 IQON FITB questions), while all seven
fail on 965 Polyvore and 3,497 IQON FITB questions. These cases are retained for qualitative paper
inspection in `reports/tables/final/failure_consensus_audit.csv`; no post-hoc method change used them.

## Interpretation

Modern contrastive vision-language representations consistently outperform supervised ResNet50
and generic self-supervised DINOv3 under the controlled probe. Fashion specialization is not a
uniform advantage: Marqo is strongest on IQON and competitive on Polyvore, while FashionCLIP is
significantly below CLIP on IQON. After excluding Marqo and GR-Lite for uncertain fashion-corpus
provenance, generic CLIP/SigLIP variants lead both clean protocols. GR-Lite improves over DINOv3,
but checkpoint/preprocessing and training-corpus differences prevent interpreting it as a pure
fine-tuning ablation.

The complete tables and figures are under `reports/tables/final/` and `reports/figures/final/`.
