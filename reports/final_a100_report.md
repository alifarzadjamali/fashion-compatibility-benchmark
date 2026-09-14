# Final A100 external-evaluation report

A100 was used unchanged as an external human/expert-grounded benchmark. No model component, C value,
threshold, calibration map, negative, candidate, or methodological decision was fitted to it. All
1,663 referenced images are present and decodable. Exact-byte and exact-decoded-pixel audits found
zero overlap with Polyvore and IQON; transformed near-duplicates are not ruled out.

LAT primary accuracy uses `argmax(gt_distribution)` and mLAT uses the selected answer's vote share.
Archive-`gt` accuracy is mandatory sensitivity because questions 92, 93, 94, 96, 98, 99, and 100
disagree with the vote-distribution majority. AAT is reported separately overall and for Color,
Style, Occasion, Season, Material, and Balance.

Polyvore-trained CLIP/Marqo/FashionCLIP obtain LAT .60/.62/.58, compared with .37 ResNet50. Their
mLAT values are .493/.501/.481. Polyvore-trained CLIP leads AAT at .67. IQON-trained Marqo and
FashionCLIP obtain LAT .58/.56, while SigLIP2 and Marqo tie at .63 AAT. These results corroborate
the VLM advantage, but most close-model intervals overlap.

The image-only OutfitTransformer transfers less well: across five seeds it averages .49 LAT/.34 AAT
from Polyvore and .418/.382 from IQON. Its strong IQON synthetic CP result therefore does not imply
stronger human-grounded compatibility performance.

The archive has an academic-use dataset page but no formal license file. The project records this as
a redistribution limitation and does not claim broader licensing rights. See
`reports/tables/final/a100_headline.csv`, `reports/tables/final/a100_aat_dimensions.csv`, and
`reports/tables/final/task_specific_baseline_a100_aggregate.csv`.

