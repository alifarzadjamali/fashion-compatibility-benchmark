# Final statistical report

All primary intervals use 2,000 paired resamples. Polyvore uses connected-component-aware
resampling; IQON uses user-cluster resampling to address repeated outfits from the same creator.
The six comparisons were fixed before IQON/A100 evaluation, and Holm correction was applied within
each dataset/metric family. Effect size, uncertainty, adjusted evidence, and practical magnitude are
reported separately.

The strongest stable findings are:

- DINOv3 exceeds ResNet50 on Polyvore CP by .0885 [.0817,.0952] and FITB by 9.85 points
  [8.82,10.88]; on IQON the effects are .0566 [.0508,.0626] and 4.94 points [4.16,5.80].
- GR-Lite exceeds DINOv3 on Polyvore by .0455 CP and 5.25 FITB points, and on IQON by .0092
  and 1.72 points. All four Holm-adjusted results are supported.
- FashionCLIP trails CLIP on Polyvore by .0193 CP and 3.25 FITB points and on IQON by .0062
  and .89 points; all predefined adjusted comparisons are supported.
- Marqo versus SigLIP2 is indistinguishable on Polyvore-D-Clean but favors Marqo on IQON by
  .0190 CP [.0144,.0234] and 1.21 FITB points [.50,1.90].
- The validation-selected best model exceeds ResNet50 by .1959 CP and 24.16 FITB points on
  Polyvore, and .1369 and 12.85 points on IQON.

A100 has only 100 questions per task. The clearest adjusted effects are the best overall versus
ResNet50 (23-34 LAT points and 32 AAT points depending on training source). Most close model
comparisons remain inconclusive. A100 is therefore corroborating human-grounded evidence, not a
high-power ranking benchmark.

Rankings are larger than benchmark-construction variability: across five constructions, CP-rank
Spearman is .893-1.0 on Polyvore and .964-1.0 on IQON; FITB is .964-1.0. PCA-256 versus native IQON
rank agreement is rho=.857/tau=.714 for both CP and FITB. LR versus XGBoost is rho=.964/tau=.905
for CP and 1.0/1.0 for FITB.

Full results: `reports/tables/final/predefined_statistical_effects.csv`,
`reports/tables/final/rank_correlations.csv`, and
`reports/tables/final/construction_seed_rank_agreement.csv`.

