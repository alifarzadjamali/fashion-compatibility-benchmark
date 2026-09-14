# Final calibration report

Calibration was evaluated without optimizing the primary models for calibration. ECE uses 15
equal-width bins as primary, with 10- and 20-bin sensitivity; Brier score, log loss, and reliability
diagrams are also reported. No test- or A100-fitted calibration was performed, and FITB candidate
scores are not described as calibrated probabilities.

On Polyvore-D-Clean, ECE-15 ranges from .0288 (ResNet50) to .0372 (SigLIP2); on IQON it ranges from
.0415 (ResNet50) to .0582 (Marqo). The higher-AUC models generally improve Brier score and log loss,
but do not necessarily minimize ECE. Marqo has the best Brier score on both datasets while its IQON
ECE is the largest, illustrating why discrimination and calibration are separate properties.

The principal representation ordering is therefore not a calibration artifact, but the best
discriminator should not automatically be presented as the best probability estimator. Results are
in `artifacts/final/calibration/calibration_metrics.csv`; diagrams are
`reports/figures/final/reliability_diagrams.pdf`.

