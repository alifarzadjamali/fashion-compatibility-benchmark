"""Calibration metrics with explicit, testable bin semantics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(labels, probabilities, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("Labels and probabilities must be aligned vectors")
    if n_bins < 2 or not np.isfinite(probabilities).all():
        raise ValueError("ECE requires finite probabilities and at least two bins")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    # Right-closed final bin ensures p=1 is included; p=0 belongs to bin zero.
    indices = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)


def calibration_metrics(labels, probabilities, primary_bins: int = 15) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece_10": expected_calibration_error(labels, probabilities, 10),
        f"ece_{primary_bins}": expected_calibration_error(labels, probabilities, primary_bins),
        "ece_20": expected_calibration_error(labels, probabilities, 20),
    }


def reliability_bins(labels, probabilities, n_bins: int = 15) -> list[dict]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    indices = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = indices == bin_index
        rows.append(
            {
                "bin": bin_index,
                "lower": bin_index / n_bins,
                "upper": (bin_index + 1) / n_bins,
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                "empirical_positive_rate": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    return rows
