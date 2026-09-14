"""Produce calibration tables and reliability diagrams for both primary CP benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from repbench.eval.calibration import calibration_metrics, reliability_bins

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_predictions(dataset: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "polyvore_d_clean":
        frame = pd.read_parquet("artifacts/clean_primary_cp_predictions.parquet")
        frame = frame[frame.model_key == model].sort_values("example_index")
        return frame.label.to_numpy(np.int8), frame.score.to_numpy(float)
    frame = pd.read_parquet(f"artifacts/final/iqon3000_clean/{model}/cp_predictions_test.parquet")
    frame = frame.sort_values("example_id")
    return frame.label.to_numpy(np.int8), frame.probability.to_numpy(float)


def main() -> None:
    output = Path("artifacts/final/calibration")
    figure_root = Path("reports/figures/final")
    output.mkdir(parents=True, exist_ok=False)
    figure_root.mkdir(parents=True, exist_ok=True)
    rows, bin_rows = [], []
    for dataset in ("polyvore_d_clean", "iqon3000_clean"):
        for model in PRIMARY_MODEL_KEYS:
            labels, probabilities = load_predictions(dataset, model)
            metrics = calibration_metrics(labels, probabilities)
            rows.append(
                {
                    "dataset": dataset,
                    "representation": model,
                    "cp_auc": roc_auc_score(labels, probabilities),
                    **metrics,
                }
            )
            bin_rows.extend(
                {"dataset": dataset, "representation": model, **row}
                for row in reliability_bins(labels, probabilities, 15)
            )
    metrics = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    metrics["auc_rank"] = metrics.groupby("dataset").cp_auc.rank(ascending=False, method="min")
    metrics["brier_rank"] = metrics.groupby("dataset").brier.rank(ascending=True, method="min")
    metrics.to_csv(output / "calibration_metrics.csv", index=False)
    bins.to_csv(output / "reliability_bins_15.csv", index=False)
    correlations = []
    for dataset, frame in metrics.groupby("dataset"):
        rho = spearmanr(frame.cp_auc, -frame.brier)
        tau = kendalltau(frame.cp_auc, -frame.brier)
        correlations.append(
            {
                "dataset": dataset,
                "comparison": "CP AUC versus inverse Brier ranking",
                "spearman_rho": rho.statistic,
                "spearman_p": rho.pvalue,
                "kendall_tau": tau.statistic,
                "kendall_p": tau.pvalue,
            }
        )
    pd.DataFrame(correlations).to_csv(output / "predictive_vs_calibration_rank.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, dataset in zip(axes, ("polyvore_d_clean", "iqon3000_clean"), strict=True):
        for model in PRIMARY_MODEL_KEYS:
            frame = bins[(bins.dataset == dataset) & (bins.representation == model)]
            frame = frame[frame["count"] > 0]
            axis.plot(
                frame.mean_probability,
                frame.empirical_positive_rate,
                marker="o",
                markersize=2,
                label=model,
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(dataset.replace("_", " "))
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed positive fraction")
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figure_root / "reliability_diagrams.png", dpi=240)
    fig.savefig(figure_root / "reliability_diagrams.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
