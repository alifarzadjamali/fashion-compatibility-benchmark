"""Render reliability diagrams from the preserved calibration table."""

from pathlib import Path

import matplotlib
import pandas as pd

from repbench.encoders.registry import PRIMARY_MODEL_KEYS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    bins = pd.read_csv("artifacts/final/calibration/reliability_bins_15.csv")
    output = Path("reports/figures/final")
    output.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output / "reliability_diagrams.png", dpi=240)
    fig.savefig(output / "reliability_diagrams.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
