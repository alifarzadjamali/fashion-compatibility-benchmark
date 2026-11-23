"""Descriptive rank statistics across settings (HANDOFF_V2 §6.3): Friedman test,
Nemenyi critical difference, average ranks and Kendall's W (Demšar 2006)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f, rankdata, studentized_range

LABEL = "descriptive"


def friedman_nemenyi(scores, models, blocks=None, *, alpha: float = 0.05,
                     higher_is_better: bool = True) -> dict:
    """``scores``: (n_blocks, n_models); blocks = settings/datasets, treatments = models.

    Rank 1 is best; ties receive average ranks and the Friedman statistic is tie-corrected
    (matches scipy.stats.friedmanchisquare). Kendall's W = chi2_F / (N (k - 1)).
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Scores must be a finite (n_blocks, n_models) matrix")
    n, k = x.shape
    if n < 2 or k < 3:
        raise ValueError("Friedman needs at least 2 blocks and 3 models")
    models = list(models)
    blocks = list(blocks) if blocks is not None else [f"block_{i}" for i in range(n)]
    if len(models) != k or len(set(models)) != k or len(blocks) != n:
        raise ValueError("Model/block labels must be unique and match the score matrix")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    ranks = rankdata(-x if higher_is_better else x, axis=1)
    mean_ranks = ranks.mean(axis=0)
    ties = sum(float(np.sum(c**3 - c)) for c in
               (np.unique(row, return_counts=True)[1] for row in ranks))
    correction = 1 - ties / (n * k * (k**2 - 1))
    statistic = (12 * n / (k * (k + 1)) * np.sum((mean_ranks - (k + 1) / 2) ** 2)) / correction
    w = statistic / (n * (k - 1))
    iman = (n - 1) * statistic / (n * (k - 1) - statistic) if statistic < n * (k - 1) else np.inf
    q = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    diff = np.abs(mean_ranks[:, None] - mean_ranks[None, :])
    return {
        "label": LABEL, "models": models, "blocks": blocks, "n_blocks": n, "n_models": k,
        "higher_is_better": higher_is_better,
        "ranks": ranks.tolist(), "average_ranks": dict(zip(models, mean_ranks.tolist())),
        "friedman_chi2": float(statistic), "friedman_df": k - 1,
        "friedman_pvalue": float(chi2.sf(statistic, k - 1)),
        "iman_davenport_f": float(iman),
        "iman_davenport_pvalue": float(f.sf(iman, k - 1, (k - 1) * (n - 1))),
        "kendall_w": float(w), "alpha": alpha, "nemenyi_q": q, "critical_difference": float(cd),
        "nemenyi_significant": (diff > cd).astype(int).tolist(),
    }


def plot_critical_difference(result: dict, path) -> None:
    """Simple CD diagram: models on the average-rank axis, CD bar, cliques joined."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    items = sorted(result["average_ranks"].items(), key=lambda kv: kv[1])
    k, cd = result["n_models"], result["critical_difference"]
    fig = Figure(figsize=(7, 1.2 + 0.3 * k))
    FigureCanvasAgg(fig)  # Agg canvas; never touches the global pyplot backend
    ax = fig.subplots()
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(k + 2), 2)
    ax.axhline(0, color="black", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.15], color="black", lw=1)
        ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=8)
    ax.plot([1, 1 + cd], [1.3, 1.3], color="black", lw=2)
    ax.text(1 + cd / 2, 1.45, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (name, rank) in enumerate(items):
        y = -(i + 1)
        ax.plot([rank, rank], [0, y], color="gray", lw=0.8)
        ax.text(k + 0.45, y, f"{name} ({rank:.2f})", ha="right", va="center", fontsize=8)
    ranks = [r for _, r in items]
    cliques = [(i, j) for i in range(k) for j in range(k)
               if j > i and ranks[j] - ranks[i] <= cd
               and not (i > 0 and ranks[j] - ranks[i - 1] <= cd)
               and not (j < k - 1 and ranks[j + 1] - ranks[i] <= cd)]
    for level, (i, j) in enumerate(cliques):
        y = -0.5 - 0.25 * level
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [y, y], color="black", lw=3)
    ax.set_title(f"Average rank ({result['label']}; Nemenyi alpha={result['alpha']})",
                 fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
