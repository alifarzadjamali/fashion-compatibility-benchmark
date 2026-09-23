from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def cp_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not labels.size:
        raise ValueError("Labels and scores must be non-empty vectors with identical shapes")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    if np.unique(labels).size < 2:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
