from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def cp_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must have identical shapes")
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
