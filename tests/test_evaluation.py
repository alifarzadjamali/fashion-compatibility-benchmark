import numpy as np

from repbench.eval.compatibility import cp_metrics


def test_cp_metrics_perfect_ranking():
    metrics = cp_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0
