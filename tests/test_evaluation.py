import numpy as np
import pytest

from repbench.eval.compatibility import cp_metrics


def test_cp_metrics_perfect_ranking():
    metrics = cp_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0


def test_cp_metrics_rejects_single_class_labels():
    with pytest.raises(ValueError, match="both positive and negative"):
        cp_metrics(np.array([1, 1]), np.array([0.2, 0.8]))
