import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.final_phase_statistics import N_BOOTSTRAP, holm, weighted_binary_bootstrap


def test_weighted_bootstrap_metrics_match_unweighted_reference():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.asarray([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])
    counts = np.ones((N_BOOTSTRAP, len(labels)), dtype=np.int16)
    group_index = np.arange(len(labels))
    auc, average_precision = weighted_binary_bootstrap(labels, scores, counts, group_index)
    np.testing.assert_allclose(auc, roc_auc_score(labels, scores))
    np.testing.assert_allclose(average_precision, average_precision_score(labels, scores))


def test_weighted_bootstrap_metrics_preserve_tie_semantics():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.asarray([0.1, 0.7, 0.7, 0.7, 0.9, 0.1])
    counts = np.ones((N_BOOTSTRAP, len(labels)), dtype=np.int16)
    auc, average_precision = weighted_binary_bootstrap(
        labels, scores, counts, np.arange(len(labels))
    )
    np.testing.assert_allclose(auc, roc_auc_score(labels, scores))
    np.testing.assert_allclose(average_precision, average_precision_score(labels, scores))


def test_holm_is_monotone_in_sorted_raw_p_values():
    raw = np.asarray([0.04, 0.001, 0.02, 0.8])
    adjusted = holm(raw)
    ordered = adjusted[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= 0)
    assert np.all((adjusted >= raw) & (adjusted <= 1))


def test_cluster_interval_join_uses_dataset_and_model_keys():
    import pandas as pd
    import pytest

    from scripts.cluster_bootstrap import attach_cluster_intervals

    summary = pd.DataFrame({"dataset": ["a", "b"], "representation": ["model", "model"]})
    intervals = pd.DataFrame([
        {"dataset": dataset, "representation": "model", "metric": metric,
         "ci_low": value, "ci_high": value + 0.1}
        for dataset, value in (("b", 0.7), ("a", 0.5))
        for metric in ("cp_auc", "pr_auc", "fitb_acc")
    ])
    joined = attach_cluster_intervals(summary, intervals)
    assert joined.cp_auc_cluster_low.tolist() == [0.5, 0.7]
    assert joined.fitb_acc_cluster_high.tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError, match="Missing cluster intervals"):
        attach_cluster_intervals(summary, intervals.iloc[1:])
    with pytest.raises(pd.errors.MergeError):
        attach_cluster_intervals(summary, pd.concat([intervals, intervals.iloc[:1]]))
