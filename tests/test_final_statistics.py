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
