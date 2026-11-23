import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from repbench.eval.paired_statistics import (
    _auc_layout,
    _weighted_auc,
    cluster_bootstrap,
    delong_covariance,
    delong_pvalue,
    exact_mcnemar,
    holm,
    pairwise_inference,
)


def test_weighted_auc_matches_sklearn_with_ties_and_zero_weights():
    rng = np.random.default_rng(31)
    labels = np.tile([0, 1], 30)
    scores = rng.integers(0, 6, size=len(labels)).astype(float)
    weights = rng.integers(0, 4, size=len(labels)).astype(float)
    assert _weighted_auc(_auc_layout(labels, scores), weights) == pytest.approx(
        roc_auc_score(labels, scores, sample_weight=weights), abs=1e-12
    )


def test_delong_points_match_auc_and_identical_models_have_p_one():
    labels = np.tile([0, 1], 12)
    scores = np.linspace(0, 1, len(labels))
    auc, covariance = delong_covariance(labels, np.column_stack([scores, scores, -scores]))
    np.testing.assert_allclose(
        auc, [roc_auc_score(labels, scores)] * 2 + [roc_auc_score(labels, -scores)]
    )
    assert delong_pvalue(auc, covariance, 0, 1) == 1.0
    assert np.linalg.eigvalsh(covariance).min() > -1e-12


def test_exact_mcnemar_and_holm_known_values():
    assert exact_mcnemar(np.ones(5, bool), np.zeros(5, bool))["pvalue"] == pytest.approx(0.0625)
    assert exact_mcnemar([True, False], [True, False])["pvalue"] == 1.0
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_bootstrap_preserves_model_pairing_and_finds_equivalence_for_identical_models():
    values = np.column_stack([np.tile([0, 1], 20)] * 2)
    bootstrap = cluster_bootstrap(
        values, metric="accuracy", groups=np.repeat(np.arange(20), 2), replicates=199, seed=42
    )
    np.testing.assert_array_equal(bootstrap["samples"][:, 0], bootstrap["samples"][:, 1])
    report = pairwise_inference(bootstrap, ["a", "b"], metric="accuracy", margins=[0.01])
    comparison = report["comparisons"][0]
    assert comparison["delta"] == comparison["ci_low"] == comparison["ci_high"] == 0.0
    assert comparison["bootstrap_p_holm"] == 1.0
    assert report["equivalence"][0]["equivalent_at_005"]


def test_two_way_bootstrap_reports_both_cluster_counts_and_is_seeded():
    values = np.column_stack([np.tile([0, 1], 12), np.tile([1, 0], 12)])
    kwargs = {
        "metric": "accuracy",
        "groups": np.repeat(np.arange(6), 4),
        "second_groups": np.tile(np.arange(4), 6),
        "replicates": 30,
        "seed": 7,
    }
    first, second = cluster_bootstrap(values, **kwargs), cluster_bootstrap(values, **kwargs)
    np.testing.assert_array_equal(first["samples"], second["samples"])
    assert first["cluster_count"] == 6
    assert first["second_cluster_count"] == 4
