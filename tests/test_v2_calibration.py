import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import log_loss

from repbench.eval.calibration import expected_calibration_error
from repbench.v2.calibration_v2 import (
    _bin_indices,
    _weighted_ece,
    apply_temperature,
    brier_decomposition,
    calibrate_and_evaluate,
    ece_bootstrap_ci,
    fit_platt,
    fit_temperature,
)


def _overconfident(seed, n=4000, scale=3.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    labels = (rng.random(n) < expit(z)).astype(int)
    return labels, expit(scale * z)


def test_temperature_recovers_scale_and_reduces_log_loss():
    fit_y, fit_p = _overconfident(0)
    eval_y, eval_p = _overconfident(1)
    t = fit_temperature(fit_y, fit_p)
    assert 2.6 < t < 3.4
    assert log_loss(eval_y, apply_temperature(eval_p, t)) < log_loss(eval_y, eval_p)
    for method in ("temperature", "platt"):
        out = calibrate_and_evaluate(fit_y, fit_p, eval_y, eval_p, method=method,
                                     replicates=200)
        assert out["after"]["log_loss"] < out["before"]["log_loss"]
        assert out["after"]["ece_15"] < out["before"]["ece_15"]
        assert {"ece_10", "ece_15", "ece_20", "brier", "brier_decomposition",
                "ece_15_ci"} <= set(out["before"])


def test_platt_fit_uses_fit_set_only():
    fit_y, fit_p = _overconfident(0)
    a, b = fit_platt(fit_y, fit_p)
    assert 0.25 < a < 0.4 and abs(b) < 0.15
    e1 = calibrate_and_evaluate(fit_y, fit_p, *_overconfident(1), method="platt", replicates=10)
    e2 = calibrate_and_evaluate(fit_y, fit_p, *_overconfident(2), method="platt", replicates=10)
    assert e1["params"] == e2["params"]


def test_brier_decomposition_identity():
    y, p = _overconfident(3)
    d = brier_decomposition(y, p, n_bins=10)
    assert abs(d["residual"]) < 0.01
    assert d["reliability"] - d["resolution"] + d["uncertainty"] + d["residual"] == \
        pytest.approx(d["brier"])
    # Exact when forecasts are constant within bins.
    centers = (np.minimum((p * 10).astype(int), 9) + 0.5) / 10
    assert abs(brier_decomposition(y, centers, n_bins=10)["residual"]) < 1e-12


def test_ece_bootstrap_matches_point_estimate_and_clusters():
    y, p = _overconfident(4, n=1000)
    idx = _bin_indices(p, 15)
    assert _weighted_ece(y.astype(float), p, idx, np.ones(len(y)), 15) == pytest.approx(
        expected_calibration_error(y, p, 15))
    plain = ece_bootstrap_ci(y, p, replicates=300, seed=7)
    assert plain == ece_bootstrap_ci(y, p, replicates=300, seed=7)
    assert plain["ci_low"] <= plain["ece"] <= plain["ci_high"]
    groups = np.arange(len(y)) // 50
    clustered = ece_bootstrap_ci(y, p, groups=groups, replicates=300, seed=7)
    assert clustered["clustered"] and clustered["n_clusters"] == 20
    with pytest.raises(ValueError):
        ece_bootstrap_ci(y, p, groups=np.zeros(len(y)))
    with pytest.raises(ValueError):
        fit_temperature(np.ones(10), np.full(10, 0.5))
    with pytest.raises(ValueError):
        calibrate_and_evaluate(y, p, y, p, method="isotonic")
