"""V2 calibration (HANDOFF_V2 §6.7): post-hoc temperature/Platt scaling fitted on a
validation set only, Murphy Brier decomposition, and (cluster) bootstrap CIs for ECE."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from repbench.eval.calibration import (
    _bin_indices,
    _validate_calibration_inputs,
    calibration_metrics,
)

EPS = 1e-12  # probabilities are clipped to [EPS, 1-EPS] before conversion to logits
LOG_T_BOUNDS = (np.log(1e-2), np.log(1e2))


def _logits(labels, probabilities):
    labels, probabilities = _validate_calibration_inputs(labels, probabilities, 2)
    if not labels.size or len(np.unique(labels)) != 2 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Calibration fitting needs binary labels with both classes")
    return labels.astype(np.float64), logit(np.clip(probabilities, EPS, 1 - EPS))


def _nll(labels, z):
    # Stable binary cross-entropy on logits.
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def fit_temperature(labels, probabilities) -> float:
    """Single temperature T>0 minimising NLL of sigmoid(logit(p)/T); fails at the bound."""
    y, z = _logits(labels, probabilities)
    res = minimize_scalar(lambda lt: _nll(y, z / np.exp(lt)), bounds=LOG_T_BOUNDS,
                          method="bounded", options={"xatol": 1e-10})
    if not res.success or min(abs(res.x - b) for b in LOG_T_BOUNDS) < 1e-6:
        raise ValueError("Temperature optimisation failed or hit the [0.01, 100] bound")
    return float(np.exp(res.x))


def fit_platt(labels, probabilities) -> tuple[float, float]:
    """Unregularised Platt scaling sigmoid(a*logit(p)+b) by MLE on the hard labels."""
    y, z = _logits(labels, probabilities)

    def objective(params):
        a, b = params
        s = a * z + b
        grad = expit(s) - y
        return _nll(y, s), np.array([np.mean(grad * z), np.mean(grad)])

    res = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="BFGS",
                   options={"gtol": 1e-10, "maxiter": 1000})
    if not np.isfinite(res.x).all():
        raise ValueError("Platt scaling did not converge")
    return float(res.x[0]), float(res.x[1])


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    p = np.asarray(probabilities, dtype=np.float64)
    return expit(logit(np.clip(p, EPS, 1 - EPS)) / temperature)


def apply_platt(probabilities, a: float, b: float) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    return expit(a * logit(np.clip(p, EPS, 1 - EPS)) + b)


def brier_decomposition(labels, probabilities, n_bins: int = 10) -> dict:
    """Murphy (1973) decomposition over equal-width bins (same bins as ECE; p=1 in last bin).

    brier = reliability - resolution + uncertainty + residual, where the residual is the
    within-bin term that vanishes when forecasts are constant within each bin.
    """
    y, p = _validate_calibration_inputs(labels, probabilities, n_bins)
    if not y.size:
        raise ValueError("Empty input")
    y = y.astype(np.float64)
    idx = _bin_indices(p, n_bins)
    n = np.bincount(idx, minlength=n_bins).astype(np.float64)
    occupied = n > 0
    pbar = np.bincount(idx, p, n_bins)[occupied] / n[occupied]
    obar = np.bincount(idx, y, n_bins)[occupied] / n[occupied]
    w, base = n[occupied] / y.size, y.mean()
    brier = float(np.mean((p - y) ** 2))
    reliability = float(np.sum(w * (pbar - obar) ** 2))
    resolution = float(np.sum(w * (obar - base) ** 2))
    uncertainty = float(base * (1 - base))
    return {"brier": brier, "reliability": reliability, "resolution": resolution,
            "uncertainty": uncertainty,
            "residual": brier - (reliability - resolution + uncertainty),
            "n_bins": n_bins, "binning": "equal-width, right-closed final bin"}


def _weighted_ece(y, p, idx, weights, n_bins):
    gaps = np.bincount(idx, weights * (p - y), n_bins)
    return float(np.abs(gaps).sum() / weights.sum())


def ece_bootstrap_ci(labels, probabilities, *, n_bins: int = 15, groups=None,
                     replicates: int = 2000, seed: int = 20260912, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for ECE; with ``groups`` whole clusters are resampled."""
    y, p = _validate_calibration_inputs(labels, probabilities, n_bins)
    if not y.size or replicates < 1 or not 0 < alpha < 1:
        raise ValueError("Need data, replicates >= 1 and alpha in (0, 1)")
    y = y.astype(np.float64)
    idx = _bin_indices(p, n_bins)
    if groups is None:
        codes, n_clusters = np.arange(y.size), y.size
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("One group ID is required per example")
        _, codes = np.unique(groups, return_inverse=True)
        n_clusters = int(codes.max()) + 1
        if n_clusters < 2:
            raise ValueError("At least two clusters are required")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for r in range(replicates):
        counts = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
        draws[r] = _weighted_ece(y, p, idx, counts[codes].astype(np.float64), n_bins)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"ece": _weighted_ece(y, p, idx, np.ones_like(y), n_bins), "ci_low": float(lo),
            "ci_high": float(hi), "n_bins": n_bins, "replicates": replicates, "seed": seed,
            "clustered": groups is not None, "n_clusters": int(n_clusters)}


def _summary(labels, probabilities, groups, replicates, seed):
    out = calibration_metrics(labels, probabilities, primary_bins=15)
    out["brier_decomposition"] = brier_decomposition(labels, probabilities, n_bins=10)
    out["ece_15_ci"] = ece_bootstrap_ci(labels, probabilities, n_bins=15, groups=groups,
                                        replicates=replicates, seed=seed)
    return out


def calibrate_and_evaluate(fit_labels, fit_probabilities, eval_labels, eval_probabilities, *,
                           method: str = "temperature", eval_groups=None,
                           replicates: int = 2000, seed: int = 20260912) -> dict:
    """Fit the recalibrator on the fit (validation) set only; report eval before/after."""
    if method == "temperature":
        t = fit_temperature(fit_labels, fit_probabilities)
        params, after = {"temperature": t}, apply_temperature(eval_probabilities, t)
    elif method == "platt":
        a, b = fit_platt(fit_labels, fit_probabilities)
        params, after = {"a": a, "b": b}, apply_platt(eval_probabilities, a, b)
    else:
        raise ValueError("method must be 'temperature' or 'platt'")
    return {
        "method": method, "params": params, "fit_n": len(np.asarray(fit_labels)),
        "before": _summary(eval_labels, eval_probabilities, eval_groups, replicates, seed),
        "after": _summary(eval_labels, after, eval_groups, replicates, seed),
        "calibrated_probabilities": after.tolist(),
    }
