"""Uniform paired inference for aligned CP scores and FITB/A100 correctness.

Bootstrap units are supplied explicitly by the protocol. Two-way resampling is
the independent-multiplicity (pigeonhole) bootstrap, not an assertion that its
intervals must be wider. DeLong and exact McNemar ignore clustering.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import binomtest, norm, rankdata
from sklearn.metrics import roc_auc_score


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    corrected = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(corrected, 1.0)
    return result


def delong_covariance(labels, scores):
    y = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("DeLong requires aligned binary labels and a score matrix")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    pos, neg = scores[y == 1].T, scores[y == 0].T
    m, n = pos.shape[1], neg.shape[1]
    if min(m, n) < 2:
        raise ValueError("DeLong covariance needs at least two examples per class")
    tx, ty = rankdata(pos, axis=1), rankdata(neg, axis=1)
    tz = rankdata(np.concatenate([pos, neg], axis=1), axis=1)
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    covariance = np.atleast_2d(np.cov(v01, ddof=1)) / m + np.atleast_2d(np.cov(v10, ddof=1)) / n
    return auc, covariance


def delong_pvalue(auc, covariance, left, right):
    difference = float(auc[left] - auc[right])
    variance = max(
        0.0, float(covariance[left, left] + covariance[right, right] - 2 * covariance[left, right])
    )
    if variance == 0:
        return 1.0 if difference == 0 else float(np.finfo(float).tiny)
    return float(max(np.finfo(float).tiny, 2 * norm.sf(abs(difference) / np.sqrt(variance))))


def exact_mcnemar(left, right):
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Correctness vectors must align")
    wins = int(np.sum(left & ~right))
    losses = int(np.sum(~left & right))
    return {
        "left_only_correct": wins,
        "right_only_correct": losses,
        "pvalue": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
    }


def _cluster_codes(groups, length):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != length:
        raise ValueError("One cluster ID is required per aligned example")
    if any(value is None or str(value) in {"nan", ""} for value in groups):
        raise ValueError("Missing cluster IDs")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two independent clusters are required")
    return inverse, len(unique)


def _auc_layout(labels, scores):
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1]
    return order, starts, labels[order]


def _weighted_auc(layout, weights):
    order, starts, labels = layout
    ordered = weights[order]
    positive = np.add.reduceat(ordered * labels, starts)
    negative = np.add.reduceat(ordered * (1 - labels), starts)
    return float(
        positive @ (np.cumsum(negative) - 0.5 * negative) / (positive.sum() * negative.sum())
    )


def cluster_adjusted_mcnemar(left, right, groups, *, replicates=10000, seed=20260912):
    """Cluster sign-flip test of paired discordance; label separately from exact McNemar."""
    left, right = np.asarray(left, int), np.asarray(right, int)
    if left.shape != right.shape or left.ndim != 1 or not np.isin([left, right], [0, 1]).all():
        raise ValueError("Aligned binary correctness is required")
    codes, count = _cluster_codes(groups, len(left))
    differences = np.bincount(codes, weights=left - right, minlength=count)
    observed = abs(differences.sum())
    if not observed:
        return {"pvalue": 1.0, "method": "cluster sign-flip discordance", "clusters": count}
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(replicates):
        randomized = abs(np.dot(rng.choice([-1.0, 1.0], size=count), differences))
        exceed += randomized >= observed
    return {
        "pvalue": float((exceed + 1) / (replicates + 1)),
        "method": "cluster sign-flip discordance",
        "clusters": count,
        "replicates": replicates,
        "p_resolution_floor": 1 / (replicates + 1),
    }


def cluster_bootstrap(
    values, *, metric, groups, labels=None, second_groups=None, replicates=2000, seed=20260912
):
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 2 or not scores.shape[0] or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite examples-by-models matrix")
    if metric not in {"cp_auc", "accuracy"} or replicates < 2:
        raise ValueError("Unsupported metric or insufficient bootstrap replicates")
    codes, n_groups = _cluster_codes(groups, len(scores))
    secondary = _cluster_codes(second_groups, len(scores)) if second_groups is not None else None
    y = None if labels is None else np.asarray(labels)
    if metric == "cp_auc" and (
        y is None or y.shape != (len(scores),) or set(np.unique(y)) != {0, 1}
    ):
        raise ValueError("CP needs both binary classes")
    if metric == "accuracy" and not np.isin(scores, [0, 1]).all():
        raise ValueError("Accuracy input must be per-example correctness, not probabilities")
    rng = np.random.default_rng(seed)
    layouts = (
        [_auc_layout(y, scores[:, column]) for column in range(scores.shape[1])]
        if metric == "cp_auc"
        else []
    )
    samples = []
    rejected = 0
    while len(samples) < replicates:
        counts = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        weights = counts[codes].astype(float)
        if secondary is not None:
            other_codes, other_count = secondary
            counts = np.bincount(rng.integers(other_count, size=other_count), minlength=other_count)
            weights *= counts[other_codes]
        if not weights.sum() or (
            metric == "cp_auc" and (not weights[y == 0].sum() or not weights[y == 1].sum())
        ):
            rejected += 1
            if rejected > 10 * replicates:
                raise ValueError("Too many degenerate cluster replicates")
            continue
        if metric == "accuracy":
            samples.append(np.average(scores, axis=0, weights=weights))
        else:
            samples.append([_weighted_auc(layout, weights) for layout in layouts])
    point = (
        scores.mean(0)
        if metric == "accuracy"
        else np.array([roc_auc_score(y, scores[:, column]) for column in range(scores.shape[1])])
    )
    return {
        "point": point,
        "samples": np.asarray(samples),
        "cluster_count": n_groups,
        "second_cluster_count": None if secondary is None else secondary[1],
        "rejected_degenerate_draws": rejected,
        "method": "two-way pigeonhole cluster bootstrap"
        if secondary
        else "paired cluster bootstrap",
        "replicates": replicates,
        "seed": seed,
    }


def pairwise_inference(bootstrap, model_keys, *, metric, margins):
    point, draws = np.asarray(bootstrap["point"]), np.asarray(bootstrap["samples"])
    if len(model_keys) != len(point) or len(set(model_keys)) != len(model_keys):
        raise ValueError("Unique model names must match bootstrap columns")
    if not margins or any(not np.isfinite(m) or m <= 0 for m in margins):
        raise ValueError("Pre-specified positive practical margins are required")
    count = len(draws)
    rows, equivalence = [], []
    for left, right in combinations(range(len(point)), 2):
        observed = float(point[left] - point[right])
        differences = draws[:, left] - draws[:, right]
        errors = differences - observed
        ci = np.quantile(differences, [0.025, 0.975])
        row = {
            "left": model_keys[left],
            "right": model_keys[right],
            "delta": observed,
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "bootstrap_p": float((1 + np.sum(np.abs(errors) >= abs(observed))) / (count + 1)),
            "p_resolution_floor": 1 / (count + 1),
            "replicates": count,
        }
        if metric == "accuracy":
            row["cohen_h"] = float(
                2 * (np.arcsin(np.sqrt(point[left])) - np.arcsin(np.sqrt(point[right])))
            )
        rows.append(row)
        for margin in margins:
            lower_p = (1 + np.sum(errors >= observed + margin)) / (count + 1)
            upper_p = (1 + np.sum(errors <= observed - margin)) / (count + 1)
            equivalence.append(
                {
                    "left": model_keys[left],
                    "right": model_keys[right],
                    "margin": margin,
                    "tost_p": float(max(lower_p, upper_p)),
                }
            )
    for row, p in zip(rows, holm([r["bootstrap_p"] for r in rows]), strict=True):
        row["bootstrap_p_holm"] = float(p)
    # Holm within each pre-specified margin, dataset and metric.
    for margin in margins:
        selected = [row for row in equivalence if row["margin"] == margin]
        adjusted = holm([r["tost_p"] for r in selected])
        for row, p in zip(selected, adjusted, strict=True):
            row["tost_p_holm"] = float(p)
            row["equivalent_at_005"] = bool(p < 0.05)
    ranks = rankdata(-draws, axis=1, method="average")
    rank_intervals = [
        {
            "model": key,
            "rank_median": float(np.median(ranks[:, index])),
            "rank_low": float(np.quantile(ranks[:, index], 0.025)),
            "rank_high": float(np.quantile(ranks[:, index], 0.975)),
        }
        for index, key in enumerate(model_keys)
    ]
    estimates = [
        {
            "model": key,
            "estimate": float(point[index]),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, key in enumerate(model_keys)
    ]
    return {
        "comparisons": rows,
        "equivalence": equivalence,
        "rank_intervals": rank_intervals,
        "model_estimates": estimates,
    }
