"""Group-aware final statistics for IQON3000-Clean and external A100."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from repbench.encoders.registry import PRIMARY_MODEL_KEYS

SEED = 20260912
N_BOOTSTRAP = 2000
OUTPUT = Path("artifacts/final/statistics")
IQON = Path("artifacts/final/iqon3000_clean")

FIXED_COMPARISONS = (
    ("dinov3_vitl16", "resnet50", "DINOv3 minus ResNet50"),
    ("gr_lite", "dinov3_vitl16", "GR-Lite minus DINOv3"),
    ("fashionclip2", "clip_vitl14_336", "FashionCLIP minus CLIP"),
    ("marqo_fashionsiglip", "siglip2_b16_384", "Marqo-FashionSigLIP minus SigLIP2"),
)
SPECIALIZED = {"fashionclip2", "marqo_fashionsiglip", "gr_lite"}


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def bootstrap_pvalue(values: np.ndarray) -> float:
    tail = min(np.sum(values <= 0), np.sum(values >= 0))
    return min(1.0, 2.0 * (float(tail) + 1.0) / (len(values) + 1.0))


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def cluster_counts(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    unique, encoded = np.unique(groups.astype(str), return_inverse=True)
    draws = rng.integers(0, len(unique), size=(N_BOOTSTRAP, len(unique)), dtype=np.int32)
    counts = np.zeros((N_BOOTSTRAP, len(unique)), dtype=np.int16)
    for row, values in enumerate(draws):
        counts[row] = np.bincount(values, minlength=len(unique))
    return counts, encoded


def weighted_binary_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_sample_counts: np.ndarray,
    group_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact weighted AUC/AP, including exact-score ties."""
    ascending = np.argsort(scores, kind="mergesort")
    descending = ascending[::-1]
    sorted_ascending_scores = scores[ascending]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_ascending_scores) != 0) + 1]
    tie_stops = np.r_[tie_starts[1:], len(scores)]
    duplicate_ties = [(start, stop) for start, stop in zip(tie_starts, tie_stops, strict=True) if stop - start > 1]
    auc = np.empty(N_BOOTSTRAP, dtype=float)
    ap = np.empty(N_BOOTSTRAP, dtype=float)
    labels_float = labels.astype(float)
    for offset in range(0, N_BOOTSTRAP, 40):
        batch_stop = min(offset + 40, N_BOOTSTRAP)
        weights = cluster_sample_counts[offset:batch_stop, group_index].astype(float)
        asc_weights = weights[:, ascending]
        asc_labels = labels_float[ascending]
        positive = asc_weights * asc_labels
        negative = asc_weights * (1.0 - asc_labels)
        negative_before = np.cumsum(negative, axis=1) - negative
        numerator = np.sum(positive * negative_before, axis=1)
        for group_start, group_stop in duplicate_ties:
            group_positive = positive[:, group_start:group_stop]
            group_negative = negative[:, group_start:group_stop]
            internal_ordered = np.sum(
                group_positive * (np.cumsum(group_negative, axis=1) - group_negative), axis=1
            )
            numerator += 0.5 * group_positive.sum(axis=1) * group_negative.sum(axis=1) - internal_ordered
        auc[offset:batch_stop] = numerator / (positive.sum(axis=1) * negative.sum(axis=1))
        desc_weights = weights[:, descending]
        desc_positive = desc_weights * labels_float[descending]
        cumulative_positive = np.cumsum(desc_positive, axis=1)
        cumulative_total = np.cumsum(desc_weights, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        ap_numerator = np.sum(desc_positive * precision, axis=1)
        # AP treats every tied score as one threshold. Correct the arbitrary stable
        # ordering contribution to use precision after the full tied block.
        desc_scores = scores[descending]
        desc_starts = np.r_[0, np.flatnonzero(np.diff(desc_scores) != 0) + 1]
        desc_stops = np.r_[desc_starts[1:], len(scores)]
        for group_start, group_stop in zip(desc_starts, desc_stops, strict=True):
            if group_stop - group_start <= 1:
                continue
            group_positive = desc_positive[:, group_start:group_stop]
            old = np.sum(group_positive * precision[:, group_start:group_stop], axis=1)
            desired = group_positive.sum(axis=1) * precision[:, group_stop - 1]
            ap_numerator += desired - old
        ap[offset:batch_stop] = ap_numerator / desc_positive.sum(axis=1)
    return auc, ap


def iqon_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    cp_frames, fitb_frames, metrics = {}, {}, {}
    for model in PRIMARY_MODEL_KEYS:
        cp_frames[model] = pd.read_parquet(IQON / model / "cp_predictions_test.parquet").sort_values("example_id")
        fitb_frames[model] = pd.read_parquet(IQON / model / "fitb_predictions_test.parquet").sort_values("question_id")
        metrics[model] = json.loads((IQON / model / "metrics.json").read_text(encoding="utf-8"))
    reference_cp = cp_frames[PRIMARY_MODEL_KEYS[0]]
    reference_fitb = fitb_frames[PRIMARY_MODEL_KEYS[0]]
    for model in PRIMARY_MODEL_KEYS[1:]:
        if not np.array_equal(reference_cp.example_id, cp_frames[model].example_id):
            raise ValueError(f"IQON CP IDs misaligned: {model}")
        if not np.array_equal(reference_fitb.question_id, fitb_frames[model].question_id):
            raise ValueError(f"IQON FITB IDs misaligned: {model}")
    cp_counts, cp_groups = cluster_counts(reference_cp.group_user_id.to_numpy(), rng)
    fitb_counts, fitb_groups = cluster_counts(reference_fitb.group_user_id.to_numpy(), rng)
    samples, summary = {}, []
    labels = reference_cp.label.to_numpy(dtype=np.int8)
    for model in PRIMARY_MODEL_KEYS:
        scores = cp_frames[model].probability.to_numpy(float)
        auc, ap = weighted_binary_bootstrap(labels, scores, cp_counts, cp_groups)
        fitb_correct = fitb_frames[model].correct.to_numpy(float)
        fitb_sample = np.empty(N_BOOTSTRAP)
        for offset in range(0, N_BOOTSTRAP, 100):
            stop = min(offset + 100, N_BOOTSTRAP)
            weights = fitb_counts[offset:stop, fitb_groups].astype(float)
            fitb_sample[offset:stop] = (weights * fitb_correct).sum(axis=1) / weights.sum(axis=1)
        samples[model] = {"cp_auc": auc, "pr_auc": ap, "fitb_accuracy": fitb_sample}
        test = metrics[model]["splits"]["test"]
        auc_ci, ap_ci, fitb_ci = interval(auc), interval(ap), interval(fitb_sample)
        summary.append(
            {
                "dataset": "iqon3000_clean",
                "representation": model,
                "cp_auc": test["cp_roc_auc"],
                "cp_auc_ci_low": auc_ci[0],
                "cp_auc_ci_high": auc_ci[1],
                "pr_auc": test["cp_pr_auc"],
                "pr_auc_ci_low": ap_ci[0],
                "pr_auc_ci_high": ap_ci[1],
                "fitb_acc": test["fitb_accuracy"],
                "fitb_ci_low": fitb_ci[0],
                "fitb_ci_high": fitb_ci[1],
                "selected_c": metrics[model]["selected_c"],
                "validation_auc": metrics[model]["validation_auc"],
            }
        )
    summary_frame = pd.DataFrame(summary)
    generic = summary_frame.loc[~summary_frame.representation.isin(SPECIALIZED)].sort_values("validation_auc").iloc[-1].representation
    specialized = summary_frame.loc[summary_frame.representation.isin(SPECIALIZED)].sort_values("validation_auc").iloc[-1].representation
    overall = summary_frame.sort_values("validation_auc").iloc[-1].representation
    comparisons = FIXED_COMPARISONS + (
        (specialized, generic, "validation-selected best specialized minus best generic"),
        (overall, "resnet50", "validation-selected best overall minus ResNet50"),
    )
    pairwise = []
    for first, second, label in comparisons:
        for metric in ("cp_auc", "fitb_accuracy"):
            difference = samples[first][metric] - samples[second][metric]
            low, high = interval(difference)
            point_column = "cp_auc" if metric == "cp_auc" else "fitb_acc"
            point = summary_frame.set_index("representation").loc[first, point_column] - summary_frame.set_index("representation").loc[second, point_column]
            pairwise.append(
                {
                    "dataset": "iqon3000_clean",
                    "comparison": label,
                    "first": first,
                    "second": second,
                    "metric": metric,
                    "effect": point,
                    "ci_low": low,
                    "ci_high": high,
                    "raw_p": bootstrap_pvalue(difference),
                    "resampling_unit": "IQON user cluster",
                }
            )
    pairwise_frame = pd.DataFrame(pairwise)
    for metric in pairwise_frame.metric.unique():
        mask = pairwise_frame.metric == metric
        pairwise_frame.loc[mask, "holm_p"] = holm(pairwise_frame.loc[mask, "raw_p"].to_numpy())
    np.savez_compressed(
        OUTPUT / "iqon_group_bootstrap_2000.npz",
        **{f"{model}_{metric}": values for model, rows in samples.items() for metric, values in rows.items()},
    )
    return summary_frame, pairwise_frame


def a100_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet("artifacts/final/a100/predictions.parquet")
    rng = np.random.default_rng(SEED + 1)
    paired_indices = {
        task: rng.integers(0, 100, size=(N_BOOTSTRAP, 100)) for task in ("LAT", "AAT")
    }
    summary, samples = [], {}
    for (source, model, task), rows in frame.groupby(["training_source", "model_key", "task"], sort=False):
        rows = rows.sort_values("question_num")
        if len(rows) != 100 or rows.question_num.duplicated().any():
            raise ValueError(f"Invalid A100 prediction alignment: {source}/{model}/{task}")
        indices = paired_indices[task]
        if task == "LAT":
            correct = rows.majority_correct.to_numpy(float)
            values = correct[indices].mean(axis=1)
            mlat = rows.human_agreement.to_numpy(float)[indices].mean(axis=1)
            archive = rows.archive_gt_correct.to_numpy(float)[indices].mean(axis=1)
            samples[(source, model, task, "accuracy")] = values
            low, high = interval(values)
            mlow, mhigh = interval(mlat)
            alow, ahigh = interval(archive)
            summary.append(
                {"training_source": source, "representation": model, "task": task,
                 "accuracy": correct.mean(), "ci_low": low, "ci_high": high,
                 "mLAT": rows.human_agreement.mean(), "mLAT_ci_low": mlow, "mLAT_ci_high": mhigh,
                 "archive_gt_accuracy": rows.archive_gt_correct.mean(),
                 "archive_gt_ci_low": alow, "archive_gt_ci_high": ahigh}
            )
        else:
            correct = rows.archive_gt_correct.to_numpy(float)
            values = correct[indices].mean(axis=1)
            samples[(source, model, task, "accuracy")] = values
            low, high = interval(values)
            summary.append(
                {"training_source": source, "representation": model, "task": task,
                 "accuracy": correct.mean(), "ci_low": low, "ci_high": high}
            )
            for dimension, subset in rows.groupby("dimension"):
                dimension_values = subset.archive_gt_correct.to_numpy(float)
                dimension_indices = rng.integers(0, len(subset), size=(N_BOOTSTRAP, len(subset)))
                sampled = dimension_values[dimension_indices].mean(axis=1)
                dlow, dhigh = interval(sampled)
                summary.append(
                    {"training_source": source, "representation": model,
                     "task": f"AAT-{dimension}", "accuracy": dimension_values.mean(),
                     "ci_low": dlow, "ci_high": dhigh, "n": len(subset)}
                )
    summary_frame = pd.DataFrame(summary)
    pairwise = []
    for source in frame.training_source.unique():
        validation = {
            model: json.loads((Path("artifacts/final") / source / model / "metrics.json").read_text())["validation_auc"]
            if source == "iqon3000_clean"
            else float(pd.read_csv("artifacts/clean_primary_results.csv").set_index("representation").loc[model, "validation_auc"])
            for model in PRIMARY_MODEL_KEYS
        }
        generic = max((model for model in PRIMARY_MODEL_KEYS if model not in SPECIALIZED), key=validation.get)
        specialized = max(SPECIALIZED, key=validation.get)
        overall = max(PRIMARY_MODEL_KEYS, key=validation.get)
        comparisons = FIXED_COMPARISONS + (
            (specialized, generic, "validation-selected best specialized minus best generic"),
            (overall, "resnet50", "validation-selected best overall minus ResNet50"),
        )
        for task in ("LAT", "AAT"):
            for first, second, label in comparisons:
                difference = samples[(source, first, task, "accuracy")] - samples[(source, second, task, "accuracy")]
                low, high = interval(difference)
                point = summary_frame.set_index(["training_source", "representation", "task"])
                effect = point.loc[(source, first, task), "accuracy"] - point.loc[(source, second, task), "accuracy"]
                pairwise.append(
                    {"training_source": source, "task": task, "comparison": label,
                     "first": first, "second": second, "effect": effect,
                     "ci_low": low, "ci_high": high, "raw_p": bootstrap_pvalue(difference)}
                )
    pairwise_frame = pd.DataFrame(pairwise)
    for (_, task), indices in pairwise_frame.groupby(["training_source", "task"]).groups.items():
        pairwise_frame.loc[indices, "holm_p"] = holm(pairwise_frame.loc[indices, "raw_p"].to_numpy())
    return summary_frame, pairwise_frame


def rank_correlations(iqon: pd.DataFrame, a100: pd.DataFrame) -> pd.DataFrame:
    polyvore = pd.read_csv("artifacts/paper_main_results_with_cluster_ci.csv")
    polyvore = polyvore[polyvore.dataset == "polyvore_d_clean"].set_index("representation")
    iqon = iqon.set_index("representation")
    rows = []

    def add(label, first, second):
        rho = spearmanr(first, second)
        tau = kendalltau(first, second)
        rows.append({"comparison": label, "n_models": len(first), "spearman_rho": rho.statistic,
                     "spearman_p": rho.pvalue, "kendall_tau": tau.statistic, "kendall_p": tau.pvalue})

    add("Polyvore CP vs IQON CP", polyvore.loc[list(PRIMARY_MODEL_KEYS), "cp_auc"], iqon.loc[list(PRIMARY_MODEL_KEYS), "cp_auc"])
    add("Polyvore FITB vs IQON FITB", polyvore.loc[list(PRIMARY_MODEL_KEYS), "fitb_acc"], iqon.loc[list(PRIMARY_MODEL_KEYS), "fitb_acc"])
    add("IQON CP vs FITB", iqon.loc[list(PRIMARY_MODEL_KEYS), "cp_auc"], iqon.loc[list(PRIMARY_MODEL_KEYS), "fitb_acc"])
    for source in ("polyvore_d_clean", "iqon3000_clean"):
        for task in ("LAT", "AAT"):
            external = a100[(a100.training_source == source) & (a100.task == task)].set_index("representation")
            source_frame = polyvore if source == "polyvore_d_clean" else iqon
            add(f"{source} CP vs A100 {task}", source_frame.loc[list(PRIMARY_MODEL_KEYS), "cp_auc"], external.loc[list(PRIMARY_MODEL_KEYS), "accuracy"])
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    iqon, iqon_pairwise = iqon_statistics()
    a100, a100_pairwise = a100_statistics()
    iqon.to_csv(OUTPUT / "iqon_primary_with_group_ci.csv", index=False)
    iqon_pairwise.to_csv(OUTPUT / "iqon_predefined_pairwise.csv", index=False)
    a100.to_csv(OUTPUT / "a100_with_ci.csv", index=False)
    a100_pairwise.to_csv(OUTPUT / "a100_predefined_pairwise.csv", index=False)
    rank_correlations(iqon, a100).to_csv(OUTPUT / "cross_dataset_rank_correlations.csv", index=False)


if __name__ == "__main__":
    main()
