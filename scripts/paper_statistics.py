"""Publication tables and paired, multiplicity-controlled statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 20260912
N_BOOTSTRAP = 2000
MODELS = (
    "resnet50",
    "dinov3_vitl16",
    "clip_vitl14_336",
    "siglip2_b16_384",
    "fashionclip2",
    "marqo_fashionsiglip",
    "gr_lite",
)
COMPARISONS = (
    ("dinov3_vitl16", "resnet50", "DINOv3 minus ResNet50"),
    ("gr_lite", "dinov3_vitl16", "GR-Lite minus DINOv3"),
    ("fashionclip2", "clip_vitl14_336", "FashionCLIP minus CLIP"),
    ("marqo_fashionsiglip", "siglip2_b16_384", "Marqo-FashionSigLIP minus SigLIP2"),
    (
        "marqo_fashionsiglip",
        "clip_vitl14_336",
        "historical-validation best fashion minus best generic",
    ),
    (
        "marqo_fashionsiglip",
        "resnet50",
        "historical-validation best overall minus ResNet50",
    ),
)
PROTOCOLS = {
    "historical_polyvore_d": "primary",
    "polyvore_d_clean": "clean_primary",
}


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def bootstrap_pvalue(differences: np.ndarray) -> float:
    lower = int(np.sum(differences <= 0))
    upper = int(np.sum(differences >= 0))
    return min(1.0, 2 * (min(lower, upper) + 1) / (len(differences) + 1))


def holm(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def aligned_frames(prefix: str):
    results = pd.read_csv(f"artifacts/{prefix}_results.csv").set_index("representation")
    cp = pd.read_parquet(f"artifacts/{prefix}_cp_predictions.parquet")
    fitb = pd.read_parquet(f"artifacts/{prefix}_fitb_predictions.parquet")
    cp_frames = {
        model: cp.loc[cp.model_key == model].sort_values("example_index") for model in MODELS
    }
    fitb_frames = {
        model: fitb.loc[fitb.model_key == model].sort_values("question_index") for model in MODELS
    }
    labels = cp_frames[MODELS[0]]["label"].to_numpy()
    questions = fitb_frames[MODELS[0]]["question_index"].to_numpy()
    for model in MODELS[1:]:
        if not np.array_equal(labels, cp_frames[model]["label"].to_numpy()):
            raise ValueError(f"Misaligned CP predictions in {prefix}")
        if not np.array_equal(questions, fitb_frames[model]["question_index"].to_numpy()):
            raise ValueError(f"Misaligned FITB predictions in {prefix}")
    return results, cp_frames, fitb_frames, labels, questions


def analyze_protocol(dataset: str, prefix: str):
    results, cp, fitb, labels, questions = aligned_frames(prefix)
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    cp_indices = np.concatenate(
        (
            positive[rng.integers(0, len(positive), (N_BOOTSTRAP, len(positive)))],
            negative[rng.integers(0, len(negative), (N_BOOTSTRAP, len(negative)))],
        ),
        axis=1,
    )
    fitb_indices = rng.integers(0, len(questions), (N_BOOTSTRAP, len(questions)))
    samples: dict[str, dict[str, np.ndarray]] = {}
    summary = []
    for model in MODELS:
        scores = cp[model]["score"].to_numpy()
        correct = fitb[model]["correct"].to_numpy(dtype=float)
        auc = np.fromiter(
            (roc_auc_score(labels[index], scores[index]) for index in cp_indices),
            dtype=float,
            count=N_BOOTSTRAP,
        )
        pr = np.fromiter(
            (average_precision_score(labels[index], scores[index]) for index in cp_indices),
            dtype=float,
            count=N_BOOTSTRAP,
        )
        fitb_sample = correct[fitb_indices].mean(axis=1)
        samples[model] = {"cp_auc": auc, "pr_auc": pr, "fitb_acc": fitb_sample}
        auc_ci, pr_ci, fitb_ci = interval(auc), interval(pr), interval(fitb_sample)
        summary.append(
            {
                "dataset": dataset,
                "representation": model,
                "cp_auc": results.loc[model, "cp_auc"],
                "cp_auc_ci_low": auc_ci[0],
                "cp_auc_ci_high": auc_ci[1],
                "pr_auc": results.loc[model, "pr_auc"],
                "pr_auc_ci_low": pr_ci[0],
                "pr_auc_ci_high": pr_ci[1],
                "fitb_acc": results.loc[model, "fitb_acc"],
                "fitb_ci_low": fitb_ci[0],
                "fitb_ci_high": fitb_ci[1],
                "selected_c": results.loc[model, "best_c"],
            }
        )
    archive = {
        f"{model}_{metric}": values
        for model, metrics in samples.items()
        for metric, values in metrics.items()
    }
    np.savez_compressed(f"artifacts/{prefix}_bootstrap_{N_BOOTSTRAP}.npz", **archive)

    pairwise = []
    for first, second, label in COMPARISONS:
        auc_difference = samples[first]["cp_auc"] - samples[second]["cp_auc"]
        fitb_difference = samples[first]["fitb_acc"] - samples[second]["fitb_acc"]
        auc_ci, fitb_ci = interval(auc_difference), interval(fitb_difference)
        first_correct = fitb[first]["correct"].to_numpy(dtype=bool)
        second_correct = fitb[second]["correct"].to_numpy(dtype=bool)
        first_only = int(np.sum(first_correct & ~second_correct))
        second_only = int(np.sum(~first_correct & second_correct))
        mcnemar = (
            binomtest(first_only, first_only + second_only, 0.5).pvalue
            if first_only + second_only
            else 1.0
        )
        pairwise.extend(
            (
                {
                    "dataset": dataset,
                    "comparison": label,
                    "metric": "CP ROC-AUC difference",
                    "effect": results.loc[first, "cp_auc"] - results.loc[second, "cp_auc"],
                    "ci_low": auc_ci[0],
                    "ci_high": auc_ci[1],
                    "raw_p": bootstrap_pvalue(auc_difference),
                    "test": "paired stratified bootstrap",
                    "discordant_first_only": np.nan,
                    "discordant_second_only": np.nan,
                },
                {
                    "dataset": dataset,
                    "comparison": label,
                    "metric": "FITB accuracy difference",
                    "effect": first_correct.mean() - second_correct.mean(),
                    "ci_low": fitb_ci[0],
                    "ci_high": fitb_ci[1],
                    "raw_p": mcnemar,
                    "test": "exact McNemar; paired-bootstrap CI",
                    "discordant_first_only": first_only,
                    "discordant_second_only": second_only,
                },
            )
        )
    pairwise_frame = pd.DataFrame(pairwise)
    for metric in pairwise_frame.metric.unique():
        mask = pairwise_frame.metric == metric
        pairwise_frame.loc[mask, "holm_p"] = holm(pairwise_frame.loc[mask, "raw_p"])
    return pd.DataFrame(summary), pairwise_frame


def rank_agreement(frame: pd.DataFrame, first: str, second: str, label: str) -> dict:
    rho = spearmanr(frame[first], frame[second])
    tau = kendalltau(frame[first], frame[second])
    return {
        "comparison": label,
        "n_models": len(frame),
        "spearman_rho": rho.statistic,
        "spearman_p": rho.pvalue,
        "kendall_tau": tau.statistic,
        "kendall_p": tau.pvalue,
    }


def main() -> None:
    summaries, pairwise = [], []
    for dataset, prefix in PROTOCOLS.items():
        protocol_summary, protocol_pairwise = analyze_protocol(dataset, prefix)
        summaries.append(protocol_summary)
        pairwise.append(protocol_pairwise)
    summary = pd.concat(summaries, ignore_index=True)
    summary["cp_rank"] = summary.groupby("dataset")["cp_auc"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["fitb_rank"] = summary.groupby("dataset")["fitb_acc"].rank(
        method="min", ascending=False
    ).astype(int)
    summary.to_csv("artifacts/paper_main_results.csv", index=False)
    pd.concat(pairwise, ignore_index=True).to_csv(
        "artifacts/paper_pairwise_statistics.csv", index=False
    )

    wide = summary.pivot(index="representation", columns="dataset")
    changes = pd.DataFrame(index=MODELS)
    for metric in ("cp_auc", "pr_auc", "fitb_acc"):
        changes[f"historical_{metric}"] = wide[metric]["historical_polyvore_d"]
        changes[f"clean_{metric}"] = wide[metric]["polyvore_d_clean"]
        changes[f"clean_minus_historical_{metric}"] = (
            changes[f"clean_{metric}"] - changes[f"historical_{metric}"]
        )
    regenerated = pd.read_csv("artifacts/historical_regenerated_results.csv").set_index(
        "representation"
    )
    for metric in ("cp_auc", "pr_auc", "fitb_acc"):
        changes[f"historical_regenerated_{metric}"] = regenerated[metric]
        changes[f"clean_minus_regenerated_{metric}"] = (
            changes[f"clean_{metric}"] - regenerated[metric]
        )
    changes.rename_axis("representation").reset_index().to_csv(
        "artifacts/protocol_performance_changes.csv", index=False
    )

    correlations = []
    for dataset in PROTOCOLS:
        frame = summary.loc[summary.dataset == dataset]
        correlations.append(rank_agreement(frame, "cp_auc", "fitb_acc", f"{dataset}: CP vs FITB"))
    rank_wide = summary.pivot(index="representation", columns="dataset")
    correlations.extend(
        (
            rank_agreement(
                rank_wide["cp_auc"],
                "historical_polyvore_d",
                "polyvore_d_clean",
                "historical vs clean CP",
            ),
            rank_agreement(
                rank_wide["fitb_acc"],
                "historical_polyvore_d",
                "polyvore_d_clean",
                "historical vs clean FITB",
            ),
        )
    )
    for dataset_prefix in ("historical", "clean"):
        tracks = {
            "pca128": pd.read_csv(f"artifacts/{dataset_prefix}_pca128_results.csv"),
            "pca256": pd.read_csv(
                "artifacts/primary_results.csv"
                if dataset_prefix == "historical"
                else "artifacts/clean_primary_results.csv"
            ),
            "pca512": pd.read_csv(f"artifacts/{dataset_prefix}_pca512_results.csv"),
            "native": pd.read_csv(
                "artifacts/native_results.csv"
                if dataset_prefix == "historical"
                else "artifacts/clean_native_results.csv"
            ),
        }
        merged = tracks["pca256"][["representation", "cp_auc", "fitb_acc"]].rename(
            columns={"cp_auc": "cp_pca256", "fitb_acc": "fitb_pca256"}
        )
        for track, frame in tracks.items():
            if track == "pca256":
                continue
            merged = merged.merge(
                frame[["representation", "cp_auc", "fitb_acc"]].rename(
                    columns={"cp_auc": f"cp_{track}", "fitb_acc": f"fitb_{track}"}
                ),
                on="representation",
            )
            correlations.append(
                rank_agreement(
                    merged,
                    "cp_pca256",
                    f"cp_{track}",
                    f"{dataset_prefix}: CP PCA256 vs {track}",
                )
            )
            correlations.append(
                rank_agreement(
                    merged,
                    "fitb_pca256",
                    f"fitb_{track}",
                    f"{dataset_prefix}: FITB PCA256 vs {track}",
                )
            )
        merged.to_csv(f"artifacts/{dataset_prefix}_pca_robustness_table.csv", index=False)
    pd.DataFrame(correlations).to_csv("artifacts/paper_rank_correlations.csv", index=False)

    audit = pd.read_csv("artifacts/model_audit.csv")
    possibly_exposed = set(
        audit.loc[
            audit.possible_polyvore_exposure.str.contains("possibly_exposed", na=False), "model_key"
        ]
    )
    clean_provenance = summary.loc[
        (summary.dataset == "polyvore_d_clean")
        & ~summary.representation.isin(possibly_exposed)
    ].copy()
    clean_provenance["cp_rank_clean_provenance"] = clean_provenance.cp_auc.rank(
        ascending=False, method="min"
    ).astype(int)
    clean_provenance["fitb_rank_clean_provenance"] = clean_provenance.fitb_acc.rank(
        ascending=False, method="min"
    ).astype(int)
    clean_provenance.to_csv("artifacts/clean_provenance_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
