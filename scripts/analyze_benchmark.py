from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 20260912
N_BOOTSTRAP = 2000

LOOKBENCH = [
    {
        "model_key": "resnet50",
        "exact_match": False,
        "lookbench_label": None,
        "lookbench_fine_recall_at_1": np.nan,
        "match_basis": "No ResNet50 result in LookBench Table 3.",
    },
    {
        "model_key": "dinov3_vitl16",
        "exact_match": True,
        "lookbench_label": "DINOv3-ViT-L",
        "lookbench_fine_recall_at_1": 43.97,
        "match_basis": "Exact ViT-L/16, 224 px, 1024-d checkpoint linked by the paper.",
    },
    {
        "model_key": "clip_vitl14_336",
        "exact_match": True,
        "lookbench_label": "CLIP-L/14",
        "lookbench_fine_recall_at_1": 39.79,
        "match_basis": "Exact CLIP ViT-L/14, 336 px, 768-d configuration.",
    },
    {
        "model_key": "siglip2_b16_384",
        "exact_match": True,
        "lookbench_label": "SigLIP2-B/16",
        "lookbench_fine_recall_at_1": 59.44,
        "match_basis": "Exact SigLIP2-B/16, 384 px, 768-d configuration.",
    },
    {
        "model_key": "fashionclip2",
        "exact_match": False,
        "lookbench_label": "Marqo-fashionCLIP",
        "lookbench_fine_recall_at_1": np.nan,
        "match_basis": "LookBench uses Marqo/marqo-fashionCLIP, not patrickjohncyh/fashion-clip.",
    },
    {
        "model_key": "marqo_fashionsiglip",
        "exact_match": True,
        "lookbench_label": "Marqo-fashionSigLIP",
        "lookbench_fine_recall_at_1": 62.77,
        "match_basis": "Exact Marqo-FashionSigLIP, 224 px, 768-d configuration.",
    },
    {
        "model_key": "gr_lite",
        "exact_match": True,
        "lookbench_label": "GR-Lite",
        "lookbench_fine_recall_at_1": 65.71,
        "match_basis": "Exact public srpone/gr-lite, 336 px, 1024-d release configuration.",
    },
]

COMPARISONS = [
    ("dinov3_vitl16", "resnet50", "self-supervised_vs_cnn"),
    ("clip_vitl14_336", "fashionclip2", "clip_generic_vs_fashion"),
    ("siglip2_b16_384", "marqo_fashionsiglip", "siglip_generic_vs_fashion"),
    ("marqo_fashionsiglip", "clip_vitl14_336", "best_fashion_vs_best_generic"),
    ("marqo_fashionsiglip", "resnet50", "best_overall_vs_cnn"),
    ("gr_lite", "dinov3_vitl16", "fashion_retrieval_vs_dinov3_parent"),
]


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    results = pd.read_csv("artifacts/primary_results.csv")
    cp = pd.read_parquet("artifacts/primary_cp_predictions.parquet")
    fitb = pd.read_parquet("artifacts/primary_fitb_predictions.parquet")
    models = results["representation"].tolist()
    cp_by_model = {
        model: cp.loc[cp.model_key == model].sort_values("example_index") for model in models
    }
    fitb_by_model = {
        model: fitb.loc[fitb.model_key == model].sort_values("question_index")
        for model in models
    }
    labels = cp_by_model[models[0]]["label"].to_numpy()
    if any(not np.array_equal(labels, frame["label"].to_numpy()) for frame in cp_by_model.values()):
        raise ValueError("CP labels are not aligned across models")
    question_ids = fitb_by_model[models[0]]["question_index"].to_numpy()
    if any(
        not np.array_equal(question_ids, frame["question_index"].to_numpy())
        for frame in fitb_by_model.values()
    ):
        raise ValueError("FITB questions are not aligned across models")

    rng = np.random.default_rng(SEED)
    cp_indices = rng.integers(0, len(labels), size=(N_BOOTSTRAP, len(labels)), dtype=np.int32)
    fitb_indices = rng.integers(
        0, len(question_ids), size=(N_BOOTSTRAP, len(question_ids)), dtype=np.int32
    )
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    summary_rows = []
    for model in models:
        scores = cp_by_model[model]["score"].to_numpy()
        correctness = fitb_by_model[model]["correct"].to_numpy(dtype=float)
        auc_values = np.fromiter(
            (roc_auc_score(labels[index], scores[index]) for index in cp_indices),
            dtype=float,
            count=N_BOOTSTRAP,
        )
        pr_values = np.fromiter(
            (average_precision_score(labels[index], scores[index]) for index in cp_indices),
            dtype=float,
            count=N_BOOTSTRAP,
        )
        fitb_values = correctness[fitb_indices].mean(axis=1)
        bootstrap[model] = {"cp_auc": auc_values, "pr_auc": pr_values, "fitb": fitb_values}
        auc_low, auc_high = interval(auc_values)
        pr_low, pr_high = interval(pr_values)
        fitb_low, fitb_high = interval(fitb_values)
        summary_rows.append(
            {
                "representation": model,
                "cp_auc_ci95_low": auc_low,
                "cp_auc_ci95_high": auc_high,
                "pr_auc_ci95_low": pr_low,
                "pr_auc_ci95_high": pr_high,
                "fitb_ci95_low": fitb_low,
                "fitb_ci95_high": fitb_high,
            }
        )

    pairwise = []
    for model_a, model_b, hypothesis in COMPARISONS:
        cp_difference = bootstrap[model_a]["cp_auc"] - bootstrap[model_b]["cp_auc"]
        fitb_difference = bootstrap[model_a]["fitb"] - bootstrap[model_b]["fitb"]
        cp_low, cp_high = interval(cp_difference)
        fitb_low, fitb_high = interval(fitb_difference)
        lower_count = int(np.sum(cp_difference <= 0))
        upper_count = int(np.sum(cp_difference >= 0))
        cp_p = min(1.0, 2 * (min(lower_count, upper_count) + 1) / (N_BOOTSTRAP + 1))
        a = fitb_by_model[model_a]["correct"].to_numpy(dtype=bool)
        b = fitb_by_model[model_b]["correct"].to_numpy(dtype=bool)
        discordant_a = int(np.sum(a & ~b))
        discordant_b = int(np.sum(~a & b))
        mcnemar_p = (
            binomtest(discordant_a, discordant_a + discordant_b, 0.5).pvalue
            if discordant_a + discordant_b
            else 1.0
        )
        pairwise.extend(
            [
                {
                    "hypothesis": hypothesis,
                    "comparison": f"{model_a}_minus_{model_b}",
                    "metric": "cp_auc_difference",
                    "estimate": float(
                        results.set_index("representation").loc[model_a, "cp_auc"]
                        - results.set_index("representation").loc[model_b, "cp_auc"]
                    ),
                    "ci95_low": cp_low,
                    "ci95_high": cp_high,
                    "raw_p": cp_p,
                    "test": "paired_bootstrap_tail",
                },
                {
                    "hypothesis": hypothesis,
                    "comparison": f"{model_a}_minus_{model_b}",
                    "metric": "fitb_accuracy_difference",
                    "estimate": float(a.mean() - b.mean()),
                    "ci95_low": fitb_low,
                    "ci95_high": fitb_high,
                    "raw_p": mcnemar_p,
                    "test": "exact_mcnemar",
                },
            ]
        )
    pairwise_frame = pd.DataFrame(pairwise)
    for metric in pairwise_frame.metric.unique():
        mask = pairwise_frame.metric == metric
        pairwise_frame.loc[mask, "holm_p"] = holm_adjust(
            pairwise_frame.loc[mask, "raw_p"].tolist()
        )
    pairwise_frame.to_csv("artifacts/primary_pairwise_statistics.csv", index=False)

    summary = results.merge(pd.DataFrame(summary_rows), on="representation")
    summary["cp_rank"] = summary["cp_auc"].rank(method="min", ascending=False).astype(int)
    summary["fitb_rank"] = summary["fitb_acc"].rank(method="min", ascending=False).astype(int)
    summary.to_csv("artifacts/primary_summary.csv", index=False)

    overlap = pd.DataFrame(LOOKBENCH).merge(
        summary[["representation", "cp_auc", "fitb_acc", "cp_rank", "fitb_rank"]],
        left_on="model_key",
        right_on="representation",
    )
    exact = overlap.loc[overlap.exact_match].copy()
    exact["lookbench_rank"] = exact["lookbench_fine_recall_at_1"].rank(
        method="min", ascending=False
    ).astype(int)
    exact["cp_subset_rank"] = exact["cp_auc"].rank(method="min", ascending=False).astype(int)
    exact["fitb_subset_rank"] = exact["fitb_acc"].rank(method="min", ascending=False).astype(int)
    overlap = overlap.merge(
        exact[["model_key", "lookbench_rank", "cp_subset_rank", "fitb_subset_rank"]],
        on="model_key",
        how="left",
    )
    overlap.to_csv("artifacts/lookbench_overlap.csv", index=False)

    correlations = []
    for metric, rank_column in (("cp_auc", "cp_subset_rank"), ("fitb_acc", "fitb_subset_rank")):
        rho = spearmanr(exact["lookbench_rank"], exact[rank_column])
        tau = kendalltau(exact["lookbench_rank"], exact[rank_column])
        correlations.append(
            {
                "compatibility_metric": metric,
                "n_models": len(exact),
                "spearman_rho": rho.statistic,
                "spearman_p": rho.pvalue,
                "kendall_tau": tau.statistic,
                "kendall_p": tau.pvalue,
            }
        )
    pd.DataFrame(correlations).to_csv("artifacts/lookbench_correlations.csv", index=False)

    reversals = []
    exact = exact.set_index("model_key")
    for model_a, model_b in combinations(exact.index, 2):
        retrieval_direction = np.sign(
            exact.loc[model_a, "lookbench_fine_recall_at_1"]
            - exact.loc[model_b, "lookbench_fine_recall_at_1"]
        )
        for metric in ("cp_auc", "fitb_acc"):
            compatibility_direction = np.sign(
                exact.loc[model_a, metric] - exact.loc[model_b, metric]
            )
            if retrieval_direction * compatibility_direction < 0:
                reversals.append(
                    {"model_a": model_a, "model_b": model_b, "metric": metric}
                )
    pd.DataFrame(reversals).to_csv("artifacts/lookbench_rank_reversals.csv", index=False)


if __name__ == "__main__":
    main()
