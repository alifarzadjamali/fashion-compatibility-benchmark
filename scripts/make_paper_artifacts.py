"""Generate reviewer-facing tables and camera-ready vector/raster figures."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

DISPLAY = {
    "resnet50": "ResNet50",
    "dinov3_vitl16": "DINOv3-L/16",
    "clip_vitl14_336": "CLIP-L/14@336",
    "siglip2_b16_384": "SigLIP2-B/16@384",
    "fashionclip2": "FashionCLIP 2.0",
    "marqo_fashionsiglip": "Marqo-FashionSigLIP",
    "gr_lite": "GR-Lite",
}
COLORS = dict(zip(DISPLAY, plt.get_cmap("tab10").colors, strict=False))


def save_figure(figure, name: str, figure_root: Path) -> None:
    figure.tight_layout()
    figure.savefig(figure_root / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_root / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def pareto_flags(frame: pd.DataFrame, metric: str, cost: str) -> pd.Series:
    flags = []
    for index, row in frame.iterrows():
        other = frame.drop(index)
        dominated = (
            (other[metric] >= row[metric])
            & (other[cost] <= row[cost])
            & ((other[metric] > row[metric]) | (other[cost] < row[cost]))
        ).any()
        flags.append(bool(dominated))
    return pd.Series(flags, index=frame.index)


def main() -> None:
    table_root = Path("reports/tables")
    figure_root = Path("reports/figures")
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    main_results = pd.read_csv("artifacts/paper_main_results.csv")
    main_results["model"] = main_results.representation.map(DISPLAY)
    main_table = main_results[
        [
            "dataset",
            "model",
            "cp_auc",
            "cp_auc_ci_low",
            "cp_auc_ci_high",
            "pr_auc",
            "pr_auc_ci_low",
            "pr_auc_ci_high",
            "fitb_acc",
            "fitb_ci_low",
            "fitb_ci_high",
            "selected_c",
            "cp_rank",
            "fitb_rank",
        ]
    ]
    main_table.to_csv(table_root / "main_results.csv", index=False)
    (table_root / "main_results.md").write_text(
        main_table.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8"
    )

    audit = pd.read_csv("artifacts/model_audit.csv")
    audit["model"] = audit.model_key.map(DISPLAY)
    family = audit[
        [
            "model",
            "family",
            "parameter_count",
            "native_embedding_dimension",
            "official_input_resolution",
        ]
    ]
    family.to_csv(table_root / "representation_families.csv", index=False)

    provenance = audit[
        [
            "model",
            "model_checkpoint",
            "exact_revision",
            "training_data_provenance",
            "known_polyvore_training_exposure",
            "possible_polyvore_exposure",
            "known_polyvore_evaluation_exposure",
            "license_usage_terms",
            "uncertainty",
        ]
    ]
    provenance.to_csv(table_root / "provenance_exposure.csv", index=False)

    historical_integrity = json.loads(
        Path("artifacts/data_integrity_report.json").read_text(encoding="utf-8")
    )
    clean_integrity = json.loads(
        Path("artifacts/polyvore_d_clean_validation.json").read_text(encoding="utf-8")
    )
    protocol_rows = []
    for protocol, report, task_source in (
        ("historical_polyvore_d", historical_integrity, "packaged official questions"),
        ("polyvore_d_clean", clean_integrity, "deterministic regenerated questions"),
    ):
        for split, values in report["splits"].items():
            protocol_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "outfits": values["outfits"],
                    "unique_items": values["unique_items"],
                    "cp_examples": values["cp_examples"],
                    "fitb_questions": values["fitb_questions"],
                    "missing_images": values["missing_images"],
                    "task_source": task_source,
                    "passes_item_disjointness": report["passes_item_disjointness"],
                }
            )
    protocols = pd.DataFrame(protocol_rows)
    protocols.to_csv(table_root / "dataset_protocol_comparison.csv", index=False)
    metadata = json.loads(
        Path("data/raw/polyvore_outfits/polyvore_item_metadata.json").read_text(encoding="utf-8")
    )
    distribution_rows = []
    category_rows = []
    for protocol, protocol_path in (
        ("historical_polyvore_d", Path("data/raw/polyvore_outfits/disjoint")),
        ("polyvore_d_clean", Path("data/protocols/polyvore_d_clean")),
    ):
        for split in ("train", "valid", "test"):
            outfits = json.loads((protocol_path / f"{split}.json").read_text(encoding="utf-8"))
            lengths = pd.Series([len(row["items"]) for row in outfits])
            distribution_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "mean_outfit_length": lengths.mean(),
                    "sd_outfit_length": lengths.std(),
                    "median_outfit_length": lengths.median(),
                    "min_outfit_length": lengths.min(),
                    "max_outfit_length": lengths.max(),
                }
            )
            categories = pd.Series(
                [
                    metadata[str(item["item_id"])]["semantic_category"].strip()
                    for row in outfits
                    for item in row["items"]
                ]
            ).value_counts(normalize=True)
            category_rows.extend(
                {
                    "protocol": protocol,
                    "split": split,
                    "semantic_category": category,
                    "proportion": proportion,
                }
                for category, proportion in categories.items()
            )
    pd.DataFrame(distribution_rows).to_csv(
        table_root / "dataset_outfit_length_statistics.csv", index=False
    )
    pd.DataFrame(category_rows).to_csv(
        table_root / "dataset_semantic_category_distributions.csv", index=False
    )

    lookbench = pd.read_csv("artifacts/lookbench_overlap.csv")
    exact = lookbench.loc[lookbench.exact_match].copy()
    retrieval_correlations = []
    retrieval_rows = []
    reversal_rows = []
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for axis, (dataset, title) in zip(
        axes,
        (("historical_polyvore_d", "Historical Polyvore-D"), ("polyvore_d_clean", "Polyvore-D-Clean")),
        strict=True,
    ):
        metrics = main_results.loc[main_results.dataset == dataset]
        merged = exact[["model_key", "lookbench_fine_recall_at_1"]].merge(
            metrics[["representation", "cp_auc", "fitb_acc"]],
            left_on="model_key",
            right_on="representation",
        )
        merged["retrieval_rank"] = merged.lookbench_fine_recall_at_1.rank(ascending=False)
        for metric in ("cp_auc", "fitb_acc"):
            merged[f"{metric}_rank"] = merged[metric].rank(ascending=False)
            rho = spearmanr(merged.retrieval_rank, merged[f"{metric}_rank"])
            tau = kendalltau(merged.retrieval_rank, merged[f"{metric}_rank"])
            retrieval_correlations.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_models": len(merged),
                    "spearman_rho": rho.statistic,
                    "spearman_p": rho.pvalue,
                    "kendall_tau": tau.statistic,
                    "kendall_p": tau.pvalue,
                }
            )
        retrieval_rows.append(merged.assign(dataset=dataset))
        indexed = merged.set_index("model_key")
        for first, second in combinations(indexed.index, 2):
            retrieval_direction = np.sign(
                indexed.loc[first, "lookbench_fine_recall_at_1"]
                - indexed.loc[second, "lookbench_fine_recall_at_1"]
            )
            for metric in ("cp_auc", "fitb_acc"):
                compatibility_direction = np.sign(
                    indexed.loc[first, metric] - indexed.loc[second, metric]
                )
                if retrieval_direction * compatibility_direction < 0:
                    reversal_rows.append(
                        {
                            "dataset": dataset,
                            "model_a": first,
                            "model_b": second,
                            "metric": metric,
                        }
                    )
        axis.scatter(merged.retrieval_rank, merged.cp_auc_rank, label="CP rank", marker="o")
        axis.scatter(merged.retrieval_rank, merged.fitb_acc_rank, label="FITB rank", marker="s")
        for _, row in merged.iterrows():
            axis.annotate(DISPLAY[row.model_key], (row.retrieval_rank, row.cp_auc_rank), fontsize=7)
        axis.set(xlabel="LookBench Fine R@1 rank", ylabel="Compatibility rank", title=title)
        axis.set_xticks(range(1, 6))
        axis.set_yticks(range(1, 6))
        axis.invert_xaxis()
        axis.invert_yaxis()
    axes[0].legend(frameon=False)
    save_figure(figure, "retrieval_vs_compatibility", figure_root)
    pd.DataFrame(retrieval_correlations).to_csv(
        table_root / "lookbench_rank_correlations.csv", index=False
    )
    pd.concat(retrieval_rows, ignore_index=True).to_csv(
        table_root / "lookbench_overlap_by_protocol.csv", index=False
    )
    pd.DataFrame(reversal_rows).to_csv(
        table_root / "lookbench_rank_reversals.csv", index=False
    )

    rank = main_results.pivot(index="representation", columns="dataset")
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for axis, metric, label in zip(axes, ("cp_rank", "fitb_rank"), ("CP", "FITB"), strict=True):
        for model_key, display_name in DISPLAY.items():
            x = rank.loc[model_key, (metric, "historical_polyvore_d")]
            y = rank.loc[model_key, (metric, "polyvore_d_clean")]
            axis.scatter(x, y, color=COLORS[model_key])
            axis.annotate(display_name, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.plot((1, 7), (1, 7), "--", color="0.5", linewidth=1)
        axis.set(xlabel="Historical rank", ylabel="Clean rank", title=label)
        axis.set_xticks(range(1, 8))
        axis.set_yticks(range(1, 8))
        axis.invert_xaxis()
        axis.invert_yaxis()
    save_figure(figure, "historical_vs_clean_ranking", figure_root)

    efficiency = pd.read_csv("artifacts/efficiency_profile.csv")
    clean_metrics = main_results.loc[main_results.dataset == "polyvore_d_clean", [
        "representation", "cp_auc", "fitb_acc"
    ]]
    pareto = efficiency.merge(clean_metrics, on="representation")
    plot_specs = (
        ("cp_auc", "estimated_gflops_per_image", "CP AUC", "Estimated GFLOPs/image"),
        ("fitb_acc", "estimated_gflops_per_image", "FITB accuracy", "Estimated GFLOPs/image"),
        ("cp_auc", "repeated_latency_ms_per_image", "CP AUC", "Latency (ms/image)"),
        ("fitb_acc", "repeated_latency_ms_per_image", "FITB accuracy", "Latency (ms/image)"),
        ("cp_auc", "active_image_parameters", "CP AUC", "Active image parameters"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, (metric, cost, ylabel, xlabel) in zip(axes.ravel(), plot_specs, strict=False):
        dominated = pareto_flags(pareto, metric, cost)
        pareto[f"dominated_{metric}_vs_{cost}"] = dominated
        for index, row in pareto.iterrows():
            axis.scatter(
                row[cost], row[metric], color=COLORS[row.representation],
                marker="x" if dominated.loc[index] else "o", alpha=0.65 if dominated.loc[index] else 1,
            )
            axis.annotate(DISPLAY[row.representation], (row[cost], row[metric]), fontsize=6)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        if cost == "active_image_parameters":
            axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes.ravel()[-1].axis("off")
    save_figure(figure, "accuracy_efficiency_pareto", figure_root)
    pareto.to_csv(table_root / "efficiency_pareto.csv", index=False)

    pairwise = pd.read_csv("artifacts/paper_pairwise_statistics.csv")
    cluster_pairwise = pd.read_csv("artifacts/component_cluster_pairwise.csv").rename(
        columns={
            "metric": "cluster_metric",
            "ci_low": "cluster_ci_low",
            "ci_high": "cluster_ci_high",
            "bootstrap_p": "cluster_bootstrap_p",
        }
    )
    metric_map = {
        "CP ROC-AUC difference": "cp_auc",
        "FITB accuracy difference": "fitb_acc",
    }
    pairwise["cluster_metric"] = pairwise.metric.map(metric_map)
    pairwise = pairwise.merge(
        cluster_pairwise[
            [
                "dataset",
                "comparison",
                "cluster_metric",
                "cluster_ci_low",
                "cluster_ci_high",
                "cluster_bootstrap_p",
            ]
        ],
        on=["dataset", "comparison", "cluster_metric"],
    )
    pairwise.to_csv(table_root / "pairwise_statistics.csv", index=False)

    cluster_intervals = pd.read_csv("artifacts/component_cluster_bootstrap.csv").pivot(
        index=["dataset", "representation"], columns="metric", values=["ci_low", "ci_high"]
    )
    cluster_intervals.columns = [f"cluster_{first}_{second}" for first, second in cluster_intervals.columns]
    cluster_intervals.reset_index().to_csv(
        table_root / "component_cluster_confidence_intervals.csv", index=False
    )

    robustness = pd.concat(
        (pd.read_csv("artifacts/historical_robustness.csv"), pd.read_csv("artifacts/clean_robustness.csv")),
        ignore_index=True,
    )
    low_data = robustness.loc[robustness.learner == "logistic"].copy()
    low_data.to_csv(table_root / "low_data_results.csv", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    for column, dataset in enumerate(("historical_polyvore_d", "polyvore_d_clean")):
        subset = low_data.loc[low_data.dataset == dataset]
        for model_key, display_name in DISPLAY.items():
            rows = subset.loc[subset.representation == model_key].sort_values("train_fraction")
            axes[0, column].plot(rows.train_fraction * 100, rows.cp_auc, marker="o", label=display_name)
            axes[1, column].plot(rows.train_fraction * 100, rows.fitb_acc, marker="o")
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("CP AUC")
        axes[1, column].set_ylabel("FITB accuracy")
        axes[1, column].set_xlabel("Compatibility labels retained (%)")
    axes[0, 1].legend(fontsize=6, frameon=False, ncol=2)
    save_figure(figure, "low_data_curves", figure_root)

    downstream = robustness.loc[robustness.train_fraction == 1].copy()
    downstream.to_csv(table_root / "downstream_learner_robustness.csv", index=False)
    learner_agreement = []
    for dataset in downstream.dataset.unique():
        pivot = downstream.loc[downstream.dataset == dataset].pivot(
            index="representation", columns="learner", values=["cp_auc", "fitb_acc"]
        )
        for learner in ("mlp", "xgboost"):
            for metric in ("cp_auc", "fitb_acc"):
                rho = spearmanr(pivot[(metric, "logistic")], pivot[(metric, learner)])
                tau = kendalltau(pivot[(metric, "logistic")], pivot[(metric, learner)])
                learner_agreement.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison": f"logistic_vs_{learner}",
                        "spearman_rho": rho.statistic,
                        "kendall_tau": tau.statistic,
                    }
                )
    pd.DataFrame(learner_agreement).to_csv(
        table_root / "downstream_learner_rank_agreement.csv", index=False
    )

    pca_rows = []
    for protocol_prefix, dataset in (("historical", "historical_polyvore_d"), ("clean", "polyvore_d_clean")):
        files = {
            "PCA-128": f"artifacts/{protocol_prefix}_pca128_results.csv",
            "PCA-256": "artifacts/primary_results.csv" if protocol_prefix == "historical" else "artifacts/clean_primary_results.csv",
            "PCA-512": f"artifacts/{protocol_prefix}_pca512_results.csv",
            "Native": "artifacts/native_results.csv" if protocol_prefix == "historical" else "artifacts/clean_native_results.csv",
        }
        for track, path in files.items():
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                pca_rows.append(
                    {"dataset": dataset, "track": track, "model": DISPLAY[row.representation], "cp_auc": row.cp_auc, "fitb_acc": row.fitb_acc}
                )
    pd.DataFrame(pca_rows).to_csv(table_root / "pca_native_robustness.csv", index=False)

    scaled = pd.read_csv("artifacts/primary_results.csv")
    unscaled = pd.read_csv("artifacts/provisional_unscaled_primary/primary_results.csv")
    standardization = scaled[["representation", "cp_auc", "fitb_acc", "best_c"]].merge(
        unscaled[["representation", "cp_auc", "fitb_acc", "best_c"]],
        on="representation",
        suffixes=("_scaled", "_unscaled"),
    )
    standardization["cp_auc_change"] = standardization.cp_auc_scaled - standardization.cp_auc_unscaled
    standardization["fitb_change"] = standardization.fitb_acc_scaled - standardization.fitb_acc_unscaled
    standardization.to_csv(table_root / "feature_standardization_audit.csv", index=False)


if __name__ == "__main__":
    main()
