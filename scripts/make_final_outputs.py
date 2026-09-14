"""Assemble publication tables and figures from immutable final-phase artifacts."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t

from repbench.encoders.registry import PRIMARY_MODEL_KEYS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES = Path("reports/tables/final")
FIGURES = Path("reports/figures/final")
NAMES = {
    "resnet50": "ResNet50",
    "dinov3_vitl16": "DINOv3 ViT-L/16",
    "clip_vitl14_336": "CLIP ViT-L/14@336",
    "siglip2_b16_384": "SigLIP2-B/16@384",
    "fashionclip2": "FashionCLIP 2.0",
    "marqo_fashionsiglip": "Marqo-FashionSigLIP",
    "gr_lite": "GR-Lite",
}


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(frame.to_markdown(index=False), encoding="utf-8")


def primary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing = pd.read_csv("artifacts/paper_main_results_with_cluster_ci.csv")
    historical = existing[existing.dataset == "historical_polyvore_d"].copy()
    polyvore = existing[existing.dataset == "polyvore_d_clean"].copy()
    for frame in (historical, polyvore):
        frame["cp_auc_ci_low"] = frame.cp_auc_cluster_low
        frame["cp_auc_ci_high"] = frame.cp_auc_cluster_high
        frame["pr_auc_ci_low"] = frame.pr_auc_cluster_low
        frame["pr_auc_ci_high"] = frame.pr_auc_cluster_high
        frame["fitb_ci_low"] = frame.fitb_acc_cluster_low
        frame["fitb_ci_high"] = frame.fitb_acc_cluster_high
    iqon = pd.read_csv("artifacts/final/statistics/iqon_primary_with_group_ci.csv")
    keep = [
        "dataset", "representation", "cp_auc", "cp_auc_ci_low", "cp_auc_ci_high",
        "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "fitb_acc", "fitb_ci_low",
        "fitb_ci_high", "selected_c",
    ]
    historical = historical[keep]
    polyvore = polyvore[keep]
    iqon = iqon[keep]
    combined = pd.concat((polyvore, iqon), ignore_index=True)
    for frame in (historical, combined):
        frame["display_name"] = frame.representation.map(NAMES)
        frame["cp_rank"] = frame.groupby("dataset").cp_auc.rank(ascending=False, method="min").astype(int)
        frame["fitb_rank"] = frame.groupby("dataset").fitb_acc.rank(ascending=False, method="min").astype(int)
    save(historical, "historical_polyvore_results")
    save(combined, "main_polyvore_iqon_results")
    return historical, polyvore, iqon


def dataset_table() -> pd.DataFrame:
    poly = json.loads(Path("data/protocols/polyvore_d_clean/manifest.json").read_text(encoding="utf-8"))
    iqon = json.loads(Path("data/protocols/iqon3000_clean/manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(Path("artifacts/iqon3000_selection_audit.json").read_text(encoding="utf-8"))
    rows = [
        {"protocol": "historical_polyvore_d", "role": "literature comparison", "train_outfits": 16995,
         "valid_outfits": 3000, "test_outfits": 15145, "zero_item_overlap": False,
         "zero_exact_image_overlap": False, "negative_source": "packaged", "limitation": "known cross-split item overlap"},
        {"protocol": "polyvore_d_clean", "role": "development/generalization evidence",
         "train_outfits": poly["splits"]["train"]["outfits"], "valid_outfits": poly["splits"]["valid"]["outfits"],
         "test_outfits": poly["splits"]["test"]["outfits"], "zero_item_overlap": True,
         "zero_exact_image_overlap": True, "negative_source": "deterministic type-matched corruption",
         "limitation": "Polyvore influenced development; not prospective blind test"},
        {"protocol": "iqon3000_clean", "role": "prospective independent replication",
         "train_outfits": iqon["splits"]["train"]["outfits"], "valid_outfits": iqon["splits"]["valid"]["outfits"],
         "test_outfits": iqon["splits"]["test"]["outfits"], "zero_item_overlap": True,
         "zero_exact_image_overlap": True, "negative_source": "deterministic type-matched corruption",
         "limitation": f"retains {selection['selected_fraction']:.1%}; users are not split-disjoint"},
        {"protocol": "A100", "role": "external human/expert-grounded evaluation", "train_outfits": 0,
         "valid_outfits": 0, "test_outfits": 200, "zero_item_overlap": None,
         "zero_exact_image_overlap": True, "negative_source": "unchanged archive candidates",
         "limitation": "100 LAT + 100 AAT; wide uncertainty; no formal archive license file"},
    ]
    frame = pd.DataFrame(rows)
    save(frame, "dataset_protocol_comparison")
    return frame


def model_and_efficiency_tables(polyvore: pd.DataFrame, iqon: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv("artifacts/model_audit.csv")
    efficiency = pd.read_csv("artifacts/efficiency_profile.csv")
    iqon_root = Path(".venv/iqon3000_clean/embeddings/iqon3000_clean")
    extraction_rows = []
    for model in PRIMARY_MODEL_KEYS:
        metadata = [json.loads((iqon_root / model / f"{split}_metadata.json").read_text()) for split in ("train", "valid", "test")]
        seconds = sum(row["elapsed_seconds"] for row in metadata)
        count = 434_089
        extraction_rows.append(
            {"representation": model, "iqon_extraction_batch_size": 32,
             "iqon_extraction_seconds": seconds, "iqon_images_per_second": count / seconds,
             "iqon_peak_vram_mb": max(row["peak_vram_mb"] for row in metadata),
             "iqon_cache_mb": sum((iqon_root / model / f"{split}.npy").stat().st_size for split in ("train", "valid", "test")) / 2**20}
        )
    extraction = pd.DataFrame(extraction_rows)
    iqon_timing = []
    for model in PRIMARY_MODEL_KEYS:
        metrics = json.loads(Path(f"artifacts/final/iqon3000_clean/{model}/metrics.json").read_text())
        iqon_timing.append({"representation": model, "iqon_pca_seconds": metrics["pca_seconds"],
                            "iqon_classifier_seconds": metrics["classifier_train_seconds"]})
    efficiency = efficiency.merge(extraction, on="representation").merge(pd.DataFrame(iqon_timing), on="representation")
    characteristics = audit.merge(
        efficiency[["representation", "active_image_parameters", "checkpoint_size_mb", "estimated_gflops_per_image",
                    "repeated_latency_ms_per_image", "repeated_images_per_second", "repeated_peak_vram_mb",
                    "iqon_extraction_seconds", "iqon_images_per_second", "iqon_cache_mb", "iqon_pca_seconds",
                    "iqon_classifier_seconds"]], left_on="model_key", right_on="representation"
    )
    save(characteristics, "model_characteristics_and_efficiency")
    save(efficiency, "efficiency_full")

    pareto = []
    score_sets = {"polyvore_cp": polyvore.set_index("representation").cp_auc,
                  "polyvore_fitb": polyvore.set_index("representation").fitb_acc,
                  "iqon_cp": iqon.set_index("representation").cp_auc,
                  "iqon_fitb": iqon.set_index("representation").fitb_acc}
    costs = {"latency_ms": "repeated_latency_ms_per_image", "gflops": "estimated_gflops_per_image",
             "peak_vram_mb": "repeated_peak_vram_mb"}
    cost_frame = efficiency.set_index("representation")
    for score_name, scores in score_sets.items():
        for cost_name, cost_column in costs.items():
            for model in PRIMARY_MODEL_KEYS:
                dominated_by = [other for other in PRIMARY_MODEL_KEYS if other != model
                                and scores[other] >= scores[model] and cost_frame.loc[other, cost_column] <= cost_frame.loc[model, cost_column]
                                and (scores[other] > scores[model] or cost_frame.loc[other, cost_column] < cost_frame.loc[model, cost_column])]
                pareto.append({"analysis": score_name, "cost": cost_name, "representation": model,
                               "score": scores[model], "cost_value": cost_frame.loc[model, cost_column],
                               "pareto_dominated": bool(dominated_by), "dominated_by": ";".join(dominated_by)})
    save(pd.DataFrame(pareto), "efficiency_pareto")
    return efficiency


def robustness_tables(polyvore: pd.DataFrame, iqon: pd.DataFrame) -> None:
    native = pd.read_csv("artifacts/iqon_native_results.csv")
    primary = iqon[["representation", "cp_auc", "fitb_acc"]].rename(columns={"cp_auc": "pca256_cp_auc", "fitb_acc": "pca256_fitb_acc"})
    merged = primary.merge(native[["representation", "cp_auc", "fitb_acc", "best_c"]].rename(
        columns={"cp_auc": "native_cp_auc", "fitb_acc": "native_fitb_acc", "best_c": "native_selected_c"}), on="representation")
    merged["native_minus_pca_cp"] = merged.native_cp_auc - merged.pca256_cp_auc
    merged["native_minus_pca_fitb"] = merged.native_fitb_acc - merged.pca256_fitb_acc
    save(merged, "iqon_pca256_vs_native")
    xgb = pd.read_csv("artifacts/iqon_xgboost_robustness.csv")
    learner = primary.merge(xgb[["representation", "cp_auc", "fitb_acc", "train_seconds"]].rename(
        columns={"cp_auc": "xgboost_cp_auc", "fitb_acc": "xgboost_fitb_acc"}), on="representation")
    save(learner, "iqon_downstream_learner_robustness")
    for name, first, second in (
        ("iqon PCA256 vs native CP", merged.pca256_cp_auc, merged.native_cp_auc),
        ("iqon PCA256 vs native FITB", merged.pca256_fitb_acc, merged.native_fitb_acc),
        ("iqon LR vs XGBoost CP", learner.pca256_cp_auc, learner.xgboost_cp_auc),
        ("iqon LR vs XGBoost FITB", learner.pca256_fitb_acc, learner.xgboost_fitb_acc),
    ):
        rho, tau = spearmanr(first, second), kendalltau(first, second)
        save(pd.DataFrame([{"comparison": name, "spearman_rho": rho.statistic, "spearman_p": rho.pvalue,
                            "kendall_tau": tau.statistic, "kendall_p": tau.pvalue}]), name.lower().replace(" ", "_").replace("256", "256"))

    stability_rows, rank_rows = [], []
    primary_by_dataset = {"polyvore_d_clean": polyvore.set_index("representation"), "iqon3000_clean": iqon.set_index("representation")}
    for dataset in ("polyvore_d_clean", "iqon3000_clean"):
        runs = pd.read_csv(f"artifacts/final/stability/{dataset}/all_runs.csv")
        grouped = runs.groupby("model_key")
        for model, frame in grouped:
            for metric in ("cp_auc", "pr_auc", "fitb_accuracy"):
                stability_rows.append({"dataset": dataset, "representation": model, "metric": metric,
                                       "mean": frame[metric].mean(), "sd": frame[metric].std(ddof=1),
                                       "min": frame[metric].min(), "max": frame[metric].max()})
        for seed, frame in runs.groupby("construction_seed"):
            aligned = frame.set_index("model_key")
            for metric, primary_metric in (("cp_auc", "cp_auc"), ("fitb_accuracy", "fitb_acc")):
                rho, tau = spearmanr(primary_by_dataset[dataset].loc[list(PRIMARY_MODEL_KEYS), primary_metric],
                                     aligned.loc[list(PRIMARY_MODEL_KEYS), metric]), kendalltau(
                                         primary_by_dataset[dataset].loc[list(PRIMARY_MODEL_KEYS), primary_metric],
                                         aligned.loc[list(PRIMARY_MODEL_KEYS), metric])
                rank_rows.append({"dataset": dataset, "construction_seed": seed, "metric": metric,
                                  "spearman_rho": rho.statistic, "kendall_tau": tau.statistic})
    save(pd.DataFrame(stability_rows), "construction_seed_stability")
    save(pd.DataFrame(rank_rows), "construction_seed_rank_agreement")

    audit = pd.read_csv("artifacts/model_audit.csv").set_index("model_key")
    sensitivity_rows = []
    for dataset, frame in (("polyvore_d_clean", polyvore), ("iqon3000_clean", iqon)):
        possibly_exposed = frame.representation.map(audit.iqon_provenance_status).eq("possibly_exposed")
        clean = frame[~possibly_exposed].copy()
        clean["cp_rank_clean_provenance"] = clean.cp_auc.rank(ascending=False, method="min").astype(int)
        clean["fitb_rank_clean_provenance"] = clean.fitb_acc.rank(ascending=False, method="min").astype(int)
        clean.insert(0, "sensitivity_dataset", dataset)
        sensitivity_rows.append(clean)
    save(pd.concat(sensitivity_rows, ignore_index=True), "clean_provenance_sensitivity")


def a100_and_statistics_tables() -> pd.DataFrame:
    a100 = pd.read_csv("artifacts/final/statistics/a100_with_ci.csv")
    save(a100[a100.task.isin(["LAT", "AAT"])], "a100_headline")
    save(a100[a100.task.str.startswith("AAT-")], "a100_aat_dimensions")
    pairwise = pd.concat((pd.read_csv("artifacts/paper_pairwise_statistics.csv"),
                          pd.read_csv("artifacts/final/statistics/iqon_predefined_pairwise.csv"),
                          pd.read_csv("artifacts/final/statistics/a100_predefined_pairwise.csv")), ignore_index=True, sort=False)
    save(pairwise, "predefined_statistical_effects")
    ranks = pd.concat((pd.read_csv("artifacts/paper_rank_correlations.csv"),
                       pd.read_csv("artifacts/final/statistics/cross_dataset_rank_correlations.csv")), ignore_index=True)
    save(ranks, "rank_correlations")
    return a100


def baseline_table() -> pd.DataFrame:
    rows = []
    for dataset in ("polyvore_d_clean", "iqon3000_clean"):
        frame = pd.read_csv(f"artifacts/final/baseline/{dataset}/summary.csv")
        cp_half_width = t.ppf(0.975, len(frame) - 1) * frame.test_cp_auc.std(ddof=1) / np.sqrt(len(frame))
        fitb_half_width = t.ppf(0.975, len(frame) - 1) * frame.test_fitb_accuracy.std(ddof=1) / np.sqrt(len(frame))
        rows.append({"dataset": dataset, "system": "image-only OutfitTransformer adaptation",
                     "training_seeds": len(frame), "cp_auc_mean": frame.test_cp_auc.mean(), "cp_auc_sd": frame.test_cp_auc.std(ddof=1),
                     "cp_seed_ci_low": frame.test_cp_auc.mean() - cp_half_width,
                     "cp_seed_ci_high": frame.test_cp_auc.mean() + cp_half_width,
                     "cp_auc_min": frame.test_cp_auc.min(), "cp_auc_max": frame.test_cp_auc.max(),
                     "pr_auc_mean": frame.test_pr_auc.mean(), "fitb_mean": frame.test_fitb_accuracy.mean(),
                     "fitb_sd": frame.test_fitb_accuracy.std(ddof=1),
                     "fitb_seed_ci_low": frame.test_fitb_accuracy.mean() - fitb_half_width,
                     "fitb_seed_ci_high": frame.test_fitb_accuracy.mean() + fitb_half_width,
                     "training_seconds_mean": frame.training_seconds.mean()})
    result = pd.DataFrame(rows)
    save(result, "task_specific_baseline")
    external_path = Path("artifacts/final/a100_outfit_transformer/summary.csv")
    if external_path.is_file():
        external = pd.read_csv(external_path)
        save(external, "task_specific_baseline_a100")
        numeric = [column for column in external.columns if column not in {"training_source", "task", "seed"}]
        aggregates = external.groupby(["training_source", "task"])[numeric].agg(["mean", "std"])
        aggregates.columns = ["_".join(column) for column in aggregates.columns]
        save(aggregates.reset_index(), "task_specific_baseline_a100_aggregate")
    return result


def figures(polyvore: pd.DataFrame, iqon: pd.DataFrame, a100: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    poly = polyvore.set_index("representation").loc[list(PRIMARY_MODEL_KEYS)]
    iq = iqon.set_index("representation").loc[list(PRIMARY_MODEL_KEYS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, label in ((axes[0], "cp_auc", "CP ROC-AUC"), (axes[1], "fitb_acc", "FITB accuracy")):
        axis.scatter(poly[metric], iq[metric])
        for model in PRIMARY_MODEL_KEYS:
            axis.annotate(NAMES[model], (poly.loc[model, metric], iq.loc[model, metric]), fontsize=7)
        axis.set_xlabel(f"Polyvore-D-Clean {label}"); axis.set_ylabel(f"IQON3000-Clean {label}"); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "polyvore_vs_iqon_ranking.png", dpi=240); fig.savefig(FIGURES / "polyvore_vs_iqon_ranking.pdf"); plt.close(fig)

    headline = a100[a100.task.isin(["LAT", "AAT"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = .38; positions = np.arange(7)
    for axis, task in zip(axes, ("LAT", "AAT"), strict=True):
        subset = headline[headline.task == task]
        for offset, source in ((-.19, "polyvore_d_clean"), (.19, "iqon3000_clean")):
            values = subset[subset.training_source == source].set_index("representation").loc[list(PRIMARY_MODEL_KEYS)]
            axis.bar(positions + offset, values.accuracy, width, label=source)
        axis.set_title(task); axis.set_xticks(positions, [NAMES[m] for m in PRIMARY_MODEL_KEYS], rotation=55, ha="right", fontsize=7); axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Accuracy"); axes[1].legend(fontsize=8); fig.tight_layout();
    fig.savefig(FIGURES / "a100_by_training_source.png", dpi=240); fig.savefig(FIGURES / "a100_by_training_source.pdf"); plt.close(fig)

    dimensions = ("Color", "Style", "Occasion", "Season", "Material", "Balance")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, source in zip(axes, ("polyvore_d_clean", "iqon3000_clean"), strict=True):
        values = np.asarray([[a100[(a100.training_source == source) & (a100.representation == model) & (a100.task == f"AAT-{dimension}")].accuracy.iloc[0]
                              for dimension in dimensions] for model in PRIMARY_MODEL_KEYS])
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(source); axis.set_xticks(range(6), dimensions, rotation=45, ha="right"); axis.set_yticks(range(7), [NAMES[m] for m in PRIMARY_MODEL_KEYS], fontsize=7)
    fig.colorbar(image, ax=axes, label="Accuracy", shrink=.8); fig.subplots_adjust(bottom=.2, wspace=.3)
    fig.savefig(FIGURES / "a100_aat_dimension_heatmap.png", dpi=240); fig.savefig(FIGURES / "a100_aat_dimension_heatmap.pdf"); plt.close(fig)

    cost = efficiency.set_index("representation")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, scores, ylabel in ((axes[0, 0], poly.cp_auc, "Polyvore CP AUC"), (axes[0, 1], poly.fitb_acc, "Polyvore FITB"),
                                  (axes[1, 0], iq.cp_auc, "IQON CP AUC"), (axes[1, 1], iq.fitb_acc, "IQON FITB")):
        axis.scatter(cost.estimated_gflops_per_image, scores)
        for model in PRIMARY_MODEL_KEYS: axis.annotate(NAMES[model], (cost.loc[model, "estimated_gflops_per_image"], scores[model]), fontsize=6)
        axis.set_xscale("log"); axis.set_xlabel("Estimated GFLOPs/image (log)"); axis.set_ylabel(ylabel); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "accuracy_efficiency_pareto_final.png", dpi=240); fig.savefig(FIGURES / "accuracy_efficiency_pareto_final.pdf"); plt.close(fig)

    stability = pd.read_csv(TABLES / "construction_seed_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, dataset in zip(axes, ("polyvore_d_clean", "iqon3000_clean"), strict=True):
        frame = stability[(stability.dataset == dataset) & (stability.metric == "cp_auc")].set_index("representation").loc[list(PRIMARY_MODEL_KEYS)]
        axis.errorbar(np.arange(7), frame["mean"], yerr=frame.sd, fmt="o"); axis.set_xticks(range(7), [NAMES[m] for m in PRIMARY_MODEL_KEYS], rotation=55, ha="right", fontsize=7); axis.set_title(dataset); axis.set_ylabel("CP AUC mean ± SD"); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "construction_seed_stability.png", dpi=240); fig.savefig(FIGURES / "construction_seed_stability.pdf"); plt.close(fig)

    native = pd.read_csv(TABLES / "iqon_pca256_vs_native.csv").set_index("representation").loc[list(PRIMARY_MODEL_KEYS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, left, right, label in ((axes[0], "pca256_cp_auc", "native_cp_auc", "CP AUC"), (axes[1], "pca256_fitb_acc", "native_fitb_acc", "FITB")):
        for position, model in enumerate(PRIMARY_MODEL_KEYS): axis.plot([0, 1], [native.loc[model, left], native.loc[model, right]], marker="o", label=NAMES[model] if axis is axes[0] else None)
        axis.set_xticks([0, 1], ["PCA-256", "Native"]); axis.set_ylabel(label); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "iqon_pca_native_robustness.png", dpi=240); fig.savefig(FIGURES / "iqon_pca_native_robustness.pdf"); plt.close(fig)

    for source, destination in (("reports/figures/retrieval_vs_compatibility.pdf", "lookbench_retrieval_vs_compatibility.pdf"),
                                ("reports/figures/low_data_curves.pdf", "polyvore_low_data_curves.pdf")):
        shutil.copy2(source, FIGURES / destination)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    _historical, polyvore, iqon = primary_tables()
    dataset_table()
    efficiency = model_and_efficiency_tables(polyvore, iqon)
    robustness_tables(polyvore, iqon)
    a100 = a100_and_statistics_tables()
    baseline_table()
    save(pd.read_csv("artifacts/model_audit.csv"), "provenance_exposure")
    save(pd.read_csv("artifacts/final/calibration/calibration_metrics.csv"), "calibration_metrics")
    protocol_hash = json.loads(Path("artifacts/final_experimental_protocol_hash.json").read_text(encoding="utf-8"))
    run_metadata = json.loads(Path("artifacts/final/iqon3000_clean/resnet50/run_metadata.json").read_text(encoding="utf-8"))
    save(pd.DataFrame([{
        "protocol": protocol_hash["protocol_name"],
        "protocol_sha256": protocol_hash["combined_sha256"],
        "iqon_split_manifest_sha256": run_metadata["protocol_manifest_sha256"],
        "git_commit": "unavailable: repository has no initial commit",
        "platform": platform.platform(),
        "python": run_metadata["python"],
        "torch": run_metadata["torch"],
        "transformers": run_metadata["transformers"],
        "scikit_learn": run_metadata["sklearn"],
        "cuda": run_metadata["cuda"],
        "gpu": run_metadata["gpu"],
        "artifact_manifest": "artifacts/final_artifact_manifest.csv",
        "report_manifest": "artifacts/final_report_manifest.csv",
    }]), "reproducibility_inventory")
    for name in (
        "representation_families.csv", "lookbench_overlap_by_protocol.csv",
        "lookbench_rank_correlations.csv", "lookbench_rank_reversals.csv",
        "pca_native_robustness.csv", "downstream_learner_robustness.csv",
        "low_data_results.csv", "feature_standardization_audit.csv",
    ):
        shutil.copy2(Path("reports/tables") / name, TABLES / name)
    figures(polyvore, iqon, a100, efficiency)


if __name__ == "__main__":
    main()
