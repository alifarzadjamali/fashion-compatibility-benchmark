from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

SEED = 20260912
RESULT_TRACKS = {
    "clean_primary": ("polyvore_d_clean", "pca256", 256),
    "historical_pca128": ("historical_polyvore_d", "pca128", 128),
    "historical_pca512": ("historical_polyvore_d", "pca512", 512),
    "clean_pca128": ("polyvore_d_clean", "pca128", 128),
    "clean_pca512": ("polyvore_d_clean", "pca512", 512),
    "clean_native": ("polyvore_d_clean", "native", None),
    "historical_regenerated": ("historical_polyvore_d_regenerated", "pca256", 256),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_digest() -> str:
    digest = hashlib.sha256()
    files = []
    for root in (Path("src"), Path("scripts"), Path("configs"), Path("tests")):
        files.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(files):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    registry = pd.read_csv("artifacts/experiment_registry.csv")
    additions = []
    for prefix, (dataset, track, pca_dim) in RESULT_TRACKS.items():
        for _, row in pd.read_csv(f"artifacts/{prefix}_results.csv").iterrows():
            additions.append(
                {
                    "run_id": f"{prefix}_{row.representation}_s{SEED}",
                    "dataset": dataset,
                    "split": "test",
                    "representation": row.representation,
                    "checkpoint": f"{row.checkpoint}@{row.revision}",
                    "track": track,
                    "pca_dim": pca_dim,
                    "classifier": "logistic_regression",
                    "seed": SEED,
                    "train_fraction": 1.0,
                    "cp_auc": row.cp_auc,
                    "fitb_acc": row.fitb_acc,
                    "pr_auc": row.pr_auc,
                    "train_seconds": row.train_seconds,
                    "extract_ips": row.extract_images_per_second,
                    "peak_vram_mb": row.peak_vram_mb,
                    "notes": "paper-readiness controlled robustness track",
                }
            )
    for prefix in ("historical", "clean"):
        for _, row in pd.read_csv(f"artifacts/{prefix}_robustness.csv").iterrows():
            additions.append(
                {
                    "run_id": (
                        f"robust_{prefix}_{row.representation}_{row.learner}_"
                        f"f{row.train_fraction:g}_s{SEED}"
                    ),
                    "dataset": row.dataset,
                    "split": "test",
                    "representation": row.representation,
                    "checkpoint": "see model_audit.csv",
                    "track": "pca256",
                    "pca_dim": 256,
                    "classifier": row.learner,
                    "seed": SEED,
                    "train_fraction": row.train_fraction,
                    "cp_auc": row.cp_auc,
                    "fitb_acc": row.fitb_acc,
                    "pr_auc": row.pr_auc,
                    "train_seconds": row.train_seconds,
                    "extract_ips": None,
                    "peak_vram_mb": None,
                    "notes": "fixed common learner or compatibility-label fraction",
                }
            )
    registry = pd.concat((registry, pd.DataFrame(additions)), ignore_index=True)
    registry = registry.drop_duplicates("run_id", keep="last").sort_values("run_id")
    registry.to_csv("artifacts/experiment_registry.csv", index=False)

    important = [
        Path("artifacts/paper_main_results.csv"),
        Path("artifacts/paper_pairwise_statistics.csv"),
        Path("artifacts/component_cluster_bootstrap.csv"),
        Path("artifacts/clean_primary_cp_predictions.parquet"),
        Path("artifacts/clean_primary_fitb_predictions.parquet"),
        Path("artifacts/efficiency_profile.csv"),
        Path("artifacts/clean_robustness.csv"),
        Path("artifacts/historical_robustness.csv"),
        Path("data/protocols/polyvore_d_clean/manifest.json"),
        Path("artifacts/clean_image_duplicate_audit.json"),
        Path("artifacts/hf_access_reverification.json"),
        Path("reports/reviewer_red_team.md"),
        Path("reports/paper_readiness_report.md"),
    ]
    packages = {}
    for package in (
        "huggingface-hub",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
        "xgboost",
    ):
        packages[package] = importlib.metadata.version(package)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "primary_generalization_protocol": "polyvore_d_clean",
        "comparison_protocol": "historical_polyvore_d",
        "secondary_bridge_protocol": "historical_polyvore_d_regenerated",
        "seed": SEED,
        "bootstrap_resamples": 2000,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "packages": packages,
        "code_lock_sha256": code_digest(),
        "artifact_sha256": {path.as_posix(): sha256(path) for path in important},
    }
    Path("artifacts/paper_reproducibility_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
