"""Fail-closed validation and checksums for the completed final experimental package."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_memory_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return None


def validate_iqon() -> list[dict]:
    root = Path("artifacts/final/iqon3000_clean")
    checks = []
    for model in PRIMARY_MODEL_KEYS:
        directory = root / model
        marker = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
        for name, expected in marker["artifacts"].items():
            path = directory / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"IQON artifact hash mismatch: {model}/{name}")
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["revision"] != PRIMARY_ENCODERS[model].revision:
            raise ValueError(f"Checkpoint mismatch in IQON run: {model}")
        for split, cp_count, fitb_count in (
            ("train", 173_600, 86_800),
            ("valid", 24_800, 12_400),
            ("test", 49_600, 24_800),
        ):
            cp = pd.read_parquet(directory / f"cp_predictions_{split}.parquet")
            fitb = pd.read_parquet(directory / f"fitb_predictions_{split}.parquet")
            candidates = pd.read_parquet(directory / f"fitb_candidate_scores_{split}.parquet")
            if len(cp) != cp_count or cp.example_id.duplicated().any():
                raise ValueError(f"Wrong/duplicate IQON CP IDs: {model}/{split}")
            if len(fitb) != fitb_count or fitb.question_id.duplicated().any():
                raise ValueError(f"Wrong/duplicate IQON FITB IDs: {model}/{split}")
            if len(candidates) != fitb_count * 4 or candidates.groupby("question_id").size().ne(4).any():
                raise ValueError(f"Wrong IQON candidates: {model}/{split}")
            if not np.isfinite(cp.probability).all() or not cp.probability.between(0, 1).all():
                raise ValueError(f"Invalid IQON probabilities: {model}/{split}")
            if cp.probability.nunique() <= 1:
                raise ValueError(f"Constant IQON predictions: {model}/{split}")
        checks.append({"scope": "iqon_primary", "model_key": model, "valid": True})
    return checks


def validate_a100() -> list[dict]:
    root = Path("artifacts/final/a100")
    predictions = pd.read_parquet(root / "predictions.parquet")
    candidates = pd.read_parquet(root / "candidate_scores.parquet")
    key = ["training_source", "model_key", "task", "question_num"]
    if len(predictions) != 2 * 7 * 2 * 100 or predictions.duplicated(key).any():
        raise ValueError("A100 prediction count/alignment failure")
    if len(candidates) != 2 * 7 * 2 * 100 * 5 or candidates.groupby(key).size().ne(5).any():
        raise ValueError("A100 candidate count/alignment failure")
    if not np.isfinite(candidates.score).all() or not candidates.score.between(0, 1).all():
        raise ValueError("A100 has non-finite candidate scores")
    return [{"scope": "a100_external", "model_key": model, "valid": True} for model in PRIMARY_MODEL_KEYS]


def validate_baseline() -> list[dict]:
    checks = []
    specifications = {
        "polyvore_d_clean": {"seeds": set(range(42, 47)), "cp": 30_284, "fitb": 15_142},
        "iqon3000_clean": {"seeds": set(range(42, 47)), "cp": 49_600, "fitb": 24_800},
    }
    for dataset, specification in specifications.items():
        root = Path("artifacts/final/baseline") / dataset
        summary = pd.read_csv(root / "summary.csv")
        if set(summary.seed.astype(int)) != specification["seeds"] or summary.seed.duplicated().any():
            raise ValueError(f"Missing or duplicate baseline seeds: {dataset}")
        for seed in specification["seeds"]:
            directory = root / f"seed_{seed}"
            cp = pd.read_parquet(directory / "cp_predictions_test.parquet")
            fitb = pd.read_parquet(directory / "fitb_predictions_test.parquet")
            candidates = pd.read_parquet(directory / "fitb_candidate_scores_test.parquet")
            if len(cp) != specification["cp"] or cp.example_index.duplicated().any():
                raise ValueError(f"Invalid baseline CP predictions: {dataset}/{seed}")
            if len(fitb) != specification["fitb"] or fitb.question_index.duplicated().any():
                raise ValueError(f"Invalid baseline FITB predictions: {dataset}/{seed}")
            if len(candidates) != specification["fitb"] * 4:
                raise ValueError(f"Invalid baseline candidate scores: {dataset}/{seed}")
            if candidates.groupby("question_index").size().ne(4).any():
                raise ValueError(f"Invalid baseline candidate structure: {dataset}/{seed}")
            if not np.isfinite(cp.probability).all() or not cp.probability.between(0, 1).all():
                raise ValueError(f"Invalid baseline probabilities: {dataset}/{seed}")
            if cp.probability.nunique() <= 1:
                raise ValueError(f"Constant baseline predictions: {dataset}/{seed}")
        checks.append({"scope": "task_baseline", "dataset": dataset, "valid": True})
    return checks


def validate_a100_baseline() -> list[dict]:
    root = Path("artifacts/final/a100_outfit_transformer")
    predictions = pd.read_parquet(root / "predictions.parquet")
    candidates = pd.read_parquet(root / "candidate_scores.parquet")
    expected = {"polyvore_d_clean": set(range(42, 47)), "iqon3000_clean": set(range(42, 47))}
    key = ["training_source", "seed", "task", "question_num"]
    for source, seeds in expected.items():
        observed = set(predictions.loc[predictions.training_source == source, "seed"].astype(int))
        if observed != seeds:
            raise ValueError(f"A100 baseline seed mismatch: {source}")
    if predictions.duplicated(key).any() or candidates.groupby(key).size().ne(5).any():
        raise ValueError("A100 baseline prediction/candidate alignment failure")
    if not np.isfinite(candidates.score).all() or not candidates.score.between(0, 1).all():
        raise ValueError("A100 baseline has invalid scores")
    return [{"scope": "a100_task_baseline", "valid": True}]


def validate_paper_outputs() -> list[dict]:
    required_reports = [
        "final_experiment_report.md",
        "final_statistics_report.md",
        "final_calibration_report.md",
        "final_efficiency_report.md",
        "final_cross_dataset_report.md",
        "final_a100_report.md",
        "final_reviewer_red_team.md",
        "paper_readiness_report_v2.md",
    ]
    required_tables = [
        "main_polyvore_iqon_results.csv",
        "dataset_protocol_comparison.csv",
        "model_characteristics_and_efficiency.csv",
        "a100_headline.csv",
        "predefined_statistical_effects.csv",
        "construction_seed_stability.csv",
        "calibration_metrics.csv",
        "efficiency_pareto.csv",
        "task_specific_baseline.csv",
        "clean_provenance_sensitivity.csv",
        "reproducibility_inventory.csv",
        "low_data_results.csv",
        "pca_native_robustness.csv",
        "downstream_learner_robustness.csv",
        "subgroup_performance.csv",
        "failure_consensus_audit.csv",
    ]
    required_figures = [
        "polyvore_vs_iqon_ranking.pdf",
        "a100_by_training_source.pdf",
        "a100_aat_dimension_heatmap.pdf",
        "lookbench_retrieval_vs_compatibility.pdf",
        "accuracy_efficiency_pareto_final.pdf",
        "reliability_diagrams.pdf",
        "construction_seed_stability.pdf",
        "polyvore_low_data_curves.pdf",
        "iqon_pca_native_robustness.pdf",
    ]
    paths = (
        [Path("reports") / name for name in required_reports]
        + [Path("reports/tables/final") / name for name in required_tables]
        + [Path("reports/figures/final") / name for name in required_figures]
    )
    missing = [path.as_posix() for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Missing/empty paper outputs: {missing}")
    return [{"scope": "paper_outputs", "valid": True, "required_files": len(paths)}]


def main() -> None:
    checks = (
        validate_iqon()
        + validate_a100()
        + validate_baseline()
        + validate_a100_baseline()
        + validate_paper_outputs()
    )
    final_root = Path("artifacts/final")
    manifest_rows = []
    for path in sorted(value for value in final_root.rglob("*") if value.is_file()):
        manifest_rows.append(
            {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv("artifacts/final_artifact_manifest.csv", index=False)
    report_rows = []
    for path in sorted(value for value in Path("reports").rglob("*") if value.is_file()):
        report_rows.append(
            {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    pd.DataFrame(report_rows).to_csv("artifacts/final_report_manifest.csv", index=False)
    run_ids = []
    for path in final_root.rglob("run_metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "run_id" in metadata:
            run_ids.append(metadata["run_id"])
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run IDs in final package")
    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "artifact_count": len(manifest),
        "duplicate_run_ids": 0,
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "ram_bytes": physical_memory_bytes(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "transformers": version("transformers"),
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "cuda": torch.version.cuda,
        },
    }
    Path("artifacts/final_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
