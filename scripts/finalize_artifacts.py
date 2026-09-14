from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS
from repbench.models.logistic import SHARED_C_GRID

PEAK_VRAM_FALLBACK = {
    "resnet50": 764.16162109375,
    "fashionclip2": 398.60302734375,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for name in ("primary", "native"):
        path = Path(f"artifacts/{name}_results.csv")
        frame = pd.read_csv(path)
        for model, value in PEAK_VRAM_FALLBACK.items():
            frame.loc[frame.representation == model, "peak_vram_mb"] = value
        frame["peak_vram_measurement"] = "CUDA max allocated; batch=64"
        frame.to_csv(path, index=False)

    registry_path = Path("artifacts/experiment_registry.csv")
    registry = pd.read_csv(registry_path)
    registry = registry.loc[~registry.run_id.str.startswith(("pca256_", "native_"))]
    rows = []
    for track_name, filename in (("pca256", "primary_results.csv"), ("native", "native_results.csv")):
        for row in pd.read_csv(Path("artifacts") / filename).itertuples():
            rows.append(
                {
                    "run_id": f"{track_name}_{row.representation}_s20260912",
                    "dataset": "historical_polyvore_d",
                    "split": "test",
                    "representation": row.representation,
                    "checkpoint": f"{row.checkpoint}@{row.revision}",
                    "track": track_name,
                    "pca_dim": 256 if track_name == "pca256" else np.nan,
                    "classifier": "standardized_logistic_regression",
                    "seed": 20260912,
                    "train_fraction": 1.0,
                    "cp_auc": row.cp_auc,
                    "fitb_acc": row.fitb_acc,
                    "pr_auc": row.pr_auc,
                    "train_seconds": row.train_seconds,
                    "extract_ips": row.extract_images_per_second,
                    "peak_vram_mb": row.peak_vram_mb,
                    "notes": "historical packaged split; known cross-split item overlap; common C grid",
                }
            )
    pd.concat([registry, pd.DataFrame(rows)], ignore_index=True).to_csv(registry_path, index=False)

    embedding_rows = []
    root = Path("data/embeddings/historical_polyvore_d")
    for model in PRIMARY_MODEL_KEYS:
        for split in ("train", "valid", "test"):
            values = np.load(root / model / f"{split}.npy", mmap_mode="r")
            ids = json.loads((root / model / f"{split}_item_ids.json").read_text())
            norms = np.linalg.norm(values, axis=1)
            embedding_rows.append(
                {
                    "model_key": model,
                    "split": split,
                    "rows": len(values),
                    "dimension": values.shape[1],
                    "id_count_matches": len(values) == len(ids),
                    "all_finite": bool(np.isfinite(values).all()),
                    "min_l2_norm": float(norms.min()),
                    "max_l2_norm": float(norms.max()),
                    "mean_l2_norm": float(norms.mean()),
                }
            )
    pd.DataFrame(embedding_rows).to_csv("artifacts/embedding_audit.csv", index=False)

    split_root = Path("data/raw/polyvore_outfits/disjoint")
    split_files = sorted(split_root.glob("*.json")) + sorted(split_root.glob("*.txt"))
    packages = [
        "huggingface-hub",
        "numpy",
        "open-clip-torch",
        "pandas",
        "pillow",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
    ]
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": "historical_polyvore_d",
        "protocol_warning": "Packaged historical disjoint split has known cross-split item overlap; not leakage-free.",
        "primary_models": list(PRIMARY_MODEL_KEYS),
        "checkpoints": {
            key: {"checkpoint": spec.checkpoint, "revision": spec.revision}
            for key, spec in PRIMARY_ENCODERS.items()
        },
        "seed": 20260912,
        "logistic_c_grid": list(SHARED_C_GRID),
        "bootstrap_resamples": 2000,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_runtime": torch.version.cuda,
        "packages": {package: version(package) for package in packages},
        "dataset_file_sha256": {str(path.relative_to(split_root)): sha256(path) for path in split_files},
        "code_lock_sha256": sha256(Path("uv.lock")),
        "primary_result_sha256": sha256(Path("artifacts/primary_summary.csv")),
        "native_result_sha256": sha256(Path("artifacts/native_results.csv")),
    }
    Path("artifacts/reproducibility_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
