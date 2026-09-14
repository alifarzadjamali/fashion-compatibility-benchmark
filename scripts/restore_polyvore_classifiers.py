"""Reconstruct deterministic Polyvore-D-Clean LR scorers for external A100 only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_benchmark import load_protocol_splits
from sklearn.linear_model import LogisticRegression

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from repbench.features.outfit_features import outfit_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--protocol", type=Path, default=Path("data/protocols/polyvore_d_clean"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final/polyvore_d_clean"))
    args = parser.parse_args()
    dataset = PolyvoreDisjoint(args.raw_root, args.protocol, "polyvore_d_clean")
    selected = pd.read_csv("artifacts/clean_primary_results.csv").set_index("representation")
    old_predictions = pd.read_parquet("artifacts/clean_primary_cp_predictions.parquet")
    for model_key in PRIMARY_MODEL_KEYS:
        destination = args.output / model_key
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite restored scorer: {destination}")
        destination.mkdir(parents=True)
        split_data = load_protocol_splits(Path("data/embeddings/historical_polyvore_d") / model_key, dataset)
        pca_path = Path(f"artifacts/pca_clean_primary_{model_key}.joblib")
        scaler_path = Path(f"artifacts/outfit_scaler_clean_primary_{model_key}.joblib")
        pca = joblib.load(pca_path)
        scaler = joblib.load(scaler_path)
        embeddings = {
            split: dict(zip(split_data[split][0], pca.transform(split_data[split][1]), strict=True))
            for split in dataset.SPLITS
        }
        cp = {split: dataset.compatibility(split) for split in ("train", "test")}
        x_train = scaler.transform(outfit_matrix([row.item_ids for row in cp["train"]], embeddings["train"]))
        y_train = np.asarray([row.label for row in cp["train"]])
        model = LogisticRegression(
            C=float(selected.loc[model_key, "best_c"]),
            max_iter=2000,
            solver="lbfgs",
            random_state=20260912,
        ).fit(x_train, y_train)
        x_test = scaler.transform(outfit_matrix([row.item_ids for row in cp["test"]], embeddings["test"]))
        recreated = model.predict_proba(x_test)[:, 1]
        expected = old_predictions[old_predictions.model_key == model_key].sort_values("example_index").score.to_numpy()
        maximum_error = float(np.max(np.abs(recreated - expected)))
        if maximum_error > 1e-8:
            raise ValueError(f"Restored {model_key} predictions differ by {maximum_error}")
        joblib.dump(model, destination / "classifier.joblib")
        metadata = {
            "model_key": model_key,
            "selected_c": float(selected.loc[model_key, "best_c"]),
            "restored_test_prediction_max_abs_error": maximum_error,
            "pca_path": str(pca_path),
            "pca_sha256": hashlib.sha256(pca_path.read_bytes()).hexdigest(),
            "outfit_scaler_path": str(scaler_path),
            "outfit_scaler_sha256": hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
