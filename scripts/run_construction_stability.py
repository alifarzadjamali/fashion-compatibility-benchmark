"""Evaluate deterministic question-construction seeds with frozen item partitions/PCA."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_benchmark import load_protocol_splits
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from repbench.features.outfit_features import outfit_matrix
from repbench.models.logistic import SHARED_C_GRID, LogisticCompatibility

SPLITS = ("train", "valid", "test")


def load_projected(args, model_key: str) -> dict[str, dict[str, np.ndarray]]:
    cache = args.embedding_root / args.embedding_dataset_key / model_key
    nested = args.primary_model_root / model_key / "pca.joblib"
    flat = args.primary_model_root / f"pca_{args.primary_artifact_prefix}_{model_key}.joblib"
    pca_path = nested if nested.is_file() else flat
    if not pca_path.is_file():
        raise FileNotFoundError(f"No frozen primary PCA found for {model_key}: {nested} or {flat}")
    pca = joblib.load(pca_path)
    reference = PolyvoreDisjoint(
        args.dataset_root, args.protocol_root / f"seed_{args.seeds[0]}", args.dataset_name
    )
    split_data = load_protocol_splits(cache, reference)
    projected = {}
    for split in SPLITS:
        ids, values = split_data[split]
        transformed = pca.transform(values)
        if not np.isfinite(transformed).all():
            raise ValueError(f"Non-finite projected embeddings: {model_key}/{split}")
        projected[split] = dict(zip(ids, transformed, strict=True))
    return projected


def run_seed(args, model_key: str, projected, seed: int) -> dict:
    protocol = args.protocol_root / f"seed_{seed}"
    dataset = PolyvoreDisjoint(args.dataset_root, protocol, args.dataset_name)
    examples = {split: dataset.compatibility(split) for split in SPLITS}
    x = {
        split: outfit_matrix([row.item_ids for row in examples[split]], projected[split])
        for split in SPLITS
    }
    y = {
        split: np.asarray([row.label for row in examples[split]], dtype=np.int8)
        for split in SPLITS
    }
    scaler = StandardScaler().fit(x["train"])
    scaled = {split: scaler.transform(x[split]).astype(np.float32) for split in SPLITS}
    classifier = LogisticCompatibility(SHARED_C_GRID, seed).fit(
        scaled["train"], y["train"], scaled["valid"], y["valid"]
    )
    probability = classifier.predict_proba(scaled["test"])
    questions = dataset.fitb("test")
    candidate_outfits = [
        question.question_item_ids + (candidate,)
        for question in questions
        for candidate in question.candidate_item_ids
    ]
    candidate_features = scaler.transform(outfit_matrix(candidate_outfits, projected["test"]))
    scores = classifier.predict_proba(candidate_features).reshape(len(questions), 4)
    targets = np.asarray([question.correct_index for question in questions])
    correct = np.argmax(scores, axis=1) == targets
    destination = args.output / model_key / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "example_index": np.arange(len(probability)),
            "label": y["test"],
            "probability": probability,
        }
    ).to_parquet(destination / "cp_predictions_test.parquet", index=False)
    pd.DataFrame(
        {
            "question_index": np.arange(len(questions)),
            "correct_index": targets,
            "predicted_index": np.argmax(scores, axis=1),
            "correct": correct,
        }
    ).to_parquet(destination / "fitb_predictions_test.parquet", index=False)
    return {
        "dataset": args.dataset_name,
        "model_key": model_key,
        "construction_seed": seed,
        "selected_c": classifier.best_c_,
        "validation_auc": classifier.validation_auc_,
        "cp_auc": roc_auc_score(y["test"], probability),
        "pr_auc": average_precision_score(y["test"], probability),
        "fitb_accuracy": float(correct.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-dataset-key", required=True)
    parser.add_argument("--primary-model-root", type=Path, required=True)
    parser.add_argument("--primary-artifact-prefix", default="clean_primary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODEL_KEYS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if any(model not in PRIMARY_MODEL_KEYS for model in args.models):
        raise SystemExit("Unknown or non-primary model key")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model_key in args.models:
        projected = load_projected(args, model_key)
        for seed in args.seeds:
            print(f"{args.dataset_name}/{model_key}/construction={seed}", flush=True)
            rows.append(run_seed(args, model_key, projected, seed))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_runs.csv", index=False)
    summary = frame.groupby("model_key")[["cp_auc", "pr_auc", "fitb_accuracy"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.to_csv(args.output / "summary.csv")


if __name__ == "__main__":
    main()
