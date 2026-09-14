from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_disjoint
from repbench.eval.compatibility import cp_metrics
from repbench.features.outfit_features import outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import LogisticCompatibility
from repbench.utils.seeds import seed_everything


def load_split(cache: Path, split: str) -> tuple[list[str], np.ndarray]:
    ids = json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8"))
    values = np.load(cache / f"{split}.npy")
    if len(ids) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid embedding cache for {cache.name}/{split}")
    return ids, values.astype(np.float32, copy=False)


def run_representation(dataset: PolyvoreDisjoint, model_key: str, seed: int) -> dict:
    cache = Path("data/embeddings/historical_polyvore_d") / model_key
    split_data = {split: load_split(cache, split) for split in dataset.SPLITS}
    pca = TrainOnlyPCA(256, seed=seed)
    transformed = pca.fit_transform_splits(*(split_data[split][1] for split in dataset.SPLITS))
    embeddings = {
        split: dict(zip(split_data[split][0], transformed[index]))
        for index, split in enumerate(dataset.SPLITS)
    }

    cp_data = {split: dataset.compatibility(split) for split in dataset.SPLITS}
    x = {
        split: outfit_matrix([example.item_ids for example in cp_data[split]], embeddings[split])
        for split in dataset.SPLITS
    }
    y = {
        split: np.asarray([example.label for example in cp_data[split]]) for split in dataset.SPLITS
    }
    started = time.perf_counter()
    classifier = LogisticCompatibility(seed=seed).fit(
        x["train"], y["train"], x["valid"], y["valid"]
    )
    train_seconds = time.perf_counter() - started
    test_scores = classifier.predict_proba(x["test"])
    cp = cp_metrics(y["test"], test_scores)

    fitb_questions = dataset.fitb("test")
    candidate_counts = {len(question.candidate_item_ids) for question in fitb_questions}
    if candidate_counts != {4}:
        raise ValueError(f"Expected exactly four FITB candidates, found {candidate_counts}")
    fitb_outfits = [
        question.question_item_ids + (candidate,)
        for question in fitb_questions
        for candidate in question.candidate_item_ids
    ]
    fitb_scores = classifier.predict_proba(outfit_matrix(fitb_outfits, embeddings["test"]))
    fitb_scores = fitb_scores.reshape(len(fitb_questions), 4)
    fitb_predictions = np.argmax(fitb_scores, axis=1)
    fitb_correct = fitb_predictions == np.asarray([q.correct_index for q in fitb_questions])
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "model_key": model_key,
            "example_index": np.arange(len(test_scores)),
            "label": y["test"],
            "score": test_scores,
        }
    ).to_parquet(artifacts / f"pilot_cp_predictions_{model_key}.parquet", index=False)
    pd.DataFrame(
        {
            "model_key": model_key,
            "question_index": np.arange(len(fitb_questions)),
            "correct_index": [q.correct_index for q in fitb_questions],
            "predicted_index": fitb_predictions,
            "correct": fitb_correct,
        }
    ).to_parquet(artifacts / f"pilot_fitb_predictions_{model_key}.parquet", index=False)
    joblib.dump(pca, artifacts / f"pca_{model_key}.joblib")
    return {
        "dataset": "historical_polyvore_d",
        "representation": model_key,
        "cp_auc": cp["roc_auc"],
        "pr_auc": cp["pr_auc"],
        "fitb_acc": float(fitb_correct.mean()),
        "validation_auc": classifier.validation_auc_,
        "best_c": classifier.best_c_,
        "pca_explained_variance": pca.explained_variance_ratio,
        "train_seconds": train_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument(
        "--models", nargs="+", default=["resnet50", "dinov2_vitb14", "clip_vitl14", "fashionclip2"]
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--allow-known-overlap", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    dataset = PolyvoreDisjoint(args.root)
    audit = inspect_disjoint(dataset)
    if not audit["passes_item_disjointness"] and not args.allow_known_overlap:
        raise SystemExit(
            "Refusing pilot: historical packaged split violates the disjointness invariant. "
            "See artifacts/data_integrity_report.json."
        )
    results = [run_representation(dataset, model_key, args.seed) for model_key in args.models]
    frame = pd.DataFrame(results)
    frame.to_csv("artifacts/pilot_results.csv", index=False)
    for task in ("cp", "fitb"):
        files = [
            Path("artifacts") / f"pilot_{task}_predictions_{model}.parquet" for model in args.models
        ]
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).to_parquet(
            Path("artifacts") / f"pilot_{task}_predictions.parquet", index=False
        )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
