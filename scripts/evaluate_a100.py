"""External-only A100 evaluation for frozen Polyvore- and IQON-trained scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from repbench.features.outfit_features import outfit_matrix


def score(classifier, features: np.ndarray) -> np.ndarray:
    values = classifier.predict_proba(features)
    return values[:, 1] if values.ndim == 2 else values


def load_items(cache: Path) -> dict[str, np.ndarray]:
    ids = json.loads((cache / "all_item_ids.json").read_text())
    values = np.load(cache / "all.npy")
    if values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError(f"Invalid A100 embedding cache: {cache}")
    return dict(zip(ids, values, strict=True))


def evaluate_task(rows, embeddings, pca, scaler, classifier, task: str, model_key: str, source: str):
    transformed_values = pca.transform(np.stack(list(embeddings.values())))
    transformed = dict(zip(embeddings, transformed_values, strict=True))
    outfits = [tuple(row["question"] + [candidate]) for row in rows for candidate in row["answers"]]
    candidate_scores = score(classifier, scaler.transform(outfit_matrix(outfits, transformed))).reshape(len(rows), 5)
    if not np.isfinite(candidate_scores).all() or candidate_scores.shape != (100, 5):
        raise ValueError(f"Invalid A100 scores for {source}/{model_key}/{task}")
    predictions = np.argmax(candidate_scores, axis=1)
    records = []
    for index, row in enumerate(rows):
        record = {
            "training_source": source,
            "model_key": model_key,
            "task": task,
            "question_num": row["question_num"],
            "dimension": row.get("dimension"),
            "prediction_index": int(predictions[index]),
            "archive_gt_index": row["archive_gt_index"],
            "archive_gt_correct": bool(predictions[index] == row["archive_gt_index"]),
        }
        if task == "LAT":
            record.update(
                {
                    "majority_gt_index": row["majority_gt_index"],
                    "majority_correct": bool(predictions[index] == row["majority_gt_index"]),
                    "human_agreement": float(row["gt_distribution"][predictions[index]]),
                }
            )
        records.append(record)
    candidates = pd.DataFrame(
        {
            "training_source": source,
            "model_key": model_key,
            "task": task,
            "question_num": np.repeat([row["question_num"] for row in rows], 5),
            "candidate_index": np.tile(np.arange(5), len(rows)),
            "candidate_item_id": [item for row in rows for item in row["answers"]],
            "score": candidate_scores.ravel(),
        }
    )
    return pd.DataFrame(records), candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("data/protocols/a100"))
    parser.add_argument("--embedding-root", type=Path, default=Path(".venv/a100/embeddings"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final/a100"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite A100 results: {args.output}")
    args.output.mkdir(parents=True)
    tasks = {task: json.loads((args.protocol / f"{task}.json").read_text()) for task in ("aat", "lat")}
    all_predictions, all_candidates, summaries = [], [], []
    for source in ("polyvore_d_clean", "iqon3000_clean"):
        for model_key in PRIMARY_MODEL_KEYS:
            raw = load_items(args.embedding_root / model_key)
            if source == "polyvore_d_clean":
                root = Path("artifacts/final/polyvore_d_clean") / model_key
                pca = joblib.load(f"artifacts/pca_clean_primary_{model_key}.joblib")
                scaler = joblib.load(f"artifacts/outfit_scaler_clean_primary_{model_key}.joblib")
                classifier = joblib.load(root / "classifier.joblib")
            else:
                root = Path("artifacts/final/iqon3000_clean") / model_key
                pca = joblib.load(root / "pca.joblib")
                scaler = joblib.load(root / "outfit_scaler.joblib")
                classifier = joblib.load(root / "classifier.joblib")
            for task, rows in tasks.items():
                predictions, candidates = evaluate_task(rows, raw, pca, scaler, classifier, task.upper(), model_key, source)
                all_predictions.append(predictions)
                all_candidates.append(candidates)
                if task == "lat":
                    summaries.append(
                        {
                            "training_source": source,
                            "model_key": model_key,
                            "task": "LAT",
                            "majority_accuracy": predictions.majority_correct.mean(),
                            "mLAT": predictions.human_agreement.mean(),
                            "archive_gt_accuracy": predictions.archive_gt_correct.mean(),
                        }
                    )
                else:
                    summaries.append(
                        {
                            "training_source": source,
                            "model_key": model_key,
                            "task": "AAT",
                            "overall_accuracy": predictions.archive_gt_correct.mean(),
                            **{
                                f"{dimension.lower()}_accuracy": predictions.loc[predictions.dimension == dimension, "archive_gt_correct"].mean()
                                for dimension in ("Color", "Style", "Occasion", "Season", "Material", "Balance")
                            },
                        }
                    )
    pd.concat(all_predictions, ignore_index=True).to_parquet(args.output / "predictions.parquet", index=False)
    pd.concat(all_candidates, ignore_index=True).to_parquet(args.output / "candidate_scores.parquet", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
