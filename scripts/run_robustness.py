"""Controlled learner-capacity and compatibility-label-efficiency analyses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from repbench.features.outfit_features import outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import LogisticCompatibility
from repbench.utils.seeds import seed_everything

SEED = 20260912
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def load_protocol_splits(cache: Path, dataset: PolyvoreDisjoint):
    source = {}
    for split in dataset.SPLITS:
        item_ids = json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8"))
        source[split] = (item_ids, np.load(cache / f"{split}.npy").astype(np.float32, copy=False))
    locations = {}
    for split, (item_ids, _) in source.items():
        for index, item_id in enumerate(item_ids):
            locations.setdefault(item_id, (split, index))
    remapped = {}
    for split in dataset.SPLITS:
        item_ids = sorted(dataset.item_ids(split))
        if source[split][0] == item_ids:
            remapped[split] = source[split]
            continue
        if any(item_id not in locations for item_id in item_ids):
            raise KeyError(f"Embedding cache is incomplete for {split}")
        remapped[split] = (
            item_ids,
            np.stack(
                [source[locations[item_id][0]][1][locations[item_id][1]] for item_id in item_ids]
            ),
        )
    return remapped


def nested_stratified_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(len(indices))]
        count = len(indices) if fraction == 1 else max(1, int(np.floor(len(indices) * fraction)))
        selected.extend(indices[:count])
    return np.sort(np.asarray(selected, dtype=np.int64))


def prepare_features(
    dataset: PolyvoreDisjoint,
    model_key: str,
    embedding_root: Path,
    dataset_key: str,
    pca_root: Path | None = None,
):
    cache = embedding_root / dataset_key / model_key
    split_data = load_protocol_splits(cache, dataset)
    if pca_root is None:
        pca = TrainOnlyPCA(256, seed=SEED)
        values = pca.fit_transform_splits(*(split_data[split][1] for split in dataset.SPLITS))
    else:
        pca = joblib.load(pca_root / model_key / "pca.joblib")
        values = tuple(pca.transform(split_data[split][1]) for split in dataset.SPLITS)
    embeddings = {
        split: dict(zip(split_data[split][0], values[index], strict=True))
        for index, split in enumerate(dataset.SPLITS)
    }
    cp = {split: dataset.compatibility(split) for split in dataset.SPLITS}
    x = {
        split: outfit_matrix([row.item_ids for row in cp[split]], embeddings[split])
        for split in dataset.SPLITS
    }
    y = {split: np.asarray([row.label for row in cp[split]]) for split in dataset.SPLITS}
    questions = dataset.fitb("test")
    fitb_x = outfit_matrix(
        [
            question.question_item_ids + (candidate,)
            for question in questions
            for candidate in question.candidate_item_ids
        ],
        embeddings["test"],
    )
    fitb_targets = np.asarray([question.correct_index for question in questions])
    return x, y, fitb_x, fitb_targets


def metrics(scores: np.ndarray, labels: np.ndarray, fitb_scores: np.ndarray, targets: np.ndarray):
    predicted = fitb_scores.reshape(len(targets), 4).argmax(axis=1)
    return {
        "cp_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "fitb_acc": float(np.mean(predicted == targets)),
    }, predicted


def fit_logistic(x, y, fitb_x, fraction: float):
    indices = nested_stratified_indices(y["train"], fraction, SEED)
    scaler = StandardScaler().fit(x["train"][indices])
    transformed = {split: scaler.transform(values) for split, values in x.items()}
    transformed_fitb = scaler.transform(fitb_x)
    started = time.perf_counter()
    model = LogisticCompatibility(seed=SEED).fit(
        transformed["train"][indices],
        y["train"][indices],
        transformed["valid"],
        y["valid"],
    )
    elapsed = time.perf_counter() - started
    return (
        model.predict_proba(transformed["test"]),
        model.predict_proba(transformed_fitb),
        elapsed,
        {"selected_c": model.best_c_, "n_train": len(indices)},
    )


def fit_mlp(x, y, fitb_x):
    scaler = StandardScaler().fit(x["train"])
    transformed = {split: scaler.transform(values) for split, values in x.items()}
    transformed_fitb = scaler.transform(fitb_x)
    started = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=SEED,
    ).fit(transformed["train"], y["train"])
    elapsed = time.perf_counter() - started
    return (
        model.predict_proba(transformed["test"])[:, 1],
        model.predict_proba(transformed_fitb)[:, 1],
        elapsed,
        {"epochs": model.n_iter_, "hidden_units": 128},
    )


def fit_xgboost(x, y, fitb_x):
    scaler = StandardScaler().fit(x["train"])
    transformed = {split: scaler.transform(values) for split, values in x.items()}
    transformed_fitb = scaler.transform(fitb_x)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=SEED,
        n_jobs=8,
    )
    started = time.perf_counter()
    model.fit(transformed["train"], y["train"], verbose=False)
    elapsed = time.perf_counter() - started
    return (
        model.predict_proba(transformed["test"])[:, 1],
        model.predict_proba(transformed_fitb)[:, 1],
        elapsed,
        {"trees": 300, "max_depth": 4, "device": model.get_params()["device"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--embedding-root", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--embedding-dataset-key", default="historical_polyvore_d")
    parser.add_argument("--pca-root", type=Path)
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=("low_data", "mlp", "xgboost"),
        default=["low_data", "mlp", "xgboost"],
    )
    args = parser.parse_args()
    seed_everything(SEED)
    dataset = PolyvoreDisjoint(args.root, args.protocol_dir, args.dataset_name)
    rows = []
    prediction_rows = []
    fitb_rows = []
    for model_key in PRIMARY_MODEL_KEYS:
        print(f"Preparing robustness features: {model_key}", flush=True)
        x, y, fitb_x, fitb_targets = prepare_features(
            dataset,
            model_key,
            args.embedding_root,
            args.embedding_dataset_key,
            args.pca_root,
        )
        jobs = [("logistic", fraction) for fraction in FRACTIONS] if "low_data" in args.jobs else []
        if "mlp" in args.jobs:
            jobs.append(("mlp", 1.0))
        if "xgboost" in args.jobs:
            jobs.append(("xgboost", 1.0))
        for learner, fraction in jobs:
            print(f"  {learner} fraction={fraction:g}", flush=True)
            if learner == "logistic":
                scores, candidate_scores, seconds, details = fit_logistic(
                    x, y, fitb_x, fraction
                )
            elif learner == "mlp":
                scores, candidate_scores, seconds, details = fit_mlp(x, y, fitb_x)
            else:
                scores, candidate_scores, seconds, details = fit_xgboost(x, y, fitb_x)
            result, fitb_predicted = metrics(
                scores, y["test"], candidate_scores, fitb_targets
            )
            run_key = f"{learner}_{fraction:g}"
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "representation": model_key,
                    "learner": learner,
                    "train_fraction": fraction,
                    "train_seconds": seconds,
                    **result,
                    **details,
                }
            )
            prediction_rows.extend(
                {
                    "dataset": args.dataset_name,
                    "representation": model_key,
                    "run_key": run_key,
                    "example_index": index,
                    "label": int(label),
                    "score": float(score),
                }
                for index, (label, score) in enumerate(zip(y["test"], scores, strict=True))
            )
            fitb_rows.extend(
                {
                    "dataset": args.dataset_name,
                    "representation": model_key,
                    "run_key": run_key,
                    "question_index": index,
                    "correct_index": int(correct),
                    "predicted_index": int(predicted),
                    "correct": bool(correct == predicted),
                }
                for index, (correct, predicted) in enumerate(
                    zip(fitb_targets, fitb_predicted, strict=True)
                )
            )
        pd.DataFrame(rows).to_csv(f"artifacts/{args.prefix}_robustness.partial.csv", index=False)
    pd.DataFrame(rows).to_csv(f"artifacts/{args.prefix}_robustness.csv", index=False)
    pd.DataFrame(prediction_rows).to_parquet(
        f"artifacts/{args.prefix}_robustness_cp_predictions.parquet", index=False
    )
    pd.DataFrame(fitb_rows).to_parquet(
        f"artifacts/{args.prefix}_robustness_fitb_predictions.parquet", index=False
    )
    Path(f"artifacts/{args.prefix}_robustness_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "fractions": FRACTIONS,
                "pca_components": 256,
                "label_efficiency_note": (
                    "PCA uses all unlabeled training-split items; compatibility labels are subset. "
                    "The outfit scaler is refit on each labeled subset."
                ),
                "mlp": "sklearn MLP 512-128-1, ReLU, alpha=1e-4, Adam, early stopping",
                "xgboost": "300 trees, depth 4, eta .05, subsample/colsample .8, hist",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
