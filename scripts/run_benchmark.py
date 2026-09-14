from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_disjoint
from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS
from repbench.eval.compatibility import cp_metrics
from repbench.features.outfit_features import outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import SHARED_C_GRID, LogisticCompatibility
from repbench.utils.seeds import seed_everything


def load_split(cache: Path, split: str) -> tuple[list[str], np.ndarray]:
    ids = json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8"))
    values = np.load(cache / f"{split}.npy")
    if len(ids) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid embedding cache for {cache.name}/{split}")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError(f"Expected L2-normalized cache for {cache.name}/{split}")
    return ids, values.astype(np.float32, copy=False)


def extraction_metrics(cache: Path) -> dict:
    metadata = [
        json.loads((cache / f"{split}_metadata.json").read_text(encoding="utf-8"))
        for split in PolyvoreDisjoint.SPLITS
    ]
    elapsed = sum(row["elapsed_seconds"] for row in metadata)
    image_count = sum(
        len(json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8")))
        for split in PolyvoreDisjoint.SPLITS
    )
    return {
        "extraction_seconds": elapsed,
        "extract_images_per_second": image_count / elapsed,
        "peak_vram_mb": max((row.get("peak_vram_mb") or np.nan) for row in metadata),
        "embedding_storage_mb": sum(
            (cache / f"{split}.npy").stat().st_size for split in PolyvoreDisjoint.SPLITS
        )
        / 2**20,
    }


def load_protocol_splits(
    cache: Path, dataset: PolyvoreDisjoint
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Remap the immutable historical item cache to any protocol over the same images."""
    source = {split: load_split(cache, split) for split in dataset.SPLITS}
    locations: dict[str, tuple[str, int]] = {}
    for split, (item_ids, values) in source.items():
        for index, item_id in enumerate(item_ids):
            if item_id in locations:
                previous_split, previous_index = locations[item_id]
                previous = source[previous_split][1][previous_index]
                cosine = float(previous @ values[index]) / (
                    float(np.linalg.norm(previous) * np.linalg.norm(values[index])) + 1e-12
                )
                if cosine < 0.99999 or np.max(np.abs(previous - values[index])) > 1e-3:
                    raise ValueError(f"Embedding mismatch for repeated item {item_id}")
            else:
                locations[item_id] = (split, index)
    remapped = {}
    for split in dataset.SPLITS:
        item_ids = sorted(dataset.item_ids(split))
        missing = [item_id for item_id in item_ids if item_id not in locations]
        if missing:
            raise KeyError(f"Historical cache lacks {len(missing)} {split} items")
        values = np.stack(
            [source[locations[item_id][0]][1][locations[item_id][1]] for item_id in item_ids]
        )
        remapped[split] = (item_ids, values)
    return remapped


def run_representation(
    dataset: PolyvoreDisjoint,
    model_key: str,
    seed: int,
    track: str,
    prefix: str,
    pca_components: int = 256,
    embedding_root: Path = Path("data/embeddings"),
    embedding_dataset_key: str = "historical_polyvore_d",
) -> dict:
    cache = embedding_root / embedding_dataset_key / model_key
    split_data = load_protocol_splits(cache, dataset)
    raw_dim = split_data["train"][1].shape[1]
    if raw_dim != PRIMARY_ENCODERS[model_key].embedding_dim:
        raise ValueError(f"Unexpected native dimension for {model_key}: {raw_dim}")

    pca = None
    if track == "pca":
        pca = TrainOnlyPCA(pca_components, seed=seed)
        values = pca.fit_transform_splits(
            *(split_data[split][1] for split in dataset.SPLITS)
        )
    else:
        values = tuple(split_data[split][1] for split in dataset.SPLITS)
    embeddings = {
        split: dict(zip(split_data[split][0], values[index], strict=True))
        for index, split in enumerate(dataset.SPLITS)
    }

    cp_data = {split: dataset.compatibility(split) for split in dataset.SPLITS}
    x = {
        split: outfit_matrix([example.item_ids for example in cp_data[split]], embeddings[split])
        for split in dataset.SPLITS
    }
    y = {
        split: np.asarray([example.label for example in cp_data[split]])
        for split in dataset.SPLITS
    }
    outfit_scaler = StandardScaler().fit(x["train"])
    x = {split: outfit_scaler.transform(features) for split, features in x.items()}
    started = time.perf_counter()
    classifier = LogisticCompatibility(c_grid=SHARED_C_GRID, seed=seed).fit(
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
    fitb_matrix = outfit_scaler.transform(outfit_matrix(fitb_outfits, embeddings["test"]))
    fitb_scores = classifier.predict_proba(fitb_matrix).reshape(len(fitb_questions), 4)
    fitb_predictions = np.argmax(fitb_scores, axis=1)
    fitb_correct = fitb_predictions == np.asarray([q.correct_index for q in fitb_questions])

    timing_repeats = 100
    timing_sample = fitb_matrix[: min(1024, len(fitb_matrix))]
    timing_started = time.perf_counter()
    for _ in range(timing_repeats):
        classifier.predict_proba(timing_sample)
    scoring_ms = (
        (time.perf_counter() - timing_started)
        * 1000
        / (timing_repeats * len(timing_sample))
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "model_key": model_key,
            "example_index": np.arange(len(test_scores)),
            "label": y["test"],
            "score": test_scores,
        }
    ).to_parquet(artifacts / f"{prefix}_cp_predictions_{model_key}.parquet", index=False)
    pd.DataFrame(
        {
            "model_key": model_key,
            "question_index": np.arange(len(fitb_questions)),
            "correct_index": [q.correct_index for q in fitb_questions],
            "predicted_index": fitb_predictions,
            "correct": fitb_correct,
        }
    ).to_parquet(artifacts / f"{prefix}_fitb_predictions_{model_key}.parquet", index=False)
    pd.DataFrame(
        {
            "model_key": model_key,
            "question_index": np.repeat(np.arange(len(fitb_questions)), 4),
            "candidate_index": np.tile(np.arange(4), len(fitb_questions)),
            "score": fitb_scores.ravel(),
        }
    ).to_parquet(
        artifacts / f"{prefix}_fitb_candidate_scores_{model_key}.parquet", index=False
    )
    pd.DataFrame(classifier.validation_curve_).assign(model_key=model_key).to_csv(
        artifacts / f"{prefix}_validation_curve_{model_key}.csv", index=False
    )
    if pca is not None:
        joblib.dump(pca, artifacts / f"pca_{prefix}_{model_key}.joblib")
    joblib.dump(outfit_scaler, artifacts / f"outfit_scaler_{prefix}_{model_key}.joblib")

    metrics = extraction_metrics(cache)
    spec = PRIMARY_ENCODERS[model_key]
    return {
        "dataset": dataset.dataset_name,
        "track": f"pca{pca_components}" if pca is not None else "native",
        "representation": model_key,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "family": spec.family,
        "native_embedding_dim": raw_dim,
        "outfit_feature_dim": x["train"].shape[1],
        "outfit_standardization": True,
        "cp_auc": cp["roc_auc"],
        "pr_auc": cp["pr_auc"],
        "fitb_acc": float(fitb_correct.mean()),
        "validation_auc": classifier.validation_auc_,
        "best_c": classifier.best_c_,
        "c_at_upper_boundary": classifier.best_c_ == max(SHARED_C_GRID),
        "pca_explained_variance": pca.explained_variance_ratio if pca else np.nan,
        "train_seconds": train_seconds,
        "single_outfit_scoring_ms": scoring_ms,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODEL_KEYS))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--track", choices=("pca", "native"), default="pca")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--dataset-name", default="historical_polyvore_d")
    parser.add_argument("--prefix")
    parser.add_argument("--allow-known-overlap", action="store_true")
    parser.add_argument("--embedding-root", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--embedding-dataset-key", default="historical_polyvore_d")
    args = parser.parse_args()
    if any(model not in PRIMARY_MODEL_KEYS for model in args.models):
        raise SystemExit("Primary benchmark accepts only the locked seven-model roster")
    seed_everything(args.seed)
    dataset = PolyvoreDisjoint(args.root, args.protocol_dir, args.dataset_name)
    audit = inspect_disjoint(dataset)
    if not audit["passes_item_disjointness"] and not args.allow_known_overlap:
        raise SystemExit(
            "Refusing benchmark: historical packaged split contains known item overlap. "
            "Pass --allow-known-overlap only for the authorized historical protocol."
        )
    prefix = args.prefix or ("primary" if args.track == "pca" else "native")
    results = []
    for model_key in args.models:
        print(f"Running {prefix} track: {model_key}", flush=True)
        result = run_representation(
            dataset,
            model_key,
            args.seed,
            args.track,
            prefix,
            args.pca_components,
            args.embedding_root,
            args.embedding_dataset_key,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(f"artifacts/{prefix}_results.partial.csv", index=False)
    frame = pd.DataFrame(results)
    frame.to_csv(f"artifacts/{prefix}_results.csv", index=False)
    for task in ("cp_predictions", "fitb_predictions", "fitb_candidate_scores"):
        files = [Path("artifacts") / f"{prefix}_{task}_{model}.parquet" for model in args.models]
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).to_parquet(
            Path("artifacts") / f"{prefix}_{task}.parquet", index=False
        )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
