"""Run the locked seven-representation IQON3000-Clean benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS
from repbench.eval.calibration import calibration_metrics, reliability_bins
from repbench.eval.compatibility import cp_metrics
from repbench.features.outfit_features import outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import SHARED_C_GRID, LogisticCompatibility
from repbench.utils.seeds import seed_everything

SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embeddings(cache: Path, dataset: PolyvoreDisjoint) -> dict[str, tuple[list[str], np.ndarray]]:
    result = {}
    spec = PRIMARY_ENCODERS[cache.name]
    for split in SPLITS:
        ids = json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8"))
        values = np.load(cache / f"{split}.npy")
        expected_ids = sorted(dataset.item_ids(split))
        if ids != expected_ids or values.shape != (len(ids), spec.embedding_dim):
            raise ValueError(f"Embedding identity/shape mismatch for {cache.name}/{split}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite embedding for {cache.name}/{split}")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
            raise ValueError(f"Non-normalized embedding for {cache.name}/{split}")
        result[split] = (ids, values.astype(np.float32, copy=False))
    return result


def identities(protocol: Path, split: str, cp_count: int, fitb_rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    rows = json.loads((protocol / f"{split}.json").read_text(encoding="utf-8"))
    users = json.loads((protocol / f"group_ids_{split}.json").read_text(encoding="utf-8"))
    ordered = sorted(rows, key=lambda row: str(row["set_id"]))
    set_ids = [str(row["set_id"]) for row in ordered]
    if cp_count != 2 * len(set_ids):
        raise ValueError("CP rows no longer align with positive-outfit identities")
    cp_sets = set_ids + set_ids
    cp_users = [users[value] for value in cp_sets]
    fitb_sets = [str(row["question"][0]).split("_", 1)[0] for row in fitb_rows]
    return cp_sets, cp_users, [users[value] for value in fitb_sets]


def run_model(args, dataset: PolyvoreDisjoint, model_key: str) -> dict:
    output = args.output_root / model_key
    complete = output / "COMPLETE.json"
    if output.exists():
        if args.skip_valid_existing and complete.is_file():
            marker = json.loads(complete.read_text(encoding="utf-8"))
            if all((output / name).is_file() and sha256(output / name) == digest for name, digest in marker["artifacts"].items()):
                print(f"{model_key}: reusing complete validated run", flush=True)
                return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        raise FileExistsError(f"Refusing to overwrite incomplete/existing run: {output}")
    output.mkdir(parents=True)
    started = time.time()
    cache = args.embedding_root / args.dataset_key / model_key
    split_data = load_embeddings(cache, dataset)
    pca = TrainOnlyPCA(256, seed=args.seed)
    pca_started = time.perf_counter()
    transformed = pca.fit_transform_splits(*(split_data[split][1] for split in SPLITS))
    pca_seconds = time.perf_counter() - pca_started
    embeddings = {
        split: dict(zip(split_data[split][0], transformed[index], strict=True))
        for index, split in enumerate(SPLITS)
    }
    del transformed
    cp_data = {split: dataset.compatibility(split) for split in SPLITS}
    x = {
        split: outfit_matrix([row.item_ids for row in cp_data[split]], embeddings[split])
        for split in SPLITS
    }
    y = {split: np.asarray([row.label for row in cp_data[split]], dtype=np.int8) for split in SPLITS}
    scaler = StandardScaler().fit(x["train"])
    x = {split: scaler.transform(values).astype(np.float32) for split, values in x.items()}
    train_started = time.perf_counter()
    classifier = LogisticCompatibility(SHARED_C_GRID, args.seed).fit(x["train"], y["train"], x["valid"], y["valid"])
    train_seconds = time.perf_counter() - train_started

    metrics = {
        "dataset": args.dataset_key,
        "representation": model_key,
        "checkpoint": PRIMARY_ENCODERS[model_key].checkpoint,
        "revision": PRIMARY_ENCODERS[model_key].revision,
        "native_embedding_dim": PRIMARY_ENCODERS[model_key].embedding_dim,
        "pca_explained_variance": pca.explained_variance_ratio,
        "pca_seconds": pca_seconds,
        "classifier_train_seconds": train_seconds,
        "selected_c": classifier.best_c_,
        "validation_auc": classifier.validation_auc_,
        "splits": {},
    }
    reliability = []
    for split in SPLITS:
        probabilities = classifier.predict_proba(x[split])
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Invalid CP probabilities for {model_key}/{split}")
        if np.ptp(probabilities) < 1e-8:
            raise ValueError(f"Constant CP probabilities for {model_key}/{split}")
        cp_sets, cp_users, _ = identities(args.protocol_root, split, len(y[split]), [])
        pd.DataFrame(
            {
                "dataset": args.dataset_key,
                "model_key": model_key,
                "split": split,
                "example_id": [f"{split}:cp:{i}" for i in range(len(y[split]))],
                "source_set_id": cp_sets,
                "group_user_id": cp_users,
                "label": y[split],
                "probability": probabilities,
            }
        ).to_parquet(output / f"cp_predictions_{split}.parquet", index=False)
        split_metrics = {
            **cp_metrics(y[split], probabilities),
            "accuracy": float(accuracy_score(y[split], probabilities >= 0.5)),
            "balanced_accuracy": float(balanced_accuracy_score(y[split], probabilities >= 0.5)),
            **calibration_metrics(y[split], probabilities),
        }
        metrics["splits"].setdefault(split, {}).update({f"cp_{key}": value for key, value in split_metrics.items()})
        reliability.extend(
            {"model_key": model_key, "split": split, **row}
            for row in reliability_bins(y[split], probabilities, 15)
        )

        raw_fitb = json.loads((args.protocol_root / f"fill_in_blank_{split}.json").read_text(encoding="utf-8"))
        questions = dataset.fitb(split)
        fitb_outfits = [
            question.question_item_ids + (candidate,)
            for question in questions
            for candidate in question.candidate_item_ids
        ]
        matrix = scaler.transform(outfit_matrix(fitb_outfits, embeddings[split]))
        scores = classifier.predict_proba(matrix).reshape(len(questions), 4)
        predicted = np.argmax(scores, axis=1)
        correct_index = np.asarray([row.correct_index for row in questions])
        correct = predicted == correct_index
        if scores.shape != (len(questions), 4) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid FITB score matrix for {model_key}/{split}")
        _, _, fitb_users = identities(args.protocol_root, split, len(y[split]), raw_fitb)
        fitb_sets = [str(row["question"][0]).split("_", 1)[0] for row in raw_fitb]
        pd.DataFrame(
            {
                "dataset": args.dataset_key,
                "model_key": model_key,
                "split": split,
                "question_id": [f"{split}:fitb:{i}" for i in range(len(questions))],
                "source_set_id": fitb_sets,
                "group_user_id": fitb_users,
                "correct_index": correct_index,
                "predicted_index": predicted,
                "correct": correct,
            }
        ).to_parquet(output / f"fitb_predictions_{split}.parquet", index=False)
        pd.DataFrame(
            {
                "question_id": np.repeat([f"{split}:fitb:{i}" for i in range(len(questions))], 4),
                "candidate_index": np.tile(np.arange(4), len(questions)),
                "candidate_item_id": [candidate for row in questions for candidate in row.candidate_item_ids],
                "score": scores.ravel(),
            }
        ).to_parquet(output / f"fitb_candidate_scores_{split}.parquet", index=False)
        metrics["splits"][split]["fitb_accuracy"] = float(correct.mean())
        metrics["splits"][split]["fitb_chance"] = 0.25

    pd.DataFrame(classifier.validation_curve_).to_csv(output / "validation_curve.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "reliability_bins.csv", index=False)
    joblib.dump(pca, output / "pca.joblib")
    joblib.dump(scaler, output / "outfit_scaler.joblib")
    joblib.dump(classifier, output / "classifier.joblib")
    metrics["elapsed_seconds"] = time.time() - started
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "run_id": f"iqon-primary-{model_key}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset_key,
        "protocol_manifest_sha256": sha256(args.protocol_root / "manifest.json"),
        "final_protocol_hash": json.loads(Path("artifacts/final_experimental_protocol_hash.json").read_text())["combined_sha256"],
        "seed": args.seed,
        "model_key": model_key,
        "revision": PRIMARY_ENCODERS[model_key].revision,
        "c_grid": list(SHARED_C_GRID),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifact_names = [path.name for path in output.iterdir() if path.is_file()]
    marker = {"artifacts": {name: sha256(output / name) for name in sorted(artifact_names)}}
    complete.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(".venv/iqon3000_clean"))
    parser.add_argument("--protocol-root", type=Path, default=Path("data/protocols/iqon3000_clean"))
    parser.add_argument("--embedding-root", type=Path, default=Path(".venv/iqon3000_clean/embeddings"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/final/iqon3000_clean"))
    parser.add_argument("--dataset-key", default="iqon3000_clean")
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODEL_KEYS))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--skip-valid-existing", action="store_true")
    args = parser.parse_args()
    if any(model not in PRIMARY_MODEL_KEYS for model in args.models):
        raise SystemExit("Only the locked seven representations are accepted")
    seed_everything(args.seed)
    dataset = PolyvoreDisjoint(args.dataset_root, args.protocol_root, args.dataset_key)
    results = []
    for model in args.models:
        print(f"running locked IQON primary: {model}", flush=True)
        results.append(run_model(args, dataset, model))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "representation": row["representation"],
            "selected_c": row["selected_c"],
            "pca_explained_variance": row["pca_explained_variance"],
            **{key: value for key, value in row["splits"]["test"].items()},
        }
        for row in results
    ]
    pd.DataFrame(rows).to_csv(args.output_root / "primary_results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
