"""External A100 evaluation of source-trained task-specific baseline checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_outfit_transformer_baseline import encode_examples, predict

from repbench.data.polyvore import CompatibilityExample
from repbench.models.outfit_transformer import OutfitTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("data/protocols/a100"))
    parser.add_argument("--embedding-cache", type=Path, default=Path(".venv/a100/embeddings/resnet50"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/final/baseline"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final/a100_outfit_transformer"))
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional common seed subset; otherwise discover completed source seeds")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite A100 baseline results: {args.output}")
    args.output.mkdir(parents=True)
    item_ids = json.loads((args.embedding_cache / "all_item_ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(args.embedding_cache / "all.npy", mmap_mode="r")
    if embeddings.shape != (len(item_ids), 2048) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid A100 ResNet50 cache")
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, candidate_rows, summary = [], [], []
    for source in ("polyvore_d_clean", "iqon3000_clean"):
        available_seeds = sorted(
            int(path.name.removeprefix("seed_"))
            for path in (args.baseline_root / source).glob("seed_*")
            if (path / "best_state.pt").is_file()
        )
        source_seeds = args.seeds if args.seeds is not None else available_seeds
        if not source_seeds or not set(source_seeds).issubset(available_seeds):
            raise ValueError(f"Requested baseline seeds are unavailable for {source}: {source_seeds}")
        for seed in source_seeds:
            checkpoint = args.baseline_root / source / f"seed_{seed}" / "best_state.pt"
            model = OutfitTransformer().to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            for task in ("lat", "aat"):
                rows = json.loads((args.protocol / f"{task}.json").read_text(encoding="utf-8"))
                examples = [
                    CompatibilityExample(0, tuple(row["question"] + [candidate]))
                    for row in rows
                    for candidate in row["answers"]
                ]
                encoded = encode_examples(examples, item_index)
                scores = predict(model, *encoded, embeddings, args.batch_size, device).reshape(100, 5)
                selected = np.argmax(scores, axis=1)
                task_predictions = []
                for index, row in enumerate(rows):
                    record = {
                        "training_source": source,
                        "seed": seed,
                        "task": task.upper(),
                        "question_num": row["question_num"],
                        "dimension": row.get("dimension"),
                        "prediction_index": int(selected[index]),
                        "archive_gt_correct": bool(selected[index] == row["archive_gt_index"]),
                    }
                    if task == "lat":
                        record["majority_correct"] = bool(selected[index] == row["majority_gt_index"])
                        record["human_agreement"] = float(row["gt_distribution"][selected[index]])
                    task_predictions.append(record)
                    for candidate_index, candidate in enumerate(row["answers"]):
                        candidate_rows.append(
                            {"training_source": source, "seed": seed, "task": task.upper(),
                             "question_num": row["question_num"], "candidate_index": candidate_index,
                             "candidate_item_id": candidate, "score": scores[index, candidate_index]}
                        )
                task_frame = pd.DataFrame(task_predictions)
                predictions.extend(task_predictions)
                if task == "lat":
                    summary.append(
                        {"training_source": source, "seed": seed, "task": "LAT",
                         "majority_accuracy": task_frame.majority_correct.mean(),
                         "mLAT": task_frame.human_agreement.mean(),
                         "archive_gt_accuracy": task_frame.archive_gt_correct.mean()}
                    )
                else:
                    summary.append(
                        {"training_source": source, "seed": seed, "task": "AAT",
                         "overall_accuracy": task_frame.archive_gt_correct.mean(),
                         **{f"{dimension.lower()}_accuracy": task_frame.loc[
                             task_frame.dimension == dimension, "archive_gt_correct"].mean()
                            for dimension in ("Color", "Style", "Occasion", "Season", "Material", "Balance")}}
                    )
    pd.DataFrame(predictions).to_parquet(args.output / "predictions.parquet", index=False)
    pd.DataFrame(candidate_rows).to_parquet(args.output / "candidate_scores.parquet", index=False)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
