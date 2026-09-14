"""Train/evaluate the fixed task-specific secondary baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_benchmark import load_protocol_splits
from sklearn.metrics import average_precision_score, roc_auc_score

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.models.outfit_transformer import OutfitTransformer, focal_binary_cross_entropy
from repbench.utils.seeds import seed_everything

SPLITS = ("train", "valid", "test")


def encode_examples(examples, item_index, max_length=20):
    indices = np.full((len(examples), max_length), -1, dtype=np.int32)
    mask = np.ones((len(examples), max_length), dtype=bool)
    for row_index, example in enumerate(examples):
        if not 2 <= len(example.item_ids) <= max_length:
            raise ValueError(f"Unsupported outfit length: {len(example.item_ids)}")
        values = [item_index[item] for item in example.item_ids]
        indices[row_index, : len(values)] = values
        mask[row_index, : len(values)] = False
    return indices, mask


def batches(indices, mask, labels, embeddings, batch_size, order):
    for offset in range(0, len(order), batch_size):
        rows = order[offset : offset + batch_size]
        item_indices = indices[rows]
        safe = np.maximum(item_indices, 0)
        values = np.asarray(embeddings[safe], dtype=np.float32)
        yield values, mask[rows], None if labels is None else labels[rows]


def predict(model, indices, mask, embeddings, batch_size, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for values, masks, _ in batches(indices, mask, None, embeddings, batch_size, np.arange(len(indices))):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(torch.from_numpy(values).to(device), torch.from_numpy(masks).to(device))
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs)


def run_seed(args, dataset, seed):
    destination = args.output / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite baseline run: {destination}")
    destination.mkdir(parents=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = args.embedding_cache
    index, embedding_values = {}, {}
    examples, encoded, labels = {}, {}, {}
    remapped_cache = load_protocol_splits(cache, dataset)
    for split in SPLITS:
        item_ids, embedding_values[split] = remapped_cache[split]
        if embedding_values[split].shape != (len(item_ids), 2048):
            raise ValueError(f"Invalid ResNet cache for {split}")
        index[split] = {item: item_index for item_index, item in enumerate(item_ids)}
        examples[split] = dataset.compatibility(split)
        encoded[split] = encode_examples(examples[split], index[split])
        labels[split] = np.asarray([row.label for row in examples[split]], dtype=np.float32)
    model = OutfitTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    best_auc, best_epoch, stale = -np.inf, 0, 0
    history = []
    runtime_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for values, masks, targets in batches(
            *encoded["train"], labels["train"], embedding_values["train"], args.batch_size,
            generator.permutation(len(labels["train"])),
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(torch.from_numpy(values).to(device), torch.from_numpy(masks).to(device))
                loss = focal_binary_cross_entropy(logits, torch.from_numpy(targets).to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite baseline loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        valid = predict(model, *encoded["valid"], embedding_values["valid"], args.batch_size, device)
        valid_auc = float(roc_auc_score(labels["valid"], valid))
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "validation_auc": valid_auc})
        if valid_auc > best_auc + 1e-7:
            best_auc, best_epoch, stale = valid_auc, epoch, 0
            torch.save(model.state_dict(), destination / "best_state.pt")
        else:
            stale += 1
        print(f"seed={seed} epoch={epoch} val_auc={valid_auc:.6f}", flush=True)
        if stale >= 5:
            break
    training_seconds = time.perf_counter() - runtime_started
    model.load_state_dict(torch.load(destination / "best_state.pt", map_location=device, weights_only=True))
    predictions = {}
    for split in SPLITS:
        probability = predict(model, *encoded[split], embedding_values[split], args.batch_size, device)
        predictions[split] = probability
        pd.DataFrame(
            {"seed": seed, "split": split, "example_index": np.arange(len(probability)), "label": labels[split], "probability": probability}
        ).to_parquet(destination / f"cp_predictions_{split}.parquet", index=False)
    questions = dataset.fitb("test")
    fitb_examples = [
        type(examples["test"][0])(0, question.question_item_ids + (candidate,))
        for question in questions
        for candidate in question.candidate_item_ids
    ]
    fitb_encoded = encode_examples(fitb_examples, index["test"])
    fitb_scores = predict(model, *fitb_encoded, embedding_values["test"], args.batch_size, device).reshape(len(questions), 4)
    fitb_correct = np.argmax(fitb_scores, axis=1) == np.asarray([row.correct_index for row in questions])
    pd.DataFrame(
        {"seed": seed, "question_index": np.arange(len(questions)), "correct": fitb_correct, "predicted_index": np.argmax(fitb_scores, axis=1)}
    ).to_parquet(destination / "fitb_predictions_test.parquet", index=False)
    pd.DataFrame(
        {
            "seed": seed,
            "question_index": np.repeat(np.arange(len(questions)), 4),
            "candidate_index": np.tile(np.arange(4), len(questions)),
            "score": fitb_scores.ravel(),
        }
    ).to_parquet(destination / "fitb_candidate_scores_test.parquet", index=False)
    pd.DataFrame(history).to_csv(destination / "training_history.csv", index=False)
    metrics = {
        "dataset": args.dataset_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_auc": best_auc,
        "test_cp_auc": float(roc_auc_score(labels["test"], predictions["test"])),
        "test_pr_auc": float(average_precision_score(labels["test"], predictions["test"])),
        "test_fitb_accuracy": float(fitb_correct.mean()),
        "training_seconds": training_seconds,
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--append", action="store_true", help="Add new seed directories without replacing prior seeds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.append)
    configuration = {
        "dataset": args.dataset_name,
        "embedding_cache": str(args.embedding_cache),
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "model": "OutfitTransformer image-only adaptation",
        "input_dimension": 2048,
        "projection_dimension": 64,
        "layers": 6,
        "heads": 16,
        "feedforward_dimension": 256,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "loss": "focal BCE gamma=2 alpha=0.25",
        "early_stopping": "validation ROC-AUC, patience=5",
    }
    config_path = args.output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        stable_keys = set(configuration) - {"seeds"}
        if any(existing[key] != configuration[key] for key in stable_keys):
            raise ValueError("Append configuration differs from frozen baseline configuration")
        configuration["seeds"] = sorted(set(existing["seeds"]) | set(args.seeds))
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    dataset = PolyvoreDisjoint(args.dataset_root, args.protocol_root, args.dataset_name)
    results = [run_seed(args, dataset, seed) for seed in args.seeds]
    if args.append and (args.output / "summary.csv").is_file():
        previous = pd.read_csv(args.output / "summary.csv")
        results = previous.to_dict("records") + results
    summary = pd.DataFrame(results).sort_values("seed")
    if summary.seed.duplicated().any():
        raise ValueError("Duplicate baseline seed in summary")
    summary.to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
