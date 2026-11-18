"""Image-only OutfitTransformer v2 baseline on frozen embeddings (HANDOFF_V2 §6.2).

  train     For every lr x seed: train on clean TRAIN CP examples, early-stop on VALID
            ROC-AUC (patience 10, <= 100 epochs). Writes the best state, history and
            VALID CP/FITB scores. Resumable: finished (lr, seed) runs are verified and
            skipped; nothing is overwritten.
  select    Pick the learning rate with the highest mean best-VALID AUC across seeds
            (validation only; ties -> smaller lr). Writes selection.json once.
  evaluate  Score one split with the selected lr for every seed. `test` requires
            --confirm-protocol-frozen.

One encoder (embedding store) per output root.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.v2.outfit_transformer_v2 import (
    LR_GRID,
    EmbeddingTable,
    TrainConfig,
    score_cp,
    score_fitb,
    set_determinism,
    train,
)
from repbench.v2.state import sha256, write_json
from repbench.v2.stores import load_embedding_store, protocol_hashes

SPLITS = ("train", "valid", "test")


def lr_name(lr):
    return f"lr_{lr:g}"


def load_dataset(args):
    protocol = args.protocol_root.resolve()
    if (protocol.name == "disjoint" or args.dataset_name.startswith("historical")) and not (
        args.allow_historical
    ):
        raise SystemExit("Refusing a historical split mapping; v2 baselines use the clean splits")
    return PolyvoreDisjoint(protocol.parent, protocol, args.dataset_name)


def has_fitb(protocol, split):
    return any(
        (protocol / name).is_file()
        for name in (f"fill_in_blank_{split}.json", f"fill_in_the_blank_{split}.json")
    )


def structure_max_length(dataset, protocol):
    """Longest set over all splits' CP outfits and FITB (partial + candidate) sets.

    Reads structure only (lengths), never labels or outcomes of the test split.
    """
    longest = 0
    for split in SPLITS:
        try:
            longest = max([longest, *(len(e.item_ids) for e in dataset.compatibility(split))])
        except FileNotFoundError:
            pass
        if has_fitb(protocol, split):
            longest = max([longest, *(len(q.question_item_ids) + 1 for q in dataset.fitb(split))])
    return longest


def run_record(args, dataset, provenance):
    protocol = args.protocol_root.resolve()
    return {
        "dataset_name": args.dataset_name,
        "protocol_root": str(protocol),
        "protocol_sha256": protocol_hashes(protocol),
        "encoder": args.encoder,
        "embeddings": provenance,
        "max_length": structure_max_length(dataset, protocol),
        "architecture": "repbench.models.outfit_transformer.OutfitTransformer (v1, unchanged)",
        "early_stopping": f"validation CP ROC-AUC, patience {args.patience}",
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
    }


def check_run_record(path, record):
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"{path} differs from current inputs; use a fresh output root")
    else:
        write_json(path, record)


def write_split_scores(destination, model, table, dataset, split, max_length, device):
    examples = dataset.compatibility(split)
    labels = np.asarray([e.label for e in examples])
    cp = score_cp(model, table, examples, max_length=max_length, device=device)
    pd.DataFrame({"example_index": np.arange(len(cp)), "label": labels, "score": cp}).to_csv(
        destination / f"cp_scores_{split}.csv", index=False
    )
    metrics = {f"{split}_cp_auc": float(roc_auc_score(labels, cp))}
    if has_fitb(dataset.split_root, split):
        questions = dataset.fitb(split)
        scores, correct = score_fitb(model, table, questions, max_length=max_length, device=device)
        k = scores.shape[1]
        pd.DataFrame(
            {
                "question_index": np.repeat(np.arange(len(questions)), k),
                "candidate_index": np.tile(np.arange(k), len(questions)),
                "score": scores.ravel(),
                "is_answer": np.concatenate([np.arange(k) == q.correct_index for q in questions]),
            }
        ).to_csv(destination / f"fitb_candidate_scores_{split}.csv", index=False)
        pd.DataFrame(
            {
                "question_index": np.arange(len(questions)),
                "predicted_index": scores.argmax(1),
                "correct_index": [q.correct_index for q in questions],
                "correct": correct,
            }
        ).to_csv(destination / f"fitb_predictions_{split}.csv", index=False)
        metrics[f"{split}_fitb_accuracy"] = float(correct.mean())
    return metrics


def command_train(args):
    dataset = load_dataset(args)
    store, provenance = load_embedding_store(args.embeddings)
    table = EmbeddingTable(store)
    del store
    args.output_root.mkdir(parents=True, exist_ok=True)
    record = run_record(args, dataset, provenance)
    check_run_record(args.output_root / "run.json", record)
    max_length = record["max_length"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_examples, valid_examples = dataset.compatibility("train"), dataset.compatibility("valid")
    for lr in args.lrs:
        for seed in args.seeds:
            destination = args.output_root / lr_name(lr) / f"seed_{seed}"
            config = TrainConfig(
                lr=lr,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            if (destination / "metrics.json").exists():
                saved = json.loads((destination / "metrics.json").read_text())
                if saved["config"] != json.loads(json.dumps(config.__dict__)):
                    raise ValueError(f"{destination} was trained with another configuration")
                print(f"Skipping finished {destination}", flush=True)
                continue
            if destination.exists():
                raise FileExistsError(f"Unfinished run {destination}; inspect and remove it first")
            if (args.output_root / "STOP").exists():
                print("STOP requested", flush=True)
                return
            destination.mkdir(parents=True)
            result = train(
                config,
                table,
                train_examples,
                valid_examples,
                max_length=max_length,
                device=device,
                log=lambda m: print(m, flush=True),
            )
            torch.save(result.best_state, destination / "best_state.pt")
            pd.DataFrame(result.history).to_csv(destination / "history.csv", index=False)
            metrics = write_split_scores(
                destination, result.model, table, dataset, "valid", max_length, device
            )
            metrics.update(
                config=result.config,
                encoder=args.encoder,
                best_epoch=result.best_epoch,
                best_valid_auc=result.best_valid_auc,
                epochs_run=result.epochs_run,
                early_stopping_fired=result.early_stopped,
                training_seconds=result.training_seconds,
                device=str(device),
                best_state_sha256=sha256(destination / "best_state.pt"),
                finished_utc=datetime.now(UTC).isoformat(),
            )
            write_json(destination / "metrics.json", metrics)


def collect(output_root):
    rows = []
    for path in sorted(output_root.glob("lr_*/seed_*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append(
            {
                "lr": m["config"]["lr"],
                "seed": m["config"]["seed"],
                "best_valid_auc": m["best_valid_auc"],
                "best_epoch": m["best_epoch"],
                "early_stopping_fired": m["early_stopping_fired"],
            }
        )
    return pd.DataFrame(rows)


def command_select(args):
    frame = collect(args.output_root)
    if frame.empty:
        raise SystemExit("No finished runs")
    seed_sets = frame.groupby("lr").seed.apply(lambda s: tuple(sorted(s)))
    if seed_sets.nunique() != 1:
        raise SystemExit(f"Incomplete grid; seeds per lr differ: {seed_sets.to_dict()}")
    if len(seed_sets.iloc[0]) < args.min_seeds:
        raise SystemExit(f"Need >= {args.min_seeds} seeds per lr")
    grid = frame.groupby("lr").best_valid_auc.agg(["mean", "std", "count"]).reset_index()
    grid = grid.sort_values(["mean", "lr"], ascending=[False, True])
    selected = float(grid.iloc[0].lr)
    selection = {
        "rule": "highest mean best-validation CP AUC across seeds; ties -> smaller lr",
        "selected_lr": selected,
        "seeds": list(seed_sets.iloc[0]),
        "grid": grid.to_dict("records"),
        "runs": frame.to_dict("records"),
        "early_stopping_fired_count": int(frame.early_stopping_fired.sum()),
        "runs_total": len(frame),
        "run_json_sha256": sha256(args.output_root / "run.json"),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(args.output_root / "selection.json", selection)
    print(json.dumps({"selected_lr": selected}), flush=True)


def command_evaluate(args):
    if args.split == "test" and not args.confirm_protocol_frozen:
        raise SystemExit("Test evaluation is blocked before the protocol freeze")
    selection = json.loads((args.output_root / "selection.json").read_text())
    run = json.loads((args.output_root / "run.json").read_text())
    if sha256(args.output_root / "run.json") != selection["run_json_sha256"]:
        raise ValueError("run.json changed after selection")
    dataset = load_dataset(args)
    if protocol_hashes(dataset.split_root) != run["protocol_sha256"]:
        raise ValueError("Protocol files changed since training")
    store, provenance = load_embedding_store(args.embeddings)
    if provenance != run["embeddings"]:
        raise ValueError("Embedding store differs from the one used in training")
    table = EmbeddingTable(store)
    del store
    set_determinism()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lr = selection["selected_lr"]
    root = args.output_root / f"eval_{args.split}"
    root.mkdir(exist_ok=False)
    summary = []
    for seed in selection["seeds"]:
        source = args.output_root / lr_name(lr) / f"seed_{seed}"
        trained = json.loads((source / "metrics.json").read_text())
        if sha256(source / "best_state.pt") != trained["best_state_sha256"]:
            raise ValueError(f"{source}/best_state.pt changed")
        from repbench.models.outfit_transformer import OutfitTransformer

        model = OutfitTransformer(input_dim=table.dim, model_dim=trained["config"]["model_dim"])
        model.load_state_dict(torch.load(source / "best_state.pt", weights_only=True))
        model = model.to(device)
        destination = root / f"seed_{seed}"
        destination.mkdir()
        metrics = write_split_scores(
            destination, model, table, dataset, args.split, run["max_length"], device
        )
        metrics.update(
            lr=lr,
            seed=seed,
            best_epoch=trained["best_epoch"],
            best_valid_auc=trained["best_valid_auc"],
            early_stopping_fired=trained["early_stopping_fired"],
        )
        write_json(destination / "metrics.json", metrics)
        summary.append(metrics)
    write_json(root / "summary.json", {"split": args.split, "lr": lr, "seeds": summary})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "select", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--output-root", type=Path, required=True)
        if name == "select":
            p.add_argument("--min-seeds", type=int, default=5)
            continue
        p.add_argument("--protocol-root", type=Path, required=True)
        p.add_argument("--dataset-name", required=True)
        p.add_argument("--embeddings", type=Path, required=True)
        p.add_argument("--device", default=None)
        p.add_argument("--allow-historical", action="store_true")
        if name == "train":
            p.add_argument("--encoder", required=True, help="Encoder key (label only)")
            p.add_argument("--lrs", type=float, nargs="+", default=list(LR_GRID))
            p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
            p.add_argument("--max-epochs", type=int, default=100)
            p.add_argument("--patience", type=int, default=10)
            p.add_argument("--batch-size", type=int, default=50)
        else:
            p.add_argument("--split", choices=["valid", "test"], required=True)
            p.add_argument("--confirm-protocol-frozen", action="store_true")
    args = parser.parse_args(argv)
    {"train": command_train, "select": command_select, "evaluate": command_evaluate}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
