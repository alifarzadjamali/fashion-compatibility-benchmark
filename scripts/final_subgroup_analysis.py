"""Post-hoc descriptive subgroup audit from immutable primary predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_MODEL_KEYS


def length_group(length: int) -> str:
    return str(length) if length <= 5 else "6+"


def category_map(dataset_name: str, protocol_root: Path) -> dict[str, str]:
    if dataset_name == "polyvore_d_clean":
        metadata = json.loads(
            Path("data/raw/polyvore_outfits/polyvore_item_metadata.json").read_text(encoding="utf-8")
        )
        return {str(item): str(values["semantic_category"]) for item, values in metadata.items()}
    rows = json.loads((protocol_root / "test.json").read_text(encoding="utf-8"))
    return {
        str(item["item_id"]): str(item["semantic_category"])
        for row in rows
        for item in row["items"]
    }


def prediction_paths(dataset_name: str, model: str) -> tuple[Path, Path]:
    if dataset_name == "polyvore_d_clean":
        return (
            Path(f"artifacts/clean_primary_cp_predictions_{model}.parquet"),
            Path(f"artifacts/clean_primary_fitb_predictions_{model}.parquet"),
        )
    root = Path("artifacts/final/iqon3000_clean") / model
    return root / "cp_predictions_test.parquet", root / "fitb_predictions_test.parquet"


def run_dataset(dataset_name: str, dataset_root: Path, protocol_root: Path) -> pd.DataFrame:
    dataset = PolyvoreDisjoint(dataset_root, protocol_root, dataset_name)
    cp_examples = dataset.compatibility("test")
    fitb_questions = dataset.fitb("test")
    categories = category_map(dataset_name, protocol_root)
    rows = []
    for model in PRIMARY_MODEL_KEYS:
        cp_path, fitb_path = prediction_paths(dataset_name, model)
        cp = pd.read_parquet(cp_path)
        fitb = pd.read_parquet(fitb_path)
        probability_column = "probability" if "probability" in cp else "score"
        if len(cp) != len(cp_examples) or len(fitb) != len(fitb_questions):
            raise ValueError(f"Subgroup prediction alignment failure: {dataset_name}/{model}")
        labels = np.asarray([example.label for example in cp_examples])
        if not np.array_equal(labels, cp.label.to_numpy()):
            raise ValueError(f"Subgroup label alignment failure: {dataset_name}/{model}")
        scores = cp[probability_column].to_numpy()
        lengths = np.asarray([length_group(len(example.item_ids)) for example in cp_examples])
        for group in sorted(set(lengths)):
            selected = lengths == group
            rows.append({"dataset": dataset_name, "representation": model, "task": "CP",
                         "subgroup_type": "outfit_length", "subgroup": group,
                         "n": int(selected.sum()), "positive_n": int(labels[selected].sum()),
                         "metric": "roc_auc", "value": roc_auc_score(labels[selected], scores[selected])})
        category_sets = [{categories[item] for item in example.item_ids} for example in cp_examples]
        for category in sorted(set().union(*category_sets)):
            selected = np.asarray([category in values for values in category_sets])
            if selected.sum() < 200 or len(np.unique(labels[selected])) < 2:
                continue
            rows.append({"dataset": dataset_name, "representation": model, "task": "CP",
                         "subgroup_type": "contains_category", "subgroup": category,
                         "n": int(selected.sum()), "positive_n": int(labels[selected].sum()),
                         "metric": "roc_auc", "value": roc_auc_score(labels[selected], scores[selected])})
        fitb_lengths = np.asarray([length_group(len(question.question_item_ids) + 1) for question in fitb_questions])
        correct = fitb.correct.to_numpy(dtype=bool)
        for group in sorted(set(fitb_lengths)):
            selected = fitb_lengths == group
            rows.append({"dataset": dataset_name, "representation": model, "task": "FITB",
                         "subgroup_type": "outfit_length", "subgroup": group,
                         "n": int(selected.sum()), "positive_n": None,
                         "metric": "accuracy", "value": correct[selected].mean()})
        blank_categories = np.asarray([
            categories[question.candidate_item_ids[question.correct_index]] for question in fitb_questions
        ])
        for category in sorted(set(blank_categories)):
            selected = blank_categories == category
            if selected.sum() < 100:
                continue
            rows.append({"dataset": dataset_name, "representation": model, "task": "FITB",
                         "subgroup_type": "blank_category", "subgroup": category,
                         "n": int(selected.sum()), "positive_n": None,
                         "metric": "accuracy", "value": correct[selected].mean()})
    frame = pd.DataFrame(rows)
    destination = Path("artifacts/final/subgroup")
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / f"{dataset_name}.csv", index=False)
    return frame


def main() -> None:
    results = pd.concat((
        run_dataset("polyvore_d_clean", Path("data/raw/polyvore_outfits"), Path("data/protocols/polyvore_d_clean")),
        run_dataset("iqon3000_clean", Path(".venv/iqon3000_clean"), Path("data/protocols/iqon3000_clean")),
    ), ignore_index=True)
    table_root = Path("reports/tables/final")
    results.to_csv(table_root / "subgroup_performance.csv", index=False)
    summary = (
        results.groupby(["dataset", "task", "subgroup_type", "subgroup", "n"], as_index=False)
        .value.agg(["min", "max", "mean"])
    )
    summary.to_csv(table_root / "subgroup_performance_summary.csv", index=False)
    (table_root / "subgroup_performance_summary.md").write_text(
        summary.to_markdown(index=False), encoding="utf-8"
    )
    failure_rows = []
    for dataset_name in ("polyvore_d_clean", "iqon3000_clean"):
        cp_correct, fitb_correct = [], []
        for model in PRIMARY_MODEL_KEYS:
            cp_path, fitb_path = prediction_paths(dataset_name, model)
            cp = pd.read_parquet(cp_path)
            score_column = "probability" if "probability" in cp else "score"
            cp_correct.append((cp[score_column].to_numpy() >= 0.5) == cp.label.to_numpy(dtype=bool))
            fitb_correct.append(pd.read_parquet(fitb_path).correct.to_numpy(dtype=bool))
        for task, values in (("CP", cp_correct), ("FITB", fitb_correct)):
            correct_count = np.stack(values, axis=1).sum(axis=1)
            failure_rows.append({
                "dataset": dataset_name,
                "task": task,
                "n": len(correct_count),
                "unanimous_wrong_n": int((correct_count == 0).sum()),
                "majority_wrong_n": int((correct_count <= 3).sum()),
                "model_disagreement_n": int(((correct_count > 0) & (correct_count < 7)).sum()),
                "unanimous_correct_n": int((correct_count == 7).sum()),
            })
    failure = pd.DataFrame(failure_rows)
    failure.to_csv(table_root / "failure_consensus_audit.csv", index=False)
    (table_root / "failure_consensus_audit.md").write_text(
        failure.to_markdown(index=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
