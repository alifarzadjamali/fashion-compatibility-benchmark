"""Sensitivity of IQON uncertainty to positive-outfit item-graph clustering.

This is a read-only reanalysis of frozen per-example predictions.  The primary
paper inference clusters by user; this deliberately conservative sensitivity
clusters each source outfit and its generated questions by the connected
component induced by shared test-split item identifiers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from repbench.encoders.registry import PRIMARY_MODEL_KEYS
from final_phase_statistics import (
    FIXED_COMPARISONS,
    N_BOOTSTRAP,
    cluster_counts,
    interval,
    weighted_binary_bootstrap,
)

ROOT = Path("artifacts/final/iqon3000_clean")
PROTOCOL = Path("data/protocols/iqon3000_clean")
SEED = 20260989


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def component_map() -> tuple[dict[str, str], Counter]:
    rows = json.loads((PROTOCOL / "test.json").read_text(encoding="utf-8"))
    union_find = UnionFind()
    for row in rows:
        item_ids = [str(item["item_id"]) for item in row["items"]]
        for item_id in item_ids:
            union_find.find(item_id)
        for item_id in item_ids[1:]:
            union_find.union(item_ids[0], item_id)
    mapping = {
        str(row["set_id"]): union_find.find(str(row["items"][0]["item_id"]))
        for row in rows
    }
    return mapping, Counter(mapping.values())


def main() -> None:
    groups, component_sizes = component_map()
    rng = np.random.default_rng(SEED)
    cp_frames = {
        model: pd.read_parquet(ROOT / model / "cp_predictions_test.parquet").sort_values("example_id")
        for model in PRIMARY_MODEL_KEYS
    }
    fitb_frames = {
        model: pd.read_parquet(ROOT / model / "fitb_predictions_test.parquet").sort_values("question_id")
        for model in PRIMARY_MODEL_KEYS
    }
    reference_cp = cp_frames[PRIMARY_MODEL_KEYS[0]]
    reference_fitb = fitb_frames[PRIMARY_MODEL_KEYS[0]]
    cp_group_labels = reference_cp.source_set_id.astype(str).map(groups).to_numpy()
    fitb_group_labels = reference_fitb.source_set_id.astype(str).map(groups).to_numpy()
    if pd.isna(cp_group_labels).any() or pd.isna(fitb_group_labels).any():
        raise ValueError("Prediction source outfit lacks item-component assignment")
    cp_counts, cp_groups = cluster_counts(cp_group_labels, rng)
    fitb_counts, fitb_groups = cluster_counts(fitb_group_labels, rng)
    labels = reference_cp.label.to_numpy(dtype=np.int8)
    samples: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for model in PRIMARY_MODEL_KEYS:
        auc, _ = weighted_binary_bootstrap(
            labels,
            cp_frames[model].probability.to_numpy(float),
            cp_counts,
            cp_groups,
        )
        correct = fitb_frames[model].correct.to_numpy(float)
        fitb = np.empty(N_BOOTSTRAP, dtype=float)
        for offset in range(0, N_BOOTSTRAP, 100):
            stop = min(offset + 100, N_BOOTSTRAP)
            weights = fitb_counts[offset:stop, fitb_groups].astype(float)
            fitb[offset:stop] = (weights * correct).sum(axis=1) / weights.sum(axis=1)
        samples[model] = {"cp_auc": auc, "fitb_accuracy": fitb}
        for metric, values in samples[model].items():
            low, high = interval(values)
            point = (
                json.loads((ROOT / model / "metrics.json").read_text(encoding="utf-8"))["splits"]["test"]
            )
            rows.append(
                {
                    "representation": model,
                    "metric": metric,
                    "point": point["cp_roc_auc" if metric == "cp_auc" else "fitb_accuracy"],
                    "component_cluster_ci_low": low,
                    "component_cluster_ci_high": high,
                    "resampling_unit": "IQON test positive-outfit item component",
                }
            )
    pd.DataFrame(rows).to_csv(
        "artifacts/iqon_item_component_bootstrap_sensitivity.csv", index=False
    )
    comparisons = FIXED_COMPARISONS + (
        ("marqo_fashionsiglip", "clip_vitl14_336", "Marqo-FashionSigLIP minus CLIP"),
        ("marqo_fashionsiglip", "resnet50", "Marqo-FashionSigLIP minus ResNet50"),
    )
    pairwise = []
    for first, second, label in comparisons:
        for metric in ("cp_auc", "fitb_accuracy"):
            values = samples[first][metric] - samples[second][metric]
            low, high = interval(values)
            pairwise.append(
                {
                    "comparison": label,
                    "first": first,
                    "second": second,
                    "metric": metric,
                    "effect": float(np.mean(values)),
                    "component_cluster_ci_low": low,
                    "component_cluster_ci_high": high,
                }
            )
    pd.DataFrame(pairwise).to_csv(
        "artifacts/iqon_item_component_bootstrap_pairwise_sensitivity.csv", index=False
    )
    report = {
        "test_positive_outfits": len(groups),
        "item_components": len(component_sizes),
        "largest_component_outfits": max(component_sizes.values()),
        "largest_component_fraction": max(component_sizes.values()) / len(groups),
        "singleton_components": sum(size == 1 for size in component_sizes.values()),
        "bootstrap_replicates": N_BOOTSTRAP,
        "interpretation": (
            "This sensitivity is conservative because one test-split component contains nearly half "
            "of positive outfits. It complements, rather than replaces, primary user-cluster inference."
        ),
    }
    Path("artifacts/iqon_item_component_dependence_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
