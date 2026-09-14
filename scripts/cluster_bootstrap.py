"""Graph-component cluster bootstrap sensitivity for item reuse within test splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from repbench.data.clean_protocol import UnionFind

SEED = 20260912
N_BOOTSTRAP = 2000
MODELS = (
    "resnet50",
    "dinov3_vitl16",
    "clip_vitl14_336",
    "siglip2_b16_384",
    "fashionclip2",
    "marqo_fashionsiglip",
    "gr_lite",
)
PROTOCOLS = {
    "historical_polyvore_d": (Path("data/raw/polyvore_outfits/disjoint"), "primary"),
    "polyvore_d_clean": (Path("data/protocols/polyvore_d_clean"), "clean_primary"),
}
COMPARISONS = (
    ("dinov3_vitl16", "resnet50", "DINOv3 minus ResNet50"),
    ("gr_lite", "dinov3_vitl16", "GR-Lite minus DINOv3"),
    ("fashionclip2", "clip_vitl14_336", "FashionCLIP minus CLIP"),
    ("marqo_fashionsiglip", "siglip2_b16_384", "Marqo-FashionSigLIP minus SigLIP2"),
    (
        "marqo_fashionsiglip",
        "clip_vitl14_336",
        "historical-validation best fashion minus best generic",
    ),
    (
        "marqo_fashionsiglip",
        "resnet50",
        "historical-validation best overall minus ResNet50",
    ),
)


def set_components(protocol: Path) -> tuple[dict[str, int], dict]:
    rows = json.loads((protocol / "test.json").read_text(encoding="utf-8"))
    union_find = UnionFind()
    for row in rows:
        item_ids = [str(item["item_id"]) for item in row["items"]]
        for item_id in item_ids:
            union_find.find(item_id)
        for item_id in item_ids[1:]:
            union_find.union(item_ids[0], item_id)
    roots = sorted({union_find.find(item_id) for item_id in union_find.parent})
    root_index = {root: index for index, root in enumerate(roots)}
    set_to_component = {
        str(row["set_id"]): root_index[union_find.find(str(row["items"][0]["item_id"]))]
        for row in rows
    }
    sizes = defaultdict(int)
    for component in set_to_component.values():
        sizes[component] += 1
    return set_to_component, {
        "component_count": len(roots),
        "largest_component_outfits": max(sizes.values()),
        "singleton_components": sum(size == 1 for size in sizes.values()),
    }


def grouping(protocol: Path, set_to_component: dict[str, int]):
    lines = [
        line.split()
        for line in (protocol / "compatibility_test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    positive = [line for line in lines if line[0] == "1"]
    negative = [line for line in lines if line[0] == "0"]
    if len(positive) != len(negative):
        raise ValueError("Cluster bootstrap expects balanced paired CP construction")
    positive_groups = np.asarray(
        [set_to_component[line[1].split("_", 1)[0]] for line in positive], dtype=np.int32
    )
    cp_groups = np.concatenate((positive_groups, positive_groups))
    fitb = json.loads((protocol / "fill_in_blank_test.json").read_text(encoding="utf-8"))
    fitb_groups = np.asarray(
        [set_to_component[row["question"][0].split("_", 1)[0]] for row in fitb],
        dtype=np.int32,
    )
    return cp_groups, fitb_groups


def main() -> None:
    rows = []
    pairwise_rows = []
    metadata = {}
    for dataset, (protocol, prefix) in PROTOCOLS.items():
        set_to_component, component_metadata = set_components(protocol)
        cp_groups, fitb_groups = grouping(protocol, set_to_component)
        n_components = component_metadata["component_count"]
        rng = np.random.default_rng(SEED)
        counts = rng.multinomial(
            n_components, np.full(n_components, 1 / n_components), size=N_BOOTSTRAP
        ).astype(np.int16)
        cp_all = pd.read_parquet(f"artifacts/{prefix}_cp_predictions.parquet")
        fitb_all = pd.read_parquet(f"artifacts/{prefix}_fitb_predictions.parquet")
        labels = cp_all.loc[cp_all.model_key == MODELS[0]].sort_values("example_index")[
            "label"
        ].to_numpy()
        if len(labels) != len(cp_groups) or len(fitb_groups) != len(
            fitb_all.loc[fitb_all.model_key == MODELS[0]]
        ):
            raise ValueError(f"Prediction/group alignment failed for {dataset}")
        metadata[dataset] = component_metadata
        samples = {}
        for model in MODELS:
            scores = cp_all.loc[cp_all.model_key == model].sort_values("example_index")[
                "score"
            ].to_numpy()
            correct = fitb_all.loc[fitb_all.model_key == model].sort_values("question_index")[
                "correct"
            ].to_numpy(dtype=float)
            auc, pr, fitb = [], [], []
            for sample in counts:
                cp_weights = sample[cp_groups]
                fitb_weights = sample[fitb_groups]
                auc.append(roc_auc_score(labels, scores, sample_weight=cp_weights))
                pr.append(average_precision_score(labels, scores, sample_weight=cp_weights))
                fitb.append(np.average(correct, weights=fitb_weights))
            for metric, values in (("cp_auc", auc), ("pr_auc", pr), ("fitb_acc", fitb)):
                samples.setdefault(model, {})[metric] = np.asarray(values)
                low, high = np.quantile(values, (0.025, 0.975))
                rows.append(
                    {
                        "dataset": dataset,
                        "representation": model,
                        "metric": metric,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_unit": "positive-outfit graph component",
                        "resamples": N_BOOTSTRAP,
                    }
                )
        np.savez_compressed(
            f"artifacts/{prefix}_component_bootstrap_{N_BOOTSTRAP}.npz",
            **{
                f"{model}_{metric}": values
                for model, model_samples in samples.items()
                for metric, values in model_samples.items()
            },
        )
        for first, second, label in COMPARISONS:
            for metric in ("cp_auc", "fitb_acc"):
                difference = samples[first][metric] - samples[second][metric]
                low, high = np.quantile(difference, (0.025, 0.975))
                lower = int(np.sum(difference <= 0))
                upper = int(np.sum(difference >= 0))
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": label,
                        "metric": metric,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_p": min(
                            1.0, 2 * (min(lower, upper) + 1) / (N_BOOTSTRAP + 1)
                        ),
                        "bootstrap_unit": "positive-outfit graph component",
                        "resamples": N_BOOTSTRAP,
                    }
                )
    pd.DataFrame(rows).to_csv("artifacts/component_cluster_bootstrap.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(
        "artifacts/component_cluster_pairwise.csv", index=False
    )
    Path("artifacts/component_cluster_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
