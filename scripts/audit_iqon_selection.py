"""Quantify selection and author-overlap limitations of IQON3000-Clean."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")


def distribution(counter: Counter, support: list[str]) -> np.ndarray:
    values = np.asarray([counter.get(key, 0) for key in support], dtype=float)
    return values / values.sum()


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)

    def kl(values, target):
        keep = values > 0
        return float(np.sum(values[keep] * np.log2(values[keep] / target[keep])))

    return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)


def counters(rows: list[dict]) -> tuple[Counter, Counter, Counter]:
    lengths = Counter(str(len(row["items"])) for row in rows)
    categories = Counter(item.get("semantic_category", "<missing>") for row in rows for item in row["items"])
    users = Counter(str(row.get("user_id", "<missing>")) for row in rows)
    return lengths, categories, users


def main() -> None:
    protocol = Path("data/protocols/iqon3000_clean")
    source = pd.read_parquet(".venv/iqon_audit/catalog/outfits.parquet")
    eligible = source[
        source.item_ids.map(lambda values: len(values) >= 2 and len(values) == len(set(values)))
    ].copy()
    rows = {
        split: json.loads((protocol / f"{split}.json").read_text(encoding="utf-8"))
        for split in SPLITS
    }
    selected = [row for split in SPLITS for row in rows[split]]
    selected_ids = {str(row["set_id"]) for row in selected}
    source_selected = eligible[eligible.set_id.astype(str).isin(selected_ids)]
    all_length = Counter(eligible.item_ids.map(len).astype(str))
    selected_length, selected_categories, _ = counters(selected)
    source_categories = Counter(
        category or "<missing>"
        for values in eligible.item_categories
        for category in values
    )
    length_support = sorted(set(all_length) | set(selected_length), key=int)
    category_support = sorted(set(source_categories) | set(selected_categories))
    user_sets = {
        split: set(json.loads((protocol / f"group_ids_{split}.json").read_text(encoding="utf-8")).values())
        for split in SPLITS
    }
    report = {
        "eligible_source_outfits": len(eligible),
        "selected_outfits": len(selected),
        "selected_fraction": len(selected) / len(eligible),
        "eligible_users": int(eligible.user_id.nunique()),
        "selected_users": int(source_selected.user_id.nunique()),
        "selected_user_fraction": float(source_selected.user_id.nunique() / eligible.user_id.nunique()),
        "outfit_length_js_divergence_bits": js_divergence(
            distribution(all_length, length_support), distribution(selected_length, length_support)
        ),
        "semantic_category_js_divergence_bits": js_divergence(
            distribution(source_categories, category_support),
            distribution(selected_categories, category_support),
        ),
        "user_overlap_counts": {
            "train_valid": len(user_sets["train"] & user_sets["valid"]),
            "train_test": len(user_sets["train"] & user_sets["test"]),
            "valid_test": len(user_sets["valid"] & user_sets["test"]),
        },
        "split_user_counts": {split: len(values) for split, values in user_sets.items()},
        "interpretation": (
            "Item/image/outfit disjointness is enforced, but users are intentionally not disjoint. "
            "The benchmark supports unseen-item, not unseen-stylist, claims."
        ),
    }
    Path("artifacts/iqon3000_selection_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "outfit_length": length_support,
            "eligible_fraction": distribution(all_length, length_support),
            "selected_fraction": distribution(selected_length, length_support),
        }
    ).to_csv("artifacts/iqon3000_selection_length_distribution.csv", index=False)
    pd.DataFrame(
        {
            "semantic_category": category_support,
            "eligible_fraction": distribution(source_categories, category_support),
            "selected_fraction": distribution(selected_categories, category_support),
        }
    ).to_csv("artifacts/iqon3000_selection_category_distribution.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
