"""Feasibility audit for a user- and item-disjoint IQON3000 protocol.

This is construction-only: it never loads a representation or evaluates a test
example. Cross-boundary outfits are discarded after deterministic user grouping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")
TARGET_FRACTIONS = {"train": 0.7, "valid": 0.1, "test": 0.2}


def assign_users(outfits: pd.DataFrame, seed: int) -> dict[str, str]:
    targets = {split: TARGET_FRACTIONS[split] * len(outfits) for split in SPLITS}
    counts = outfits.groupby("user_id").size().to_dict()
    rng = np.random.default_rng(seed)
    tie = dict(zip(sorted(counts), rng.permutation(len(counts)), strict=True))
    ordered = sorted(counts, key=lambda user: (-counts[user], tie[user], user))
    assigned_counts = Counter()
    assignments: dict[str, str] = {}
    for user in ordered:
        # Minimize normalized overflow, then normalized distance to target.
        chosen = min(
            SPLITS,
            key=lambda split: (
                max(0.0, (assigned_counts[split] + counts[user] - targets[split]) / targets[split]),
                (assigned_counts[split] + counts[user]) / targets[split],
                SPLITS.index(split),
            ),
        )
        assignments[user] = chosen
        assigned_counts[chosen] += counts[user]
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--owner-balance-exponent", type=float, default=0.5)
    args = parser.parse_args()

    outfits = pd.read_parquet(args.catalog_dir / "outfits.parquet")
    outfits["canonical"] = outfits.item_ids.map(lambda values: tuple(sorted(map(str, values))))
    outfits = outfits.sort_values("set_id").drop_duplicates("canonical", keep="first").copy()
    assignments = assign_users(outfits, args.seed)
    outfits["split"] = outfits.user_id.map(assignments)

    item_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in outfits.itertuples(index=False):
        for item_id in row.item_ids:
            item_split_counts[str(item_id)][row.split] += 1
    item_owner = {
        item_id: min(
            SPLITS,
            key=lambda split: (
                -counts[split] / TARGET_FRACTIONS[split] ** args.owner_balance_exponent,
                SPLITS.index(split),
            ),
        )
        for item_id, counts in item_split_counts.items()
    }

    # CRC+size is a screen for byte-identical images. Assign all IDs sharing a
    # signature to one owner for this feasibility pass; SHA-256 verification is
    # required before final construction.
    items = pd.read_parquet(args.catalog_dir / "items.parquet")
    items["signature"] = list(zip(items.encoded_crc32, items.encoded_size, strict=True))
    signature_ids = items.groupby("signature").item_id.agg(list)
    for ids in signature_ids:
        owners = Counter()
        for item_id in ids:
            counts = item_split_counts.get(str(item_id), Counter())
            owners.update(counts)
        if not owners:
            continue
        owner = min(
            SPLITS,
            key=lambda split: (
                -owners[split] / TARGET_FRACTIONS[split] ** args.owner_balance_exponent,
                SPLITS.index(split),
            ),
        )
        for item_id in ids:
            item_owner[str(item_id)] = owner

    def is_pure(row: pd.Series) -> bool:
        return all(item_owner[str(item)] == row.split for item in row.item_ids)

    outfits["retained"] = outfits.apply(is_pure, axis=1)
    retained = outfits[outfits.retained].copy()
    split_counts = retained.split.value_counts().reindex(SPLITS, fill_value=0).to_dict()
    user_counts = retained.groupby("split").user_id.nunique().reindex(SPLITS, fill_value=0).to_dict()
    item_sets = {
        split: {str(item) for values in retained.loc[retained.split == split, "item_ids"] for item in values}
        for split in SPLITS
    }
    report = {
        "seed": args.seed,
        "owner_balance_exponent": args.owner_balance_exponent,
        "input_unique_outfits": len(outfits),
        "initial_user_partition_counts": outfits.split.value_counts().reindex(SPLITS, fill_value=0).to_dict(),
        "retained_outfit_counts": split_counts,
        "retained_fraction": len(retained) / len(outfits),
        "retained_split_fractions": {split: split_counts[split] / len(retained) for split in SPLITS},
        "retained_user_counts": user_counts,
        "retained_item_counts": {split: len(item_sets[split]) for split in SPLITS},
        "item_overlap": {
            "train_valid": len(item_sets["train"] & item_sets["valid"]),
            "train_test": len(item_sets["train"] & item_sets["test"]),
            "valid_test": len(item_sets["valid"] & item_sets["test"]),
        },
        "removed_boundary_outfits": int((~outfits.retained).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
