"""Audit balanced hypergraph packing for leakage-safe IQON splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")
FRACTIONS = {"train": 0.7, "valid": 0.1, "test": 0.2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--totals", nargs="+", type=int, default=[50_000, 75_000, 100_000, 125_000, 150_000, 200_000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    outfits = pd.read_parquet(args.catalog_dir / "outfits.parquet")
    outfits["canonical"] = outfits.item_ids.map(lambda values: tuple(sorted(map(str, values))))
    outfits = outfits.sort_values("set_id").drop_duplicates("canonical", keep="first").reset_index(drop=True)
    items = pd.read_parquet(args.catalog_dir / "items.parquet")
    if args.image_audit:
        verified = pd.read_parquet(args.image_audit)
        valid_items = set(verified.item_id.astype(str))
        outfits = outfits[outfits.item_ids.map(lambda values: all(str(item) in valid_items for item in values))].reset_index(drop=True)
        items = items[items.item_id.astype(str).isin(valid_items)].merge(
            verified[["item_id", "decoded_rgb_sha256"]], on="item_id", how="inner"
        )
    signature_to_group: dict[object, int] = {}
    item_group: dict[str, int] = {}
    for row in items.sort_values("item_id").itertuples(index=False):
        signature = (
            row.decoded_rgb_sha256
            if args.image_audit
            else (int(row.encoded_crc32), int(row.encoded_size))
        )
        group = signature_to_group.setdefault(signature, len(signature_to_group))
        item_group[str(row.item_id)] = group
    outfit_groups = [tuple({item_group[str(item)] for item in values}) for values in outfits.item_ids]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(outfits))
    reports = []
    for total in args.totals:
        targets = {split: round(total * FRACTIONS[split]) for split in SPLITS}
        targets["train"] += total - sum(targets.values())
        counts = Counter()
        owner: dict[int, str] = {}
        retained: list[list[int]] = []
        conflicts = 0
        full = 0
        for index in order:
            groups = outfit_groups[index]
            existing = {owner[group] for group in groups if group in owner}
            if len(existing) > 1:
                conflicts += 1
                continue
            if existing:
                split = existing.pop()
                if counts[split] >= targets[split]:
                    full += 1
                    continue
            else:
                available = [split for split in SPLITS if counts[split] < targets[split]]
                if not available:
                    break
                split = max(
                    available,
                    key=lambda value: ((targets[value] - counts[value]) / targets[value], -SPLITS.index(value)),
                )
            owner.update((group, split) for group in groups)
            counts[split] += 1
            retained.append([int(index), SPLITS.index(split)])
            if sum(counts.values()) == total:
                break
        reports.append(
            {
                "requested_total": total,
                "targets": targets,
                "retained": dict(counts),
                "filled": sum(counts.values()) == total,
                "conflicts_skipped": conflicts,
                "owned_but_split_full_skipped": full,
                "retained_rows": retained if sum(counts.values()) == max(args.totals) else None,
            }
        )
        print(json.dumps({key: reports[-1][key] for key in ("requested_total", "retained", "filled")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "reports": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
