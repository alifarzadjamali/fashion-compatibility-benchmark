"""Prospective leakage-safe IQON3000 protocol construction."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from repbench.data.clean_protocol import SPLITS, file_sha256, generate_questions

TARGET_COUNTS = {"train": 86_800, "valid": 12_400, "test": 24_800}


def _canonical(values) -> tuple[str, ...]:
    return tuple(sorted(map(str, values)))


def _build_item_categories(outfits: pd.DataFrame) -> tuple[dict[str, str], int]:
    observations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in outfits.itertuples(index=False):
        for item_id, category in zip(row.item_ids, row.item_categories, strict=True):
            observations[str(item_id)][str(category)] += 1
    inconsistent = sum(len(values) > 1 for values in observations.values())
    categories = {
        item_id: min(values, key=lambda category: (-values[category], category))
        for item_id, values in observations.items()
    }
    return categories, inconsistent


def _pack(
    outfits: pd.DataFrame,
    item_to_group: dict[str, str],
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(outfits))
    group_owner: dict[str, str] = {}
    assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
    conflicts = 0
    full = 0
    for index in order:
        groups = {item_to_group[str(item)] for item in outfits.iloc[index].item_ids}
        existing = {group_owner[group] for group in groups if group in group_owner}
        if len(existing) > 1:
            conflicts += 1
            continue
        if existing:
            split = existing.pop()
            if len(assigned[split]) >= TARGET_COUNTS[split]:
                full += 1
                continue
        else:
            available = [split for split in SPLITS if len(assigned[split]) < TARGET_COUNTS[split]]
            if not available:
                break
            split = max(
                available,
                key=lambda value: (
                    (TARGET_COUNTS[value] - len(assigned[value])) / TARGET_COUNTS[value],
                    -SPLITS.index(value),
                ),
            )
        group_owner.update((group, split) for group in groups)
        assigned[split].append(int(index))
        if all(len(assigned[split]) == TARGET_COUNTS[split] for split in SPLITS):
            break
    if any(len(assigned[split]) != TARGET_COUNTS[split] for split in SPLITS):
        raise RuntimeError(
            "Verified leakage groups cannot fill locked IQON capacities: "
            + repr({split: len(assigned[split]) for split in SPLITS})
        )
    return assigned, {"conflict_outfits_skipped": conflicts, "owned_full_split_skipped": full}


def _protocol_rows(frame: pd.DataFrame, categories: dict[str, str]) -> list[dict]:
    rows = []
    for row in frame.sort_values("set_id").itertuples(index=False):
        rows.append(
            {
                "set_id": str(row.set_id),
                "user_id": str(row.user_id),
                "items": [
                    {
                        "item_id": str(item_id),
                        "index": index,
                        "semantic_category": categories[str(item_id)],
                    }
                    for index, item_id in enumerate(row.item_ids, start=1)
                ],
            }
        )
    return rows


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def construct_iqon_clean(
    catalog_dir: Path,
    output_root: Path,
    source_audit_path: Path,
    archive_path: Path,
    seed: int = 20260912,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    outfits = pd.read_parquet(catalog_dir / "outfits.parquet")
    outfits["canonical"] = outfits.item_ids.map(_canonical)
    duplicate_count = int(outfits.canonical.duplicated().sum())
    outfits = outfits.sort_values("set_id").drop_duplicates("canonical", keep="first").reset_index(drop=True)
    images = pd.read_parquet(catalog_dir / "image_audit_resolved.parquet")
    images["item_id"] = images.item_id.astype(str)
    valid_items = set(images.item_id)
    usable_mask = outfits.item_ids.map(lambda values: all(str(item) in valid_items for item in values))
    corrupt_outfits = int((~usable_mask).sum())
    outfits = outfits[usable_mask].reset_index(drop=True)
    categories, inconsistent_categories = _build_item_categories(outfits)
    image_by_item = images.set_index("item_id")
    item_to_group = image_by_item.decoded_rgb_sha256.to_dict()
    assigned, packing = _pack(outfits, item_to_group, seed)

    metadata = {item: {"semantic_category": category} for item, category in categories.items()}
    selected_frames = {split: outfits.iloc[indices].copy() for split, indices in assigned.items()}
    selected_rows = {
        split: _protocol_rows(frame, categories) for split, frame in selected_frames.items()
    }
    split_items = {
        split: {str(item) for values in frame.item_ids for item in values}
        for split, frame in selected_frames.items()
    }
    selected_all = set().union(*split_items.values())
    equivalence = {item: item_to_group[item] for item in selected_all}

    manifest = {
        "protocol": "iqon3000_clean",
        "version": "1.0",
        "construction_seed": seed,
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": file_sha256(archive_path),
        "source_audit": json.loads(source_audit_path.read_text(encoding="utf-8")),
        "target_outfit_counts": TARGET_COUNTS,
        "construction_rules": [
            "Exclude source outfits with fewer than two distinct item IDs.",
            "Keep one deterministic copy of duplicate sorted item-ID outfit tuples.",
            "Exclude outfits containing an item without a decodable canonical image.",
            "Treat item IDs and verified decoded-RGB duplicate-image groups as indivisible hypergraph nodes.",
            "Traverse canonical outfits in a seed-20260912 permutation and greedily fill exact 70/10/20 capacities.",
            "Discard outfits touching groups already owned by multiple splits or a full split.",
            "Generate category/length-matched CP negatives and four-way same-category FITB within each split only.",
            "Never use user ID as a model feature; retain it solely for audit and group-aware uncertainty.",
        ],
        "duplicate_outfits_removed": duplicate_count,
        "outfits_with_corrupt_images_removed": corrupt_outfits,
        "item_category_inconsistency_count": inconsistent_categories,
        "packing": packing,
        "split_overlap": {},
        "splits": {},
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        left_rows = selected_frames[left]
        right_rows = selected_frames[right]
        left_images = image_by_item.loc[list(split_items[left])]
        right_images = image_by_item.loc[list(split_items[right])]
        manifest["split_overlap"][f"{left}_{right}"] = {
            "item_ids": len(split_items[left] & split_items[right]),
            "encoded_sha256": len(set(left_images.encoded_sha256) & set(right_images.encoded_sha256)),
            "decoded_rgb_sha256": len(set(left_images.decoded_rgb_sha256) & set(right_images.decoded_rgb_sha256)),
            "outfit_tuples": len(set(left_rows.canonical) & set(right_rows.canonical)),
        }
    for split_index, split in enumerate(SPLITS):
        rows = selected_rows[split]
        outfit_path = output_root / f"{split}.json"
        _write_json(outfit_path, rows)
        cp_lines, fitb_rows, diagnostics = generate_questions(
            rows,
            metadata,
            seed + 10_000 * (split_index + 1),
            {item: equivalence[item] for item in split_items[split]},
        )
        cp_path = output_root / f"compatibility_{split}.txt"
        cp_path.write_text("\n".join(cp_lines) + "\n", encoding="utf-8")
        fitb_path = output_root / f"fill_in_blank_{split}.json"
        _write_json(fitb_path, fitb_rows)
        groups_path = output_root / f"group_ids_{split}.json"
        _write_json(
            groups_path,
            {str(row.set_id): str(row.user_id) for row in selected_frames[split].itertuples(index=False)},
        )
        manifest["splits"][split] = {
            "outfits": len(rows),
            "unique_items": len(split_items[split]),
            "users": int(selected_frames[split].user_id.nunique()),
            "outfit_length_counts": dict(
                sorted(Counter(len(row["items"]) for row in rows).items())
            ),
            "semantic_category_occurrences": dict(sorted(Counter(
                item["semantic_category"] for row in rows for item in row["items"]
            ).items())),
            "question_generation": diagnostics,
            "sha256": {
                path.name: file_sha256(path)
                for path in (outfit_path, cp_path, fitb_path, groups_path)
            },
        }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def extract_selected_images(
    protocol_root: Path,
    catalog_dir: Path,
    archive_path: Path,
    images_root: Path,
) -> dict:
    if images_root.exists() and any(images_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty image cache: {images_root}")
    images_root.mkdir(parents=True, exist_ok=True)
    item_ids = {
        str(item["item_id"])
        for split in SPLITS
        for row in json.loads((protocol_root / f"{split}.json").read_text(encoding="utf-8"))
        for item in row["items"]
    }
    audit = pd.read_parquet(catalog_dir / "image_audit_resolved.parquet").set_index("item_id")
    missing = item_ids - set(audit.index.astype(str))
    if missing:
        raise KeyError(f"Verified image audit lacks {len(missing)} protocol items")
    bytes_written = 0
    with zipfile.ZipFile(archive_path) as archive:
        for index, item_id in enumerate(sorted(item_ids), start=1):
            row = audit.loc[item_id]
            raw = archive.read(row.zip_entry)
            if hashlib.sha256(raw).hexdigest() != row.encoded_sha256:
                raise ValueError(f"Archive image hash changed for item {item_id}")
            destination = images_root / f"{item_id}.jpg"
            destination.write_bytes(raw)
            bytes_written += len(raw)
            if index % 25_000 == 0:
                print(f"extracted {index:,}/{len(item_ids):,} protocol images", flush=True)
    return {"items": len(item_ids), "bytes": bytes_written, "root": str(images_root.resolve())}
