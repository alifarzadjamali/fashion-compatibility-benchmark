"""Audit the immutable IQON3000 archive before protocol construction.

The script deliberately reads the ZIP in place.  It writes only derived metadata;
raw images are neither modified nor re-encoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
            return value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    outfits: list[dict] = []
    item_occurrences: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    user_counts: Counter[str] = Counter()
    encoded_signatures: dict[str, set[tuple[int, int]]] = defaultdict(set)
    image_entry_for_item: dict[str, str] = {}
    union_find = UnionFind()
    malformed: list[dict] = []

    with zipfile.ZipFile(args.archive) as archive:
        infos = archive.infolist()
        json_infos = [entry for entry in infos if entry.filename.lower().endswith(".json")]
        for entry in infos:
            if not entry.filename.lower().endswith(".jpg"):
                continue
            item_id = Path(entry.filename).stem.removesuffix("_m")
            encoded_signatures[item_id].add((entry.CRC, entry.file_size))
            image_entry_for_item.setdefault(item_id, entry.filename)

        canonical_seen: dict[tuple[str, ...], str] = {}
        duplicate_outfits: list[dict] = []
        for index, entry in enumerate(json_infos, start=1):
            try:
                row = json.loads(archive.read(entry))
                set_id = str(row["setId"])
                user_id = str(row["user"])
                items = row["items"]
                item_ids = [str(item["itemId"]) for item in items]
                if len(item_ids) < 2 or len(item_ids) != len(set(item_ids)):
                    raise ValueError("outfit has fewer than two distinct items")
            except Exception as exc:  # noqa: BLE001 - audit must retain malformed-entry evidence
                malformed.append({"entry": entry.filename, "error": repr(exc)})
                continue
            for item_id in item_ids:
                union_find.find(item_id)
            for item_id in item_ids[1:]:
                union_find.union(item_ids[0], item_id)
            canonical = tuple(sorted(item_ids))
            if canonical in canonical_seen:
                duplicate_outfits.append(
                    {"kept_set_id": canonical_seen[canonical], "duplicate_set_id": set_id}
                )
            else:
                canonical_seen[canonical] = set_id
            item_occurrences.update(item_ids)
            user_counts[user_id] += 1
            for item in items:
                raw_category = str(item.get("category x color", "")).split("×", 1)[0].strip()
                category_counts[raw_category or "<missing>"] += 1
            outfits.append(
                {
                    "set_id": set_id,
                    "user_id": user_id,
                    "item_ids": item_ids,
                    "item_categories": [
                        str(item.get("category x color", "")).split("×", 1)[0].strip()
                        or "<missing>"
                        for item in items
                    ],
                    "like_count": str(row.get("likeCount", "")),
                    "json_entry": entry.filename,
                }
            )
            if index % 25_000 == 0:
                print(f"audited {index:,}/{len(json_infos):,} outfits", flush=True)

    component_outfits: Counter[str] = Counter()
    component_items: Counter[str] = Counter()
    for row in outfits:
        component_outfits[union_find.find(row["item_ids"][0])] += 1
    for item_id in item_occurrences:
        component_items[union_find.find(item_id)] += 1
    component_sizes = sorted(
        (
            {
                "root": root,
                "outfits": count,
                "items": component_items[root],
            }
            for root, count in component_outfits.items()
        ),
        key=lambda row: (-row["outfits"], row["root"]),
    )
    missing_image_ids = sorted(set(item_occurrences) - set(image_entry_for_item))
    orphan_image_ids = sorted(set(image_entry_for_item) - set(item_occurrences))
    inconsistent_item_images = sorted(
        item_id for item_id, signatures in encoded_signatures.items() if len(signatures) > 1
    )
    signature_to_items: dict[tuple[int, int], list[str]] = defaultdict(list)
    for item_id, signatures in encoded_signatures.items():
        for signature in signatures:
            signature_to_items[signature].append(item_id)
    cross_id_encoded_groups = [
        sorted(values) for values in signature_to_items.values() if len(set(values)) > 1
    ]

    report = {
        "dataset": "IQON3000 original archive",
        "source_archive": str(args.archive.resolve()),
        "archive_sha256": sha256_file(args.archive),
        "audit_timestamp_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "zip_entry_count": len(infos),
        "json_entry_count": len(json_infos),
        "valid_outfit_count": len(outfits),
        "malformed_outfit_count": len(malformed),
        "unique_outfit_count": len(canonical_seen),
        "duplicate_outfit_count": len(duplicate_outfits),
        "unique_user_count": len(user_counts),
        "unique_item_count": len(item_occurrences),
        "image_entry_count": sum(1 for entry in infos if entry.filename.lower().endswith(".jpg")),
        "unique_image_item_id_count": len(image_entry_for_item),
        "missing_image_item_count": len(missing_image_ids),
        "orphan_image_item_count": len(orphan_image_ids),
        "items_with_inconsistent_encoded_files": len(inconsistent_item_images),
        "cross_item_encoded_duplicate_group_count_crc_size_screen": len(cross_id_encoded_groups),
        "component_count": len(component_sizes),
        "largest_component_outfits": component_sizes[0]["outfits"] if component_sizes else 0,
        "largest_component_items": component_sizes[0]["items"] if component_sizes else 0,
        "largest_component_fraction": (
            component_sizes[0]["outfits"] / len(outfits) if outfits else 0.0
        ),
        "outfit_length_counts": dict(
            sorted(Counter(len(row["item_ids"]) for row in outfits).items())
        ),
        "item_occurrence_counts": dict(sorted(Counter(item_occurrences.values()).items())),
        "category_occurrence_counts": category_counts.most_common(),
        "user_outfit_count_summary": {
            "min": min(user_counts.values()) if user_counts else 0,
            "max": max(user_counts.values()) if user_counts else 0,
        },
        "largest_components": component_sizes[:100],
        "malformed_examples": malformed[:100],
        "duplicate_outfit_examples": duplicate_outfits[:100],
        "missing_image_examples": missing_image_ids[:100],
        "orphan_image_examples": orphan_image_ids[:100],
        "inconsistent_item_image_examples": inconsistent_item_images[:100],
        "cross_item_encoded_duplicate_groups_screen_examples": cross_id_encoded_groups[:100],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.catalog_dir is not None:
        args.catalog_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(outfits).to_parquet(args.catalog_dir / "outfits.parquet", index=False)
        pd.DataFrame(
            {
                "item_id": sorted(image_entry_for_item),
                "zip_entry": [image_entry_for_item[item] for item in sorted(image_entry_for_item)],
                "encoded_signature_count": [
                    len(encoded_signatures[item]) for item in sorted(image_entry_for_item)
                ],
                "encoded_crc32": [
                    min(encoded_signatures[item])[0] for item in sorted(image_entry_for_item)
                ],
                "encoded_size": [
                    min(encoded_signatures[item])[1] for item in sorted(image_entry_for_item)
                ],
            }
        ).to_parquet(args.catalog_dir / "items.parquet", index=False)
    print(json.dumps({key: report[key] for key in (
        "valid_outfit_count", "unique_item_count", "unique_user_count",
        "duplicate_outfit_count", "missing_image_item_count", "component_count",
        "largest_component_outfits", "largest_component_fraction",
    )}, indent=2))


if __name__ == "__main__":
    main()
