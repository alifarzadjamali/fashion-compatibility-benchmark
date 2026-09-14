"""Audit exact file and decoded-pixel duplicates across clean item splits."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from repbench.data.polyvore import PolyvoreDisjoint


def hashes(path: Path) -> tuple[str, str]:
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixel = hashlib.sha256()
        pixel.update(str(rgb.size).encode())
        pixel.update(rgb.tobytes())
    return file_hash, pixel.hexdigest()


def main() -> None:
    dataset = PolyvoreDisjoint(
        Path("data/raw/polyvore_outfits"),
        Path("data/protocols/polyvore_d_clean"),
        "polyvore_d_clean",
    )
    assignments = {
        item_id: split for split in dataset.SPLITS for item_id in dataset.item_ids(split)
    }
    item_ids = sorted(assignments)
    paths = [dataset.image_path(item_id) for item_id in item_ids]
    with ThreadPoolExecutor(max_workers=16) as pool:
        values = list(pool.map(hashes, paths, chunksize=128))
    findings = {}
    for kind, hash_index in (("file_sha256", 0), ("decoded_rgb_sha256", 1)):
        groups = {}
        for item_id, value in zip(item_ids, values, strict=True):
            groups.setdefault(value[hash_index], []).append(item_id)
        cross_split = []
        for digest, ids in groups.items():
            splits = sorted({assignments[item_id] for item_id in ids})
            if len(splits) > 1:
                cross_split.append(
                    {"hash": digest, "item_ids": ids, "splits": splits}
                )
        findings[kind] = {
            "unique_hashes": len(groups),
            "duplicate_groups_anywhere": sum(len(ids) > 1 for ids in groups.values()),
            "cross_split_duplicate_groups": len(cross_split),
            "cross_split_item_instances": sum(len(row["item_ids"]) for row in cross_split),
            "cross_split_groups": cross_split,
        }
    report = {
        "protocol": "polyvore_d_clean",
        "unique_items": len(item_ids),
        "methods": {
            "file_sha256": "exact encoded-file equality",
            "decoded_rgb_sha256": "exact width, height, and decoded RGB pixel equality",
        },
        "findings": findings,
        "passes_exact_image_disjointness": all(
            row["cross_split_duplicate_groups"] == 0 for row in findings.values()
        ),
        "limitation": "Does not rule out resized, cropped, or visually near-duplicate images.",
    }
    Path("artifacts/clean_image_duplicate_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({**report, "findings": {k: {**v, "cross_split_groups": "stored in artifact"} for k, v in findings.items()}}, indent=2))


if __name__ == "__main__":
    main()
