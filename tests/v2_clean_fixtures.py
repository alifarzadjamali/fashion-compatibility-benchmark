"""Deterministic synthetic raw inputs for the three clean-protocol builders.

Used by tests/test_v2_image_equivalence.py to prove that the builders' default
(``image_equivalence=None``) output is byte-for-byte unchanged, and that a given
image-equivalence mapping joins near-duplicate items into one indivisible unit.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "valid", "test")


def _png(seed: int, size: int = 48) -> bytes:
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (4, 4, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    small.resize((size, size), Image.Resampling.NEAREST).save(buffer, format="PNG")
    return buffer.getvalue()


def file_digests(root: Path, skip=()) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(root).iterdir())
        if path.is_file() and path.name not in skip
    }


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ------------------------------------------------------------------ Polyvore


def polyvore_ids():
    tops = [str(100_000 + k) for k in range(60)]
    bottoms = [str(200_000 + k) for k in range(60)]
    shoes = [str(300_000 + k) for k in range(60)]
    return tops, bottoms, shoes


def make_polyvore_raw(root: Path) -> Path:
    """60 historical outfits (40/8/12); outfits 0-9 share one top (giant component);
    items 200001 and 200002 have byte-identical image files."""
    root = Path(root)
    (root / "disjoint").mkdir(parents=True)
    (root / "images").mkdir()
    tops, bottoms, shoes = polyvore_ids()
    rows = []
    for k in range(60):
        top = tops[0] if k < 10 else tops[k]
        rows.append(
            {
                "set_id": f"s{k:03d}",
                "items": [
                    {"item_id": top, "index": 1},
                    {"item_id": bottoms[k], "index": 2},
                    {"item_id": shoes[k], "index": 3},
                ],
            }
        )
    for split, part in zip(SPLITS, (rows[:40], rows[40:48], rows[48:]), strict=True):
        (root / "disjoint" / f"{split}.json").write_text(json.dumps(part), encoding="utf-8")
    metadata = {}
    for category, ids in (("top", tops), ("bottom", bottoms), ("shoes", shoes)):
        for item in ids:
            metadata[item] = {"semantic_category": category}
            seed = 200_001 if item == "200002" else int(item)
            (root / "images" / f"{item}.jpg").write_bytes(_png(seed))
    (root / "polyvore_item_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


# ------------------------------------------------------------------ IQON

IQON_TARGETS = {"train": 28, "valid": 4, "test": 8}


def make_iqon_raw(root: Path) -> dict:
    """70 outfits of three items; items i2_1 and i2_2 share a decoded-RGB hash."""
    import pandas as pd

    root = Path(root)
    catalog = root / "catalog"
    catalog.mkdir(parents=True)
    rng = random.Random(11)
    rows = []
    for k in range(70):
        shared = f"i0_{k % 5}" if k % 7 == 0 else f"i0_{100 + k}"
        rows.append(
            {
                "set_id": f"o{k:03d}",
                "user_id": f"u{k % 9}",
                "item_ids": [shared, f"i1_{k}", f"i2_{k}"],
                "item_categories": ["tops", "bottoms", "shoes"],
            }
        )
    rows.append(dict(rows[3], set_id="o999"))  # duplicate tuple
    rng.shuffle(rows)
    pd.DataFrame(rows).to_parquet(catalog / "outfits.parquet")
    items = sorted({item for row in rows for item in row["item_ids"]})
    audit = []
    for item in items:
        identity = "i2_1" if item == "i2_2" else item
        audit.append(
            {
                "item_id": item,
                "zip_entry": f"images/{item}.jpg",
                "encoded_sha256": hashlib.sha256(f"enc-{item}".encode()).hexdigest(),
                "decoded_rgb_sha256": hashlib.sha256(f"dec-{identity}".encode()).hexdigest(),
            }
        )
    pd.DataFrame(audit).to_parquet(catalog / "image_audit_resolved.parquet")
    (root / "source_audit.json").write_text(json.dumps({"fixture": True}), encoding="utf-8")
    (root / "archive.zip").write_bytes(b"fixture archive")
    return {
        "catalog_dir": catalog,
        "source_audit_path": root / "source_audit.json",
        "archive_path": root / "archive.zip",
    }


def iqon_digest(output: Path) -> dict:
    digests = file_digests(output, skip={"manifest.json"})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("source_archive")  # absolute path of the fixture directory
    digests["manifest.json(without source_archive)"] = canonical_sha256(manifest)
    return digests


# ------------------------------------------------------------------ POG

POG_CATEGORIES = ("top", "bottom", "shoe", "bag")
POG_TARGET = 300


def pog_item(category: str, index: int) -> str:
    return hashlib.md5(f"{category}-{index}".encode()).hexdigest()


def make_pog_raw(root: Path, with_images: bool = False) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    items = {c: [pog_item(c, i) for i in range(120)] for c in POG_CATEGORIES}
    with (root / "item_data.txt").open("w", encoding="utf-8") as stream:
        for category, ids in items.items():
            for item in ids:
                stream.write(f"{item},cate_{category},//img.example/{item}.png,title\n")
    with (root / "outfit_data.txt").open("w", encoding="utf-8") as stream:
        for index in range(340):
            length = rng.choice((3, 4))
            chosen = [
                rng.choice(items[c][: 60 if index % 3 else 120]) for c in POG_CATEGORIES[:length]
            ]
            outfit_id = hashlib.md5(f"outfit-{index}".encode()).hexdigest()
            stream.write(f"{outfit_id},{';'.join(chosen)}\n")
    all_items = sorted(i for ids in items.values() for i in ids)
    twin_a, twin_b = all_items[0], all_items[1]
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for item in all_items:
            path = f"images/{item[:2]}/{item}.png"
            if with_images:
                data = _png(int(item[:8], 16) if item != twin_b else int(twin_a[:8], 16))
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(data)
                sha = hashlib.sha256(data).hexdigest()
            else:
                sha = hashlib.sha256((twin_a if item == twin_b else item).encode()).hexdigest()
            stream.write(json.dumps({"item_id": item, "path": path, "sha256": sha}) + "\n")
    return {
        "items_path": root / "item_data.txt",
        "outfits_path": root / "outfit_data.txt",
        "download_manifest": root / "manifest.jsonl",
        "items": items,
    }
