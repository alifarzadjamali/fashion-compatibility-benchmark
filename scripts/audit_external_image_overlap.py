"""Audit exact encoded and decoded-image overlap between A100 and training corpora."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image


def signatures(path: Path) -> tuple[str, str]:
    encoded = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    return encoded, digest.hexdigest()


def hash_paths(paths: list[Path]) -> tuple[set[str], set[str]]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        values = list(pool.map(signatures, paths, chunksize=128))
    return {value[0] for value in values}, {value[1] for value in values}


def main() -> None:
    a100_paths = sorted(Path(".venv/a100/images").glob("*.jpg"))
    polyvore_paths = sorted(Path("data/raw/polyvore_outfits/images").glob("*.jpg"))
    a100_encoded, a100_decoded = hash_paths(a100_paths)
    polyvore_encoded, polyvore_decoded = hash_paths(polyvore_paths)
    iqon = pd.read_parquet(
        ".venv/iqon_audit/catalog/image_audit_resolved.parquet",
        columns=["encoded_sha256", "decoded_rgb_sha256"],
    )
    iqon_encoded = set(iqon.encoded_sha256.dropna())
    iqon_decoded = set(iqon.decoded_rgb_sha256.dropna())
    report = {
        "a100_referenced_images": len(a100_paths),
        "polyvore_images": len(polyvore_paths),
        "iqon_catalog_images": len(iqon),
        "a100_polyvore_encoded_overlap": len(a100_encoded & polyvore_encoded),
        "a100_polyvore_decoded_overlap": len(a100_decoded & polyvore_decoded),
        "a100_iqon_encoded_overlap": len(a100_encoded & iqon_encoded),
        "a100_iqon_decoded_overlap": len(a100_decoded & iqon_decoded),
        "polyvore_iqon_encoded_overlap": len(polyvore_encoded & iqon_encoded),
        "polyvore_iqon_decoded_overlap": len(polyvore_decoded & iqon_decoded),
        "scope": "exact encoded bytes and exact decoded RGB pixels; transformed near-duplicates not ruled out",
    }
    Path("artifacts/a100_training_corpora_image_overlap_v2.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
