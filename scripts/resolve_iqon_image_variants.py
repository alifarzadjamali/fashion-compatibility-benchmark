"""Resolve corrupt canonical IQON entries from repeated item-image variants."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def signatures(raw: bytes) -> dict:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
        exact = hashlib.sha256(struct.pack(">II", rgb.width, rgb.height) + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
        return {
            "encoded_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_rgb_sha256": exact,
            "dhash64": f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}",
            "width": image.width,
            "height": image.height,
            "source_mode": image.mode,
            "source_format": image.format,
            "encoded_bytes": len(raw),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed = {row["item_id"] for row in json.loads(args.report.read_text())["error_examples"]}
    candidates: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            if not entry.filename.lower().endswith(".jpg"):
                continue
            item_id = Path(entry.filename).stem.removesuffix("_m")
            if item_id in failed:
                candidates[item_id].append(entry.filename)
        resolved = []
        unresolved = []
        for item_id in sorted(failed):
            errors = []
            for entry in sorted(candidates[item_id]):
                try:
                    resolved.append({"item_id": item_id, "zip_entry": entry, **signatures(archive.read(entry))})
                    break
                except Exception as exc:  # noqa: BLE001 - try every archived variant
                    errors.append({"entry": entry, "error": repr(exc)})
            else:
                unresolved.append({"item_id": item_id, "attempts": errors})
    frame = pd.read_parquet(args.image_audit)
    combined = pd.concat([frame, pd.DataFrame(resolved)], ignore_index=True).sort_values("item_id")
    if combined.item_id.duplicated().any():
        raise ValueError("Variant resolution produced duplicate item IDs")
    combined.to_parquet(args.output, index=False)
    print(json.dumps({"failed_canonical": len(failed), "resolved": len(resolved), "unresolved": unresolved}, indent=2))


if __name__ == "__main__":
    main()
