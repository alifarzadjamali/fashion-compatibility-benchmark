"""Verify IQON item images and derive immutable duplicate signatures."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


def decoded_digest(image: Image.Image) -> tuple[str, str]:
    rgb = image.convert("RGB")
    payload = struct.pack(">II", rgb.width, rgb.height) + rgb.tobytes()
    exact = hashlib.sha256(payload).hexdigest()
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    dhash = f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"
    return exact, dhash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    items = pd.read_parquet(args.items).sort_values("item_id")
    records: list[dict] = []
    errors: list[dict] = []
    started = time.time()
    with zipfile.ZipFile(args.archive) as archive:
        for index, row in enumerate(items.itertuples(index=False), start=1):
            try:
                raw = archive.read(row.zip_entry)
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    decoded_sha, dhash = decoded_digest(image)
                    records.append(
                        {
                            "item_id": str(row.item_id),
                            "zip_entry": row.zip_entry,
                            "encoded_sha256": hashlib.sha256(raw).hexdigest(),
                            "decoded_rgb_sha256": decoded_sha,
                            "dhash64": dhash,
                            "width": image.width,
                            "height": image.height,
                            "source_mode": image.mode,
                            "source_format": image.format,
                            "encoded_bytes": len(raw),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - corrupt files are expected audit evidence
                errors.append({"item_id": str(row.item_id), "zip_entry": row.zip_entry, "error": repr(exc)})
            if index % 25_000 == 0:
                print(f"validated {index:,}/{len(items):,} item images", flush=True)
    frame = pd.DataFrame(records)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False)
    report = {
        "archive": str(args.archive.resolve()),
        "catalog_item_count": len(items),
        "validated_image_count": len(frame),
        "error_count": len(errors),
        "error_examples": errors[:100],
        "encoded_duplicate_group_count": int((frame.groupby("encoded_sha256").size() > 1).sum()),
        "decoded_duplicate_group_count": int((frame.groupby("decoded_rgb_sha256").size() > 1).sum()),
        "dhash_collision_group_count_screen_only": int((frame.groupby("dhash64").size() > 1).sum()),
        "formats": frame.source_format.value_counts().to_dict(),
        "source_modes": frame.source_mode.value_counts().to_dict(),
        "resolution_counts_top20": {
            f"{width}x{height}": int(count)
            for (width, height), count in frame.groupby(["width", "height"]).size().sort_values(ascending=False).head(20).items()
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
