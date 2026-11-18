"""Resume verified image extraction from IQON's immutable source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from repbench.v2.state import require_budget, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    ids = sorted(
        {
            str(item["item_id"])
            for split in ("train", "valid", "test")
            for row in json.loads((args.protocol / f"{split}.json").read_text())
            for item in row["items"]
        }
    )
    audit = pd.read_parquet(args.catalog / "image_audit_resolved.parquet").set_index("item_id")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reused = byte_count = 0
    with zipfile.ZipFile(args.archive) as archive:
        for index, item_id in enumerate(ids):
            if index % 25000 == 0:
                require_budget(300)
                print(f"IQON image extraction: {index}/{len(ids)}", flush=True)
            row = audit.loc[item_id]
            destination = args.output / f"{item_id}.jpg"
            if destination.exists():
                if sha256(destination) != row.encoded_sha256:
                    raise ValueError(f"Existing image checksum mismatch: {item_id}")
                reused += 1
                continue
            data = archive.read(row.zip_entry)
            if hashlib.sha256(data).hexdigest() != row.encoded_sha256:
                raise ValueError(f"Archive item checksum mismatch: {item_id}")
            partial = destination.with_suffix(".partial")
            partial.write_bytes(data)
            partial.rename(destination)
            byte_count += len(data)
    write_json(
        args.report,
        {
            "stage": "IQON verified image extraction",
            "status": "PASS",
            "items": len(ids),
            "reused_verified_images": reused,
            "bytes_written": byte_count,
            "elapsed_seconds": time.perf_counter() - started,
            "image_audit_sha256": sha256(args.catalog / "image_audit_resolved.parquet"),
        },
    )


if __name__ == "__main__":
    main()
