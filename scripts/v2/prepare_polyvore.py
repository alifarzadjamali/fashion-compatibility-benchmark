"""Extract the versioned Polyvore archive without touching any v1 directory."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

from repbench.v2.state import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Use a fresh extraction destination/report")
    if sha256(args.archive) != args.expected_sha256:
        raise ValueError("Archive checksum mismatch")
    source_path = Path(__file__).resolve().parents[1] / "prepare_polyvore.py"
    spec = importlib.util.spec_from_file_location("legacy_polyvore_preparation", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    module.extract_kaggle_archive(args.archive, args.output)
    count = sum(1 for _ in (args.output / "images").glob("*.jpg"))
    write_json(
        args.report,
        {
            "stage": "polyvore extraction",
            "image_count": count,
            "expected_v1_image_count": 152785,
            "status": "PASS" if count == 152785 else "FAIL",
            "elapsed_seconds": time.perf_counter() - started,
            "archive_sha256": args.expected_sha256,
            "extractor_sha256": sha256(source_path),
        },
    )
    raise SystemExit(0 if count == 152785 else 1)


if __name__ == "__main__":
    main()
