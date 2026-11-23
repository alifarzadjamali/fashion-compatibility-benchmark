"""Prepare the official A100 archive as a v2 question protocol directory (read-only input).

Output (refuses to overwrite; built in ``<output>.partial`` then renamed):
  images/<item>.jpg        archive JPEG bytes (non-JPEG/alpha images -> PNG on white)
  image_paths.json         item -> file under images/ (read by scripts/v2/extract_v2.py)
  questions.json           canonical questions (LAT majority + archive gt, AAT gt, facet)
  lat.json, aat.json       v1 schema (repbench.data.a100.prepare_a100), for the MLLM runner
  test.json, fill_in_blank_test.json, compatibility_test.txt (empty)
                           Polyvore-format split 'test' so extract_v2.py can embed A100
                           (``--dataset a100 --splits test``: frozen-protocol guarded)
  manifest.json            archive sha256, counts, label rules, file hashes

Usage:
  python scripts/v2/prepare_a100_v2.py --archive ~/repbench-mirror/a100_audit/A100.zip \
      --output "$REPBENCH_DATA_ROOT/a100_v2"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.v2.a100 import A100_ARCHIVE_SHA256, prepare_a100_v2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=A100_ARCHIVE_SHA256,
                        help="archive checksum to enforce ('none' disables; fixtures only)")
    args = parser.parse_args(argv)
    expected = None if args.expected_sha256.lower() == "none" else args.expected_sha256
    manifest = prepare_a100_v2(args.archive.expanduser(), args.output.expanduser(),
                               expected_sha256=expected)
    print(json.dumps({"output": str(args.output), "archive_sha256": manifest["archive_sha256"],
                      "tasks": manifest["tasks"], "items": manifest["items"],
                      "composited": len(manifest["composited_items"]),
                      "lat_majority_ties": manifest["lat_majority_ties"]}, indent=1))


if __name__ == "__main__":
    main()
