"""Build the POG clean protocol twice and require hash-identical builds (HANDOFF_V2 §4).

Phase 1 data construction only: no model is run and no test outcome is read.
Steps: (optional) stream ``user_data.txt`` once into a per-outfit click summary,
build into ``--output-a`` and ``--output-b`` from identical inputs, compare every
output file by SHA-256, and require both audits (zero cross-split item, exact-image
and outfit overlap; zero CP/FITB collisions) to pass. Nothing is overwritten.

Example:
  .venv/bin/python scripts/v2/build_pog_protocol.py \
      --items raw/item_data.txt --outfits raw/outfit_data.txt \
      --user-data user/user_data.txt --user-clicks-out work/pog_user_clicks.json \
      --output-a work/pog_clean_a --output-b work/pog_clean_b --report work/pog_build_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # works under python -I

from repbench.v2.pog import (
    CONSTRUCTION_SEED,
    MIN_CATEGORY_GROUPS,
    POG_TARGET_OUTFITS,
    POG_TARGET_SEED,
    construct_pog_clean,
    load_target_outfits,
    protocol_digest,
    scan_user_clicks,
)


def _stream_user_lines(path: Path, digest, counter: dict):
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            counter["bytes"] += len(raw)
            yield raw.decode("utf-8", errors="replace")


def build_user_clicks(args) -> float:
    started = time.monotonic()
    target, _ = load_target_outfits(args.items, args.outfits, args.target_seed, args.target_outfits)
    digest = hashlib.sha256()
    counter = {"bytes": 0}
    summary = scan_user_clicks(
        _stream_user_lines(args.user_data, digest, counter), {o for o, _ in target}
    )
    summary["source"] = {
        "name": args.user_data.name,
        "sha256": digest.hexdigest(),
        "bytes": counter["bytes"],
        "target_seed": args.target_seed,
        "target_outfits": len(target),
        "target_outfits_clicked": len(summary["outfits"]),
    }
    with args.user_clicks_out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=1, ensure_ascii=False)
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=Path, required=True, help="extracted item_data.txt")
    parser.add_argument("--outfits", type=Path, required=True, help="extracted outfit_data.txt")
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--download-manifest", type=Path, help="manifest.jsonl from pog_download.py"
    )
    users = parser.add_mutually_exclusive_group()
    users.add_argument("--user-data", type=Path, help="extracted user_data.txt (scanned once)")
    users.add_argument("--user-clicks", type=Path, help="existing click summary JSON")
    parser.add_argument(
        "--user-clicks-out", type=Path, help="where --user-data's summary is written"
    )
    parser.add_argument("--category-map", type=Path, help="JSON {cate_id: semantic category}")
    parser.add_argument("--category-map-name", default=None)
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--target-seed", type=int, default=POG_TARGET_SEED)
    parser.add_argument("--target-outfits", type=int, default=POG_TARGET_OUTFITS)
    parser.add_argument("--min-category-images", type=int, default=MIN_CATEGORY_GROUPS)
    args = parser.parse_args()

    outputs = [args.output_a, args.output_b, args.report]
    if args.user_data is not None:
        if args.user_clicks_out is None:
            parser.error("--user-data requires --user-clicks-out")
        outputs.append(args.user_clicks_out)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if args.output_a.resolve() == args.output_b.resolve():
        parser.error("--output-a and --output-b must differ")

    timings = {}
    user_clicks = args.user_clicks
    if args.user_data is not None:
        timings["user_scan_seconds"] = round(build_user_clicks(args), 1)
        user_clicks = args.user_clicks_out
        print(f"user click summary written: {user_clicks}", flush=True)

    category_map = None
    category_map_name = args.category_map_name or "identity_cate_id"
    if args.category_map is not None:
        category_map = json.loads(args.category_map.read_text(encoding="utf-8"))
        category_map_name = args.category_map_name or args.category_map.name

    manifests = {}
    for label, output in (("a", args.output_a), ("b", args.output_b)):
        started = time.monotonic()
        manifests[label] = construct_pog_clean(
            args.items,
            args.outfits,
            output,
            download_manifest=args.download_manifest,
            user_clicks=user_clicks,
            category_map=category_map,
            category_map_name=category_map_name,
            seed=args.seed,
            target_seed=args.target_seed,
            target_outfits=args.target_outfits,
            min_category_groups=args.min_category_images,
        )
        timings[f"build_{label}_seconds"] = round(time.monotonic() - started, 1)
        print(f"build {label} done in {timings[f'build_{label}_seconds']} s", flush=True)

    digest_a = protocol_digest(args.output_a)
    digest_b = protocol_digest(args.output_b)
    identical = digest_a == digest_b
    audits_pass = all(manifest["audit"]["passes"] for manifest in manifests.values())
    manifest = manifests["a"]
    report = {
        "stage": "POG clean protocol construction (two deterministic builds)",
        "status": "PASS" if identical and audits_pass else "FAIL",
        "builds_hash_identical": identical,
        "audits_pass": audits_pass,
        "file_sha256": digest_a,
        "mismatched_files": sorted(
            name
            for name in set(digest_a) | set(digest_b)
            if digest_a.get(name) != digest_b.get(name)
        ),
        "splits": {split: row["outfits"] for split, row in manifest["splits"].items()},
        "retention": manifest["retention"]["stages"],
        "pool_components": manifest["pool_components"],
        "audit_cross_split": manifest["audit"]["cross_split"],
        "timings": timings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "splits", "pool_components", "timings")}, indent=1
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("POG protocol builds differ or audits failed")


if __name__ == "__main__":
    main()
