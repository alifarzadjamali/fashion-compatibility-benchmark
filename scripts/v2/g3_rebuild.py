"""Owner-amended G3 split reconstruction checks; not original-hash reproduction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from repbench.v2.state import require_budget, require_gate, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["polyvore_d_clean", "iqon3000_clean"], required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    require_gate(args.g2, "G2")
    require_budget(1800)
    reference_path = Path("configs/v1_published_reference.json")
    reference = json.loads(reference_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifests, timings = [], []
    for attempt in ("rebuild_a", "rebuild_b"):
        start = time.perf_counter()
        destination = args.output / attempt
        if args.dataset == "polyvore_d_clean":
            if not args.raw_root:
                raise ValueError("--raw-root is required")
            from repbench.data.clean_protocol import construct_clean_protocol

            manifest = construct_clean_protocol(args.raw_root, destination, seed=20260912)
        else:
            if not all([args.catalog, args.source_audit, args.archive]):
                raise ValueError("IQON needs --catalog, --source-audit and --archive")
            from repbench.data.iqon import construct_iqon_clean

            manifest = construct_iqon_clean(
                args.catalog, destination, args.source_audit, args.archive, seed=20260912
            )
        timings.append(time.perf_counter() - start)
        manifests.append(manifest)
    hashes = []
    for folder in (args.output / "rebuild_a", args.output / "rebuild_b"):
        hashes.append({p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()})
    counts = [[m["splits"][s]["outfits"] for s in ("train", "valid", "test")] for m in manifests]
    expected = reference["outfit_counts_train_valid_test"][args.dataset]
    passed = hashes[0] == hashes[1] and all(c == expected for c in counts)
    write_json(
        args.output / "reconstruction_check.json",
        {
            "gate_part": "G3 revised split reconstruction (metric reproduction still required)",
            "dataset": args.dataset,
            "status": "PASS" if passed else "FAIL",
            "counts": counts,
            "published_expected_counts": expected,
            "two_rebuild_hashes_match": hashes[0] == hashes[1],
            "hashes": hashes,
            "rebuild_seconds": timings,
            "published_reference_sha256": sha256(reference_path),
            "original_split_identity": "UNVERIFIABLE: original files/hashes unavailable",
        },
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
