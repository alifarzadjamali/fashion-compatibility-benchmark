"""Build a negative-generation bridge without changing historical positive splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.data.clean_protocol import SPLITS, file_sha256, generate_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/protocols/historical_polyvore_d_regenerated")
    )
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (args.root / "polyvore_item_metadata.json").read_text(encoding="utf-8")
    )
    manifest = {
        "protocol": "historical_polyvore_d_regenerated",
        "role": "secondary bridge isolating question-generation from repartitioning",
        "warning": "Retains historical cross-split item overlap; not leakage-free.",
        "seed": args.seed,
        "splits": {},
    }
    for split_index, split in enumerate(SPLITS):
        source = args.root / "disjoint" / f"{split}.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        outfit_path = args.output / f"{split}.json"
        outfit_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        cp_lines, fitb_rows, diagnostics = generate_questions(
            rows, metadata, args.seed + 10_000 * (split_index + 1)
        )
        cp_path = args.output / f"compatibility_{split}.txt"
        cp_path.write_text("\n".join(cp_lines) + "\n", encoding="utf-8")
        fitb_path = args.output / f"fill_in_blank_{split}.json"
        fitb_path.write_text(json.dumps(fitb_rows, indent=2), encoding="utf-8")
        manifest["splits"][split] = {
            "outfits": len(rows),
            "question_generation": diagnostics,
            "sha256": {
                outfit_path.name: file_sha256(outfit_path),
                cp_path.name: file_sha256(cp_path),
                fitb_path.name: file_sha256(fitb_path),
            },
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
