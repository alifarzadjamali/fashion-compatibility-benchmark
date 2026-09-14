"""Regenerate only CP negatives and FITB candidates on fixed item partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from repbench.data.clean_protocol import SPLITS, file_sha256, generate_questions


def polyvore_equivalence(protocol: Path, image_root: Path) -> dict[str, str]:
    """Map every selected Polyvore item to exact encoded bytes.

    The clean split builder makes exact-image groups split-indivisible.  Construction
    stability must additionally avoid presenting an exact copy of the held-out answer
    as a distractor inside a split.
    """
    item_ids = {
        str(item["item_id"])
        for split in SPLITS
        for row in json.loads((protocol / f"{split}.json").read_text(encoding="utf-8"))
        for item in row["items"]
    }
    result = {}
    for item_id in sorted(item_ids):
        path = image_root / "images" / f"{item_id}.jpg"
        result[item_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("iqon3000_clean", "polyvore_d_clean"), required=True)
    parser.add_argument("--polyvore-metadata", type=Path, default=Path("data/raw/polyvore_outfits/polyvore_item_metadata.json"))
    parser.add_argument("--polyvore-root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--iqon-image-audit", type=Path, default=Path(".venv/iqon_audit/catalog/image_audit_resolved.parquet"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    base_hash = file_sha256(args.base_protocol / "manifest.json")
    polyvore_hashes = (
        polyvore_equivalence(args.base_protocol, args.polyvore_root)
        if args.dataset == "polyvore_d_clean"
        else None
    )
    for seed in args.seeds:
        output = args.output_root / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"dataset": args.dataset, "construction_seed": seed, "base_manifest_sha256": base_hash, "splits": {}}
        for split_index, split in enumerate(SPLITS):
            rows = json.loads((args.base_protocol / f"{split}.json").read_text(encoding="utf-8"))
            if args.dataset == "iqon3000_clean":
                metadata = {
                    str(item["item_id"]): {"semantic_category": item["semantic_category"]}
                    for row in rows for item in row["items"]
                }
                selected = set(metadata)
                audit = pd.read_parquet(args.iqon_image_audit).set_index("item_id")
                equivalence = audit.loc[list(selected)].decoded_rgb_sha256.to_dict()
            else:
                metadata = json.loads(args.polyvore_metadata.read_text(encoding="utf-8"))
                selected = {str(item["item_id"]) for row in rows for item in row["items"]}
                equivalence = {item_id: polyvore_hashes[item_id] for item_id in selected}
            cp, fitb, diagnostics = generate_questions(
                rows, metadata, seed + 10_000 * (split_index + 1), equivalence
            )
            shutil.copy2(args.base_protocol / f"{split}.json", output / f"{split}.json")
            group_source = args.base_protocol / f"group_ids_{split}.json"
            if group_source.exists():
                shutil.copy2(group_source, output / group_source.name)
            cp_path = output / f"compatibility_{split}.txt"
            cp_path.write_text("\n".join(cp) + "\n", encoding="utf-8")
            fitb_path = output / f"fill_in_blank_{split}.json"
            fitb_path.write_text(json.dumps(fitb, indent=2), encoding="utf-8")
            manifest["splits"][split] = {
                "question_generation": diagnostics,
                "cp_sha256": file_sha256(cp_path),
                "fitb_sha256": file_sha256(fitb_path),
            }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "manifest_sha256": file_sha256(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
