from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.data.iqon import construct_iqon_clean, extract_selected_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", type=Path, default=Path(".venv/iqon_audit/catalog"))
    parser.add_argument("--archive", type=Path, default=Path(".venv/iqon_audit/IQON3000.zip"))
    parser.add_argument("--source-audit", type=Path, default=Path("artifacts/iqon3000_source_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("data/protocols/iqon3000_clean"))
    parser.add_argument("--images-root", type=Path, default=Path(".venv/iqon3000_clean/images"))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--extract-images", action="store_true")
    args = parser.parse_args()
    manifest = construct_iqon_clean(
        args.catalog_dir, args.output, args.source_audit, args.archive, args.seed
    )
    print(json.dumps({"protocol": manifest["protocol"], "splits": manifest["splits"]}, indent=2))
    if args.extract_images:
        print(json.dumps(extract_selected_images(args.output, args.catalog_dir, args.archive, args.images_root), indent=2))


if __name__ == "__main__":
    main()
