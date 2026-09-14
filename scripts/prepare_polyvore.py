from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".venv/cache/huggingface").resolve()))

from huggingface_hub import snapshot_download

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_disjoint

KAGGLE_PREFIX = "polyvore-outfit-dataset/polyvore_outfits/"


def extract_kaggle_archive(archive: Path, root: Path) -> None:
    """Extract only official disjoint protocol files and their referenced images."""
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        item_ids: set[str] = set()
        for split in ("train", "valid", "test"):
            member = f"{KAGGLE_PREFIX}disjoint/{split}.json"
            outfits = json.loads(source.read(member))
            item_ids.update(str(item["item_id"]) for outfit in outfits for item in outfit["items"])

        selected = [
            name
            for name in source.namelist()
            if name.startswith(f"{KAGGLE_PREFIX}disjoint/")
            or name
            in {
                f"{KAGGLE_PREFIX}categories.csv",
                f"{KAGGLE_PREFIX}polyvore_item_metadata.json",
                f"{KAGGLE_PREFIX}polyvore_outfit_titles.json",
            }
        ]
        selected.extend(f"{KAGGLE_PREFIX}images/{item_id}.jpg" for item_id in sorted(item_ids))
        archive_names = set(source.namelist())
        missing_members = [name for name in selected if name not in archive_names]
        if missing_members:
            raise FileNotFoundError(
                f"Archive lacks {len(missing_members)} required members; first={missing_members[:10]}"
            )
        for index, member in enumerate(selected, 1):
            relative = Path(member.removeprefix(KAGGLE_PREFIX))
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
            if index % 25000 == 0:
                print(f"Extracted {index}/{len(selected)} files", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--repo-id", default="mvasil/polyvore-outfits")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--kaggle-archive", type=Path)
    args = parser.parse_args()
    if args.kaggle_archive:
        extract_kaggle_archive(args.kaggle_archive, args.root)
    elif not args.no_download:
        args.root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.root,
            allow_patterns=["disjoint/**", "data/disjoint/**", "*.md", "*.csv", "*.json"],
        )
    report = inspect_disjoint(PolyvoreDisjoint(args.root))
    output = Path("artifacts/data_integrity_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passes_item_disjointness"]:
        raise SystemExit(
            "Dataset failed the mandatory item-disjointness gate; see integrity report"
        )
    if not args.skip_images and any(row["missing_images"] for row in report["splits"].values()):
        raise SystemExit("Dataset has missing images; see integrity report")


if __name__ == "__main__":
    main()
