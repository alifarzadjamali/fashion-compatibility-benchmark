"""Read-only preparation and validation of the official A100 archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

AAT_DIMENSIONS = {
    "Color": (1, 20),
    "Style": (21, 52),
    "Occasion": (53, 67),
    "Season": (68, 79),
    "Material": (80, 91),
    "Balance": (92, 100),
}


def _item_id(task: str, reference: str) -> str:
    suffix = reference.rsplit("/", 1)[-1].split("_", 1)[-1] if task == "LAT" else reference.rsplit("_", 1)[-1]
    return f"{task}_{suffix}"


def _image_basename(task: str, reference: str) -> str:
    return (reference.split("_", 1)[-1] if task == "LAT" else reference.rsplit("_", 1)[-1]) + ".jpg"


def prepare_a100(archive_path: Path, output_root: Path, images_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    images_root.mkdir(parents=True, exist_ok=False)
    manifest = {"archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "tasks": {}}
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if "__MACOSX" not in name]
        for task in ("AAT", "LAT"):
            label_name = next(name for name in names if name.endswith(f"/{task}/label/{task}.json"))
            image_entries = {
                Path(name).name: name
                for name in names
                if f"/{task}/image/" in name and name.lower().endswith(".jpg")
            }
            raw_rows = json.loads(archive.read(label_name))
            rows = []
            referenced = set()
            for row in raw_rows:
                number = int(row["quesion_num"])
                question = [_item_id(task, value) for value in row["question"]]
                answers = [_item_id(task, value) for value in row["answers"]]
                for reference, item_id in zip(row["question"] + row["answers"], question + answers, strict=True):
                    basename = _image_basename(task, reference)
                    if basename not in image_entries:
                        raise FileNotFoundError(f"A100 {task} reference lacks image: {reference}")
                    destination = images_root / f"{item_id}.jpg"
                    if not destination.exists():
                        destination.write_bytes(archive.read(image_entries[basename]))
                    with Image.open(destination) as image:
                        image.verify()
                    referenced.add(item_id)
                converted = {
                    "question_num": number,
                    "question": question,
                    "answers": answers,
                    "archive_gt_index": int(row["gt"]) - 1,
                }
                if task == "LAT":
                    distribution = [float(value) for value in row["gt_distribution"]]
                    if len(distribution) != 5 or abs(sum(distribution) - 1.0) > 1e-6:
                        raise ValueError(f"Invalid LAT distribution for question {number}")
                    converted["gt_distribution"] = distribution
                    converted["majority_gt_index"] = int(max(range(5), key=distribution.__getitem__))
                else:
                    converted["dimension"] = next(
                        name for name, (lower, upper) in AAT_DIMENSIONS.items() if lower <= number <= upper
                    )
                rows.append(converted)
            path = output_root / f"{task.lower()}.json"
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            manifest["tasks"][task] = {
                "questions": len(rows),
                "candidates_per_question": sorted({len(row["answers"]) for row in rows}),
                "referenced_images": len(referenced),
                "archive_images": len(image_entries),
                "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    lat = json.loads((output_root / "lat.json").read_text())
    manifest["lat_archive_majority_disagreements"] = [
        row["question_num"] for row in lat if row["archive_gt_index"] != row["majority_gt_index"]
    ]
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
