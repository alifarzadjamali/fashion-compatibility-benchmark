"""Audit A100 reference counts and exact image duplication within and across tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


def signatures(path: Path) -> tuple[str, str]:
    encoded = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    return encoded, digest.hexdigest()


def referenced_ids(path: Path) -> tuple[set[str], int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    references = [item_id for row in rows for item_id in row["question"] + row["answers"]]
    return set(references), len(references)


def main() -> None:
    protocol_root = Path("data/protocols/a100")
    image_root = Path(".venv/a100/images")
    task_data = {}
    task_signatures = {}
    for task in ("LAT", "AAT"):
        ids, reference_instances = referenced_ids(protocol_root / f"{task.lower()}.json")
        paths = sorted(image_root / f"{item_id}.jpg" for item_id in ids)
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing referenced {task} image")
        values = [signatures(path) for path in paths]
        encoded = {value[0] for value in values}
        decoded = {value[1] for value in values}
        task_signatures[task] = (encoded, decoded)
        task_data[task] = {
            "questions": 100,
            "reference_instances": reference_instances,
            "referenced_file_paths": len(paths),
            "unique_encoded_images": len(encoded),
            "unique_decoded_rgb_images": len(decoded),
        }
    lat_encoded, lat_decoded = task_signatures["LAT"]
    aat_encoded, aat_decoded = task_signatures["AAT"]
    report = {
        "tasks": task_data,
        "cross_task_encoded_overlap": len(lat_encoded & aat_encoded),
        "cross_task_decoded_rgb_overlap": len(lat_decoded & aat_decoded),
        "global_referenced_file_paths": sum(row["referenced_file_paths"] for row in task_data.values()),
        "global_unique_encoded_images": len(lat_encoded | aat_encoded),
        "global_unique_decoded_rgb_images": len(lat_decoded | aat_decoded),
        "scope": "Exact encoded bytes and exact decoded RGB pixels; perceptual/transformed duplicates are not ruled out.",
    }
    Path("artifacts/a100_internal_image_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
