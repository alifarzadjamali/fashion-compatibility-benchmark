"""Validate CP/FITB question safeguards, including exact-image equivalence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

SPLITS = ("train", "valid", "test")


def decoded_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def item_hashes(dataset: str, item_ids: set[str]) -> dict[str, str]:
    if dataset == "polyvore_d_clean":
        root = Path("data/raw/polyvore_outfits/images")
        ordered = sorted(item_ids)
        with ThreadPoolExecutor(max_workers=16) as pool:
            values = pool.map(lambda item_id: decoded_hash(root / f"{item_id}.jpg"), ordered, chunksize=128)
            return dict(zip(ordered, values, strict=True))
    audit = pd.read_parquet(
        ".venv/iqon_audit/catalog/image_audit_resolved.parquet",
        columns=["item_id", "decoded_rgb_sha256"],
    )
    audit.item_id = audit.item_id.astype(str)
    values = audit[audit.item_id.isin(item_ids)].set_index("item_id").decoded_rgb_sha256.to_dict()
    if set(values) != item_ids:
        raise ValueError(f"Missing IQON exact-image hashes: {len(item_ids - set(values))}")
    return values


def audit_split(root: Path, split: str, hashes: dict[str, str]) -> dict:
    outfits = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    key_to_id = {
        f"{row['set_id']}_{int(item['index'])}": str(item["item_id"])
        for row in outfits
        for item in row["items"]
    }
    positive_ids = {tuple(sorted(str(item["item_id"]) for item in row["items"])) for row in outfits}
    positive_images = {tuple(sorted(hashes[item_id] for item_id in values)) for values in positive_ids}
    cp_lines = [line.split() for line in (root / f"compatibility_{split}.txt").read_text(encoding="utf-8").splitlines()]
    negatives = [tuple(sorted(key_to_id[key] for key in line[1:])) for line in cp_lines if line[0] == "0"]
    negative_images = [tuple(sorted(hashes[item_id] for item_id in values)) for values in negatives]
    fitb = json.loads((root / f"fill_in_blank_{split}.json").read_text(encoding="utf-8"))
    known_positive_distractors = 0
    duplicate_candidate_ids = 0
    duplicate_candidate_images = 0
    distractor_usage = Counter()
    for row in fitb:
        question = [key_to_id[key] for key in row["question"]]
        answers = [key_to_id[key] for key in row["answers"]]
        # The generator balances the correct answer by question order. Infer it from the
        # only candidate that completes a known source outfit; safeguards ensure uniqueness.
        known = [tuple(sorted(question + [candidate])) in positive_ids for candidate in answers]
        if sum(known) != 1:
            raise ValueError(f"FITB question does not have exactly one known-positive completion in {split}")
        correct_index = known.index(True)
        duplicate_candidate_ids += len(set(answers)) != len(answers)
        duplicate_candidate_images += len({hashes[item_id] for item_id in answers}) != len(answers)
        for index, candidate in enumerate(answers):
            if index != correct_index:
                distractor_usage[candidate] += 1
                known_positive_distractors += tuple(sorted(question + [candidate])) in positive_ids
    return {
        "positive_outfits": len(positive_ids),
        "cp_negatives": len(negatives),
        "cp_unique_negative_item_sets": len(set(negatives)),
        "cp_negative_item_sets_matching_positive": len(set(negatives) & positive_ids),
        "cp_negative_exact_image_sets_matching_positive": sum(value in positive_images for value in negative_images),
        "fitb_questions": len(fitb),
        "fitb_questions_with_duplicate_candidate_ids": duplicate_candidate_ids,
        "fitb_questions_with_duplicate_exact_images": duplicate_candidate_images,
        "fitb_distractor_completions_matching_known_positive": known_positive_distractors,
        "fitb_max_distractor_reuse_across_questions": max(distractor_usage.values()),
    }


def main() -> None:
    report = {}
    for dataset in ("polyvore_d_clean", "iqon3000_clean"):
        root = Path("data/protocols") / dataset
        rows = [
            row
            for split in SPLITS
            for row in json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
        ]
        ids = {str(item["item_id"]) for row in rows for item in row["items"]}
        hashes = item_hashes(dataset, ids)
        report[dataset] = {split: audit_split(root, split, hashes) for split in SPLITS}
    Path("artifacts/question_construction_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
