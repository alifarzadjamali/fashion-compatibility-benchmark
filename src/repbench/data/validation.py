"""Dataset integrity checks that fail closed."""

from __future__ import annotations

import json
from collections import Counter

from .polyvore import PolyvoreDisjoint


def inspect_disjoint(dataset: PolyvoreDisjoint) -> dict:
    split_items = {split: dataset.item_ids(split) for split in dataset.SPLITS}
    overlaps = {
        f"{a}_{b}": sorted(split_items[a] & split_items[b])
        for i, a in enumerate(dataset.SPLITS)
        for b in dataset.SPLITS[i + 1 :]
    }
    report: dict = {
        "splits": {},
        "item_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "passes_item_disjointness": not any(overlaps.values()),
    }
    for split in dataset.SPLITS:
        outfits = dataset.outfits(split)
        cp = dataset.compatibility(split)
        fitb = dataset.fitb(split)
        missing = sorted(
            item for item in split_items[split] if not dataset.image_path(item).is_file()
        )
        labels = Counter(example.label for example in cp)
        answers = Counter(question.correct_index for question in fitb)
        report["splits"][split] = {
            "outfits": len(outfits),
            "unique_items": len(split_items[split]),
            "cp_examples": len(cp),
            "cp_positive": labels[1],
            "cp_negative": labels[0],
            "fitb_questions": len(fitb),
            "fitb_answer_index_counts": dict(sorted(answers.items())),
            "missing_images": len(missing),
        }
    return report


def validate_disjoint(dataset: PolyvoreDisjoint, check_images: bool = True) -> dict:
    report = inspect_disjoint(dataset)
    if not report["passes_item_disjointness"]:
        raise AssertionError(report["item_overlap_counts"])
    if check_images:
        missing = {split: row["missing_images"] for split, row in report["splits"].items()}
        if any(missing.values()):
            raise FileNotFoundError(f"Missing image counts: {missing}")
    return report


def inspect_clean_protocol(dataset: PolyvoreDisjoint) -> dict:
    """Exhaustive checks for the generated protocol's audited split-level leakage mechanisms."""
    report = inspect_disjoint(dataset)
    split_items = {split: dataset.item_ids(split) for split in dataset.SPLITS}
    metadata = json.loads(
        (dataset.root / "polyvore_item_metadata.json").read_text(encoding="utf-8")
    )
    global_outfits: set[tuple[str, ...]] = set()
    duplicate_outfits = 0
    split_checks = {}
    for split in dataset.SPLITS:
        outfits = dataset.outfits(split)
        positive_sets = {tuple(sorted(outfit.item_ids)) for outfit in outfits}
        duplicate_outfits += len(outfits) - len(positive_sets)
        duplicate_outfits += len(positive_sets & global_outfits)
        global_outfits.update(positive_sets)

        cp = dataset.compatibility(split)
        positives = [example for example in cp if example.label == 1]
        negatives = [example for example in cp if example.label == 0]
        category_matches = 0
        for positive, negative in zip(positives, negatives, strict=True):
            positive_categories = sorted(
                metadata[item_id]["semantic_category"].strip() for item_id in positive.item_ids
            )
            negative_categories = sorted(
                metadata[item_id]["semantic_category"].strip() for item_id in negative.item_ids
            )
            category_matches += positive_categories == negative_categories
        negative_sets = [tuple(sorted(example.item_ids)) for example in negatives]

        fitb = dataset.fitb(split)
        fitb_valid = 0
        answer_counts = Counter()
        for question in fitb:
            candidates = question.candidate_item_ids
            correct = candidates[question.correct_index]
            completed = tuple(sorted(question.question_item_ids + (correct,)))
            correct_category = metadata[correct]["semantic_category"].strip()
            valid = (
                len(candidates) == 4
                and len(set(candidates)) == 4
                and completed in positive_sets
                and all(candidate in split_items[split] for candidate in candidates)
                and all(
                    metadata[candidate]["semantic_category"].strip() == correct_category
                    for candidate in candidates
                )
                and all(
                    tuple(sorted(question.question_item_ids + (candidate,))) not in positive_sets
                    for index, candidate in enumerate(candidates)
                    if index != question.correct_index
                )
            )
            fitb_valid += valid
            answer_counts[question.correct_index] += 1
        split_checks[split] = {
            "cp_balanced": len(positives) == len(negatives) == len(outfits),
            "cp_category_and_length_matched": category_matches == len(outfits),
            "cp_unique_negatives": len(negative_sets) == len(set(negative_sets)),
            "cp_negatives_not_positives": not (set(negative_sets) & positive_sets),
            "fitb_all_valid": fitb_valid == len(fitb) == len(outfits),
            "fitb_answer_index_counts": dict(sorted(answer_counts.items())),
        }
    report["global_duplicate_outfits"] = duplicate_outfits
    report["clean_protocol_checks"] = split_checks
    report["passes_clean_protocol"] = (
        report["passes_item_disjointness"]
        and duplicate_outfits == 0
        and all(row["missing_images"] == 0 for row in report["splits"].values())
        and all(all(value for key, value in row.items() if key != "fitb_answer_index_counts") for row in split_checks.values())
    )
    return report


def validate_clean_protocol(dataset: PolyvoreDisjoint) -> dict:
    report = inspect_clean_protocol(dataset)
    if not report["passes_clean_protocol"]:
        raise AssertionError(report)
    return report
