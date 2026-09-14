"""Deterministic construction of an item-disjoint Polyvore-D protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Component:
    outfits: tuple[dict, ...]
    item_ids: frozenset[str]

    @property
    def size(self) -> int:
        return len(self.outfits)

    @property
    def stable_key(self) -> str:
        return min(str(row["set_id"]) for row in self.outfits)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def canonical_outfit(row: dict) -> tuple[str, ...]:
    return tuple(sorted(str(item["item_id"]) for item in row["items"]))


def deduplicate_outfits(rows_by_source: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    seen: dict[tuple[str, ...], dict] = {}
    duplicates: list[dict] = []
    for source in SPLITS:
        for row in rows_by_source[source]:
            enriched = {**row, "_historical_source": source}
            key = canonical_outfit(row)
            if key in seen:
                duplicates.append(
                    {
                        "kept_set_id": str(seen[key]["set_id"]),
                        "removed_set_id": str(row["set_id"]),
                        "historical_source": source,
                        "item_ids": list(key),
                    }
                )
            else:
                seen[key] = enriched
    return list(seen.values()), duplicates


def build_components(
    outfits: list[dict], equivalent_item_groups: list[list[str]] | None = None
) -> list[Component]:
    union_find = UnionFind()
    for row in outfits:
        item_ids = [str(item["item_id"]) for item in row["items"]]
        for item_id in item_ids:
            union_find.find(item_id)
        for item_id in item_ids[1:]:
            union_find.union(item_ids[0], item_id)
    for group in equivalent_item_groups or []:
        for item_id in group[1:]:
            union_find.union(group[0], item_id)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in outfits:
        grouped[union_find.find(str(row["items"][0]["item_id"]))].append(row)
    components = []
    for rows in grouped.values():
        item_ids = frozenset(
            str(item["item_id"]) for row in rows for item in row["items"]
        )
        components.append(Component(tuple(rows), item_ids))
    return sorted(components, key=lambda value: (-value.size, value.stable_key))


def exact_file_duplicate_groups(raw_root: Path, item_ids: set[str]) -> list[list[str]]:
    def digest(item_id: str) -> tuple[str, str]:
        path = raw_root / "images" / f"{item_id}.jpg"
        return item_id, hashlib.sha256(path.read_bytes()).hexdigest()

    groups: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for item_id, value in pool.map(digest, sorted(item_ids), chunksize=128):
            groups[value].append(item_id)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def proportional_targets(total: int, original_counts: dict[str, int]) -> dict[str, int]:
    denominator = sum(original_counts.values())
    exact = {split: total * original_counts[split] / denominator for split in SPLITS}
    targets = {split: int(np.floor(exact[split])) for split in SPLITS}
    remainder = total - sum(targets.values())
    order = sorted(SPLITS, key=lambda split: (-(exact[split] - targets[split]), split))
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def assign_components(
    components: list[Component], targets: dict[str, int]
) -> dict[str, list[dict]]:
    """Deterministic best-fit assignment, reserving the giant component for training."""
    if not components:
        raise ValueError("Cannot split an empty outfit graph")
    assigned: dict[str, list[dict]] = {split: [] for split in SPLITS}
    remaining = dict(targets)
    largest = components[0]
    if largest.size > remaining["train"]:
        raise ValueError("Largest connected component does not fit the requested train split")
    assigned["train"].extend(largest.outfits)
    remaining["train"] -= largest.size
    for component in components[1:]:
        feasible = [split for split in SPLITS if remaining[split] >= component.size]
        if not feasible:
            raise ValueError(
                f"Component of {component.size} outfits cannot fit remaining targets {remaining}"
            )
        # Best fit minimizes unused capacity; fixed split ordering resolves exact ties.
        chosen = min(feasible, key=lambda split: (remaining[split] - component.size, SPLITS.index(split)))
        assigned[chosen].extend(component.outfits)
        remaining[chosen] -= component.size
    if any(remaining.values()):
        raise AssertionError(f"Component assignment did not meet targets: {remaining}")
    return assigned


def _item_key_maps(outfits: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    item_to_key: dict[str, str] = {}
    item_to_set: dict[str, str] = {}
    for row in sorted(outfits, key=lambda value: str(value["set_id"])):
        set_id = str(row["set_id"])
        for item in row["items"]:
            item_id = str(item["item_id"])
            item_to_key.setdefault(item_id, f"{set_id}_{int(item['index'])}")
            item_to_set.setdefault(item_id, set_id)
    return item_to_key, item_to_set


def generate_questions(
    outfits: list[dict],
    metadata: dict[str, dict],
    seed: int,
    item_equivalence: dict[str, str] | None = None,
) -> tuple[list[str], list[dict], dict]:
    """Create length/category-matched CP negatives and four-way FITB questions."""
    rng = np.random.default_rng(seed)
    item_to_key, item_to_set = _item_key_maps(outfits)
    categories = {
        item_id: str(metadata[item_id]["semantic_category"]).strip()
        for item_id in item_to_key
    }
    equivalence = item_equivalence or {item_id: item_id for item_id in item_to_key}
    if set(equivalence) != set(item_to_key):
        raise ValueError("Item-equivalence mapping must cover exactly the protocol item IDs")
    if any(not category for category in categories.values()):
        raise ValueError("Every protocol item must have a non-empty semantic category")
    pools: dict[str, list[str]] = defaultdict(list)
    for item_id, category in categories.items():
        pools[category].append(item_id)
    for category, values in pools.items():
        pools[category] = list(np.asarray(sorted(values))[rng.permutation(len(values))])
    cursors = Counter()

    def next_item(category: str, excluded: set[str], excluded_sets: set[str]) -> str:
        pool = pools[category]
        excluded_groups = {equivalence[item_id] for item_id in excluded}
        for _ in range(len(pool)):
            candidate = pool[cursors[category] % len(pool)]
            cursors[category] += 1
            if (
                candidate not in excluded
                and equivalence[candidate] not in excluded_groups
                and item_to_set[candidate] not in excluded_sets
            ):
                return candidate
        raise ValueError(f"No eligible candidate in semantic category {category!r}")

    positive_sets = {canonical_outfit(row) for row in outfits}
    negative_sets: set[tuple[str, ...]] = set()
    positive_lines: list[str] = []
    negative_lines: list[str] = []
    fitb_rows: list[dict] = []
    negative_usage = Counter()
    distractor_usage = Counter()
    ordered = sorted(outfits, key=lambda value: str(value["set_id"]))
    fitb_order = list(np.asarray(ordered, dtype=object)[rng.permutation(len(ordered))])

    for row in ordered:
        set_id = str(row["set_id"])
        own_ids = [str(item["item_id"]) for item in row["items"]]
        own_keys = [f"{set_id}_{int(item['index'])}" for item in row["items"]]
        positive_lines.append("1 " + " ".join(own_keys))
        for _ in range(100):
            selected: list[str] = []
            for item_id in own_ids:
                selected.append(
                    next_item(categories[item_id], set(own_ids) | set(selected), {set_id})
                )
            candidate_set = tuple(sorted(selected))
            if candidate_set not in positive_sets and candidate_set not in negative_sets:
                break
        else:
            raise RuntimeError(f"Could not construct a unique negative for set {set_id}")
        negative_sets.add(candidate_set)
        negative_usage.update(selected)
        negative_lines.append("0 " + " ".join(item_to_key[item_id] for item_id in selected))

    for question_index, row in enumerate(fitb_order):
        set_id = str(row["set_id"])
        items = row["items"]
        blank = int(rng.integers(0, len(items)))
        correct_id = str(items[blank]["item_id"])
        question_ids = [str(item["item_id"]) for i, item in enumerate(items) if i != blank]
        distractors: list[str] = []
        attempts = 0
        while len(distractors) < 3:
            attempts += 1
            if attempts > len(pools[categories[correct_id]]) * 2:
                raise RuntimeError(f"Could not create FITB distractors for set {set_id}")
            candidate = next_item(
                categories[correct_id], set(question_ids) | {correct_id} | set(distractors), {set_id}
            )
            if tuple(sorted(question_ids + [candidate])) in positive_sets:
                continue
            distractors.append(candidate)
        distractor_usage.update(distractors)
        correct_index = question_index % 4
        answers = [item_to_key[item_id] for item_id in distractors]
        answers.insert(correct_index, f"{set_id}_{int(items[blank]['index'])}")
        fitb_rows.append(
            {
                "question": [
                    f"{set_id}_{int(item['index'])}" for i, item in enumerate(items) if i != blank
                ],
                "blank_position": int(items[blank]["index"]),
                "answers": answers,
            }
        )

    diagnostics = {
        "cp_negative_strategy": "full semantic-category-matched corruption",
        "cp_positive_count": len(positive_lines),
        "cp_negative_count": len(negative_lines),
        "cp_unique_negative_count": len(negative_sets),
        "cp_negative_item_usage_max": max(negative_usage.values()),
        "fitb_strategy": "one deterministic blank; three same-category within-split distractors",
        "fitb_questions": len(fitb_rows),
        "fitb_answer_index_counts": dict(sorted(Counter(i % 4 for i in range(len(fitb_rows))).items())),
        "fitb_distractor_item_usage_max": max(distractor_usage.values()),
    }
    return positive_lines + negative_lines, fitb_rows, diagnostics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def construct_clean_protocol(
    raw_root: Path, output_root: Path, seed: int = 20260912
) -> dict:
    historical_root = raw_root / "disjoint"
    if output_root.resolve() == historical_root.resolve():
        raise ValueError("Refusing to overwrite the historical protocol")
    rows_by_source = {
        split: json.loads((historical_root / f"{split}.json").read_text(encoding="utf-8"))
        for split in SPLITS
    }
    outfits, duplicates = deduplicate_outfits(rows_by_source)
    item_ids = {str(item["item_id"]) for row in outfits for item in row["items"]}
    image_duplicate_groups = exact_file_duplicate_groups(raw_root, item_ids)
    components = build_components(outfits, image_duplicate_groups)
    original_counts = {split: len(rows_by_source[split]) for split in SPLITS}
    targets = proportional_targets(len(outfits), original_counts)
    assigned = assign_components(components, targets)
    metadata = json.loads((raw_root / "polyvore_item_metadata.json").read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "protocol": "polyvore_d_clean",
        "seed": seed,
        "source_protocol": "historical_polyvore_d",
        "construction_rules": [
            "Pool all historical positive outfits before repartitioning.",
            "Remove exact duplicate outfits by sorted item-ID tuple.",
            "Build the full positive-outfit/item bipartite graph and assign whole components.",
            "Link distinct item IDs whose encoded image files have identical SHA-256 hashes.",
            "Assign the largest component to train, then deterministic best-fit remaining components.",
            "Match historical outfit-count proportions using largest-remainder integer targets.",
            "Generate one category/length-matched full-corruption CP negative per positive.",
            "Generate one four-choice FITB question per outfit with same-category distractors.",
            "Draw all negative and candidate items only from the question's assigned split.",
        ],
        "historical_outfit_counts": original_counts,
        "deduplicated_outfit_count": len(outfits),
        "duplicate_outfits_removed": len(duplicates),
        "exact_image_duplicate_groups_linked": len(image_duplicate_groups),
        "target_outfit_counts": targets,
        "component_count": len(components),
        "largest_component_outfits": components[0].size,
        "largest_component_items": len(components[0].item_ids),
        "splits": {},
    }
    for split_index, split in enumerate(SPLITS):
        clean_rows = []
        for row in sorted(assigned[split], key=lambda value: str(value["set_id"])):
            clean_rows.append({"items": row["items"], "set_id": row["set_id"]})
        outfit_path = output_root / f"{split}.json"
        outfit_path.write_text(json.dumps(clean_rows, indent=2), encoding="utf-8")
        cp_lines, fitb_rows, diagnostics = generate_questions(
            clean_rows, metadata, seed + 10_000 * (split_index + 1)
        )
        cp_path = output_root / f"compatibility_{split}.txt"
        cp_path.write_text("\n".join(cp_lines) + "\n", encoding="utf-8")
        fitb_path = output_root / f"fill_in_blank_{split}.json"
        fitb_path.write_text(json.dumps(fitb_rows, indent=2), encoding="utf-8")
        source_counts = Counter(row["_historical_source"] for row in assigned[split])
        item_ids = {str(item["item_id"]) for row in clean_rows for item in row["items"]}
        manifest["splits"][split] = {
            "outfits": len(clean_rows),
            "unique_items": len(item_ids),
            "historical_source_counts": dict(sorted(source_counts.items())),
            "question_generation": diagnostics,
            "sha256": {
                outfit_path.name: file_sha256(outfit_path),
                cp_path.name: file_sha256(cp_path),
                fitb_path.name: file_sha256(fitb_path),
            },
        }
    (output_root / "duplicates_removed.json").write_text(
        json.dumps(duplicates, indent=2), encoding="utf-8"
    )
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
