"""Leakage-safe POG (Alibaba iFashion) protocol construction for v2 (HANDOFF_V2 §4, §6.5).

The positive-outfit pool is the frozen image-acquisition target (124,000 outfits drawn
with ``random.Random(20261118).sample`` in ``scripts/v2/pog_download.py``); only those
items have images.  The builder mirrors the IQON-Clean construction:

* image identity (SHA-256 of the downloaded bytes when a download manifest is given,
  otherwise the item ID) is an indivisible hypergraph node; outfits are hyperedges;
* outfits are traversed in a seeded permutation and greedily packed; an outfit that
  touches nodes owned by two different splits is discarded (IQON ``_pack`` rule);
* CP negatives and FITB distractors are drawn within a split, category- and
  length-matched, with the shared ``generate_questions`` generator and an
  exact-image equivalence so distractors never repeat an image;
* outputs use the Polyvore reader format so ``PolyvoreDisjoint`` reads them.

Differences from IQON that the owner must confirm before the freeze are recorded in
``manifest.json`` under ``provisional_decisions``.  No model is run here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from repbench.data.clean_protocol import SPLITS, UnionFind, file_sha256, generate_questions
from repbench.data.polyvore import PolyvoreDisjoint

POG_TARGET_SEED = 20261118
POG_TARGET_OUTFITS = 124_000
CONSTRUCTION_SEED = 20260912
SPLIT_RATIOS = {"train": 0.7, "valid": 0.1, "test": 0.2}
MIN_CATEGORY_GROUPS = 10
PROTOCOL_NAME = "pog_clean"
PROTOCOL_VERSION = "0.1-provisional"
_REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- inputs
def _load_target_function():
    """Import ``load_target`` from the frozen downloader so selection cannot drift."""
    path = _REPO_ROOT / "scripts" / "v2" / "pog_download.py"
    spec = importlib.util.spec_from_file_location("repbench_v2_pog_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_target


def load_target_outfits(
    items_path: Path, outfits_path: Path, seed: int, target_outfits: int
) -> tuple[list[tuple[str, list[str]]], set[str]]:
    """Return the frozen target as sorted ``(outfit_id, items)`` plus items with a URL."""
    outfit_ids, urls = _load_target_function()(items_path, outfits_path, seed, target_outfits)
    wanted = set(outfit_ids)
    target: dict[str, list[str]] = {}
    with outfits_path.open(encoding="utf-8") as stream:
        for line in stream:
            outfit_id, _, items = line.rstrip("\n").partition(",")
            if items and outfit_id in wanted:
                if outfit_id in target:
                    raise ValueError(f"Duplicate outfit ID in outfit_data: {outfit_id}")
                target[outfit_id] = items.split(";")
    if set(target) != wanted:
        raise ValueError("Target outfit IDs could not be re-read from outfit_data")
    return sorted(target.items()), set(urls)


def load_item_categories(items_path: Path, needed: set[str]) -> dict[str, str]:
    categories: dict[str, str] = {}
    with items_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) >= 2 and parts[0] in needed and parts[1].strip():
                if parts[0] in categories and categories[parts[0]] != parts[1].strip():
                    raise ValueError(f"Item {parts[0]} has conflicting cate_id values")
                categories[parts[0]] = parts[1].strip()
    return categories


def read_download_manifest(path: Path) -> tuple[dict[str, dict], int]:
    """Read ``manifest.jsonl`` written by ``pog_download.py``; returns records and torn lines."""
    records: dict[str, dict] = {}
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            item_id = str(record["item_id"])
            sha = str(record["sha256"])
            relative = str(record["path"])
        except (json.JSONDecodeError, KeyError, TypeError):
            invalid += 1
            continue
        if len(sha) != 64 or relative.startswith("/") or ".." in Path(relative).parts:
            invalid += 1
            continue
        if item_id in records and records[item_id]["sha256"] != sha:
            raise ValueError(f"Download manifest has conflicting hashes for item {item_id}")
        records[item_id] = {"sha256": sha, "path": relative}
    return records, invalid


def scan_user_clicks(lines: Iterable[str], target_outfit_ids: set[str]) -> dict:
    """Stream POG ``user_data.txt`` and record, per target outfit, its clicking users.

    POG lines are ``user_id,item_id;item_id;...,outfit_id`` (README, wenyuer/POG). The
    last field may hold several ``;``-joined outfit IDs; all are honoured. Only the
    lexicographically smallest user ID and the number of distinct user lines are kept,
    so memory stays proportional to the target, not to the 8 GB source.
    """
    smallest: dict[str, str] = {}
    lines_per_outfit: Counter[str] = Counter()
    stats = Counter()
    for line in lines:
        stats["lines"] += 1
        fields = line.rstrip("\r\n").split(",")
        if len(fields) < 3 or not fields[0]:
            stats["malformed_lines"] += 1
            continue
        user_id = fields[0]
        outfit_ids = {value for value in fields[-1].split(";") if value}
        if not outfit_ids:
            stats["lines_without_outfit"] += 1
            continue
        stats["lines_with_outfit"] += 1
        for outfit_id in outfit_ids & target_outfit_ids:
            stats["target_outfit_clicks"] += 1
            lines_per_outfit[outfit_id] += 1
            if outfit_id not in smallest or user_id < smallest[outfit_id]:
                smallest[outfit_id] = user_id
    return {
        "rule": "min_user_id: lexicographically smallest user_id whose line names the outfit",
        "scan_counts": dict(sorted(stats.items())),
        "outfits": {
            outfit_id: {
                "min_user_id": smallest[outfit_id],
                "click_lines": lines_per_outfit[outfit_id],
            }
            for outfit_id in sorted(smallest)
        },
    }


# ----------------------------------------------------------------------- utilities
def _canonical_json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _components(
    outfits: list[tuple[str, list[str]]], item_to_group: Mapping[str, str]
) -> list[list[str]]:
    """Connected components of the outfit/image-group hypergraph as sorted outfit-ID lists."""
    union_find = UnionFind()
    for _, items in outfits:
        groups = [item_to_group[item] for item in items]
        for group in groups:
            union_find.find(group)
        for group in groups[1:]:
            union_find.union(groups[0], group)
    grouped: dict[str, list[str]] = defaultdict(list)
    for outfit_id, items in outfits:
        grouped[union_find.find(item_to_group[items[0]])].append(outfit_id)
    return sorted(
        (sorted(rows) for rows in grouped.values()), key=lambda rows: (-len(rows), rows[0])
    )


def _component_summary(components: list[list[str]]) -> dict:
    total = sum(map(len, components))
    sizes = [len(rows) for rows in components]
    return {
        "outfits": total,
        "components": len(components),
        "largest_component_outfits": sizes[0] if sizes else 0,
        "largest_component_fraction": round(sizes[0] / total, 6) if total else 0.0,
        "singleton_components": sum(size == 1 for size in sizes),
        "top_component_sizes": sizes[:10],
    }


def _distribution(values: Iterable) -> dict:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def pack_outfits(
    outfits: list[tuple[str, list[str]]],
    item_to_group: Mapping[str, str],
    seed: int,
    ratios: Mapping[str, float] = SPLIT_RATIOS,
) -> tuple[dict[str, list[int]], dict]:
    """IQON-style greedy hypergraph packing with proportional seeding.

    IQON filled fixed capacities from a larger pool. The POG pool is fixed by the image
    target, so a new (unowned) outfit seeds the split with the largest deficit against
    ``ratio * assigned``; outfits whose nodes are owned by one split join it; outfits
    touching nodes owned by two or more splits are discarded.
    """
    order = np.random.default_rng(seed).permutation(len(outfits))
    owner: dict[str, str] = {}
    assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
    conflicts = 0
    seeded = Counter()
    for index in order:
        groups = {item_to_group[item] for item in outfits[int(index)][1]}
        existing = {owner[group] for group in groups if group in owner}
        if len(existing) > 1:
            conflicts += 1
            continue
        if existing:
            split = existing.pop()
        else:
            total = sum(len(rows) for rows in assigned.values()) + 1
            split = max(
                SPLITS,
                key=lambda value: (
                    ratios[value] * total - len(assigned[value]),
                    -SPLITS.index(value),
                ),
            )
            seeded[split] += 1
        owner.update((group, split) for group in groups)
        assigned[split].append(int(index))
    return assigned, {
        "conflict_outfits_discarded": conflicts,
        "new_component_seeds": {split: seeded[split] for split in SPLITS},
    }


def _enforce_category_support(
    split_outfits: dict[str, list[tuple[str, list[str]]]],
    categories: Mapping[str, str],
    item_to_group: Mapping[str, str],
    minimum: int,
) -> tuple[dict[str, list[tuple[str, list[str]]]], dict]:
    """Drop outfits whose items sit in categories with too few distinct images in-split.

    Same-category negatives and FITB distractors need a pool; iterate to a fixed point.
    """
    report = {}
    result = {}
    for split in SPLITS:
        rows = list(split_outfits[split])
        dropped = 0
        rounds = 0
        weak: set[str] = set()
        while True:
            rounds += 1
            support: dict[str, set[str]] = defaultdict(set)
            for _, items in rows:
                for item in items:
                    support[categories[item]].add(item_to_group[item])
            weak_now = {category for category, groups in support.items() if len(groups) < minimum}
            if not weak_now:
                break
            weak |= weak_now
            kept = [row for row in rows if not any(categories[item] in weak_now for item in row[1])]
            dropped += len(rows) - len(kept)
            rows = kept
        result[split] = rows
        report[split] = {
            "outfits_dropped": dropped,
            "categories_excluded": sorted(weak),
            "rounds": rounds,
        }
    return result, report


# ---------------------------------------------------------------------------- audit
def audit_protocol(protocol_dir: Path) -> dict:
    """Re-read the written protocol through ``PolyvoreDisjoint`` and audit leakage/collisions."""
    dataset = PolyvoreDisjoint(protocol_dir, protocol_dir, PROTOCOL_NAME)
    metadata = json.loads(
        (protocol_dir / "polyvore_item_metadata.json").read_text(encoding="utf-8")
    )
    category = {item: row["semantic_category"] for item, row in metadata.items()}
    image = {item: row["image_identity"] for item, row in metadata.items()}
    items = {split: dataset.item_ids(split) for split in SPLITS}
    images = {split: {image[item] for item in items[split]} for split in SPLITS}
    tuples = {
        split: [tuple(sorted(outfit.item_ids)) for outfit in dataset.outfits(split)]
        for split in SPLITS
    }
    report: dict = {"cross_split": {}, "splits": {}}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            report["cross_split"][f"{left}_{right}"] = {
                "item_ids": len(items[left] & items[right]),
                "image_identities": len(images[left] & images[right]),
                "outfit_tuples": len(set(tuples[left]) & set(tuples[right])),
            }
    all_positive = set().union(*(set(values) for values in tuples.values()))
    ok = all(value == 0 for row in report["cross_split"].values() for value in row.values())
    for split in SPLITS:
        positives = set(tuples[split])
        cp = dataset.compatibility(split)
        pos = [row for row in cp if row.label == 1]
        neg = [row for row in cp if row.label == 0]
        neg_sets = [tuple(sorted(row.item_ids)) for row in neg]
        matched = sum(
            sorted(category[item] for item in p.item_ids)
            == sorted(category[item] for item in n.item_ids)
            for p, n in zip(pos, neg, strict=True)
        )
        fitb = dataset.fitb(split)
        valid = 0
        for question in fitb:
            candidates = question.candidate_item_ids
            correct = candidates[question.correct_index]
            valid += (
                len(candidates) == 4
                and len(set(candidates)) == 4
                and len({image[item] for item in candidates}) == 4
                and tuple(sorted(question.question_item_ids + (correct,))) in positives
                and all(item in items[split] for item in candidates)
                and all(category[item] == category[correct] for item in candidates)
                and not any(
                    tuple(sorted(question.question_item_ids + (candidate,))) in all_positive
                    for index, candidate in enumerate(candidates)
                    if index != question.correct_index
                )
            )
        checks = {
            "outfits": len(tuples[split]),
            "duplicate_outfit_tuples": len(tuples[split]) - len(positives),
            "cp_positive": len(pos),
            "cp_negative": len(neg),
            "cp_balanced": len(pos) == len(neg) == len(tuples[split]),
            "cp_category_and_length_matched": matched == len(neg),
            "cp_negative_duplicates": len(neg_sets) - len(set(neg_sets)),
            "cp_negative_equals_positive": len(set(neg_sets) & all_positive),
            "fitb_questions": len(fitb),
            "fitb_invalid": len(fitb) - valid,
            "fitb_answer_index_counts": _distribution(q.correct_index for q in fitb),
        }
        report["splits"][split] = checks
        ok &= (
            checks["duplicate_outfit_tuples"] == 0
            and checks["cp_balanced"]
            and checks["cp_category_and_length_matched"]
            and checks["cp_negative_duplicates"] == 0
            and checks["cp_negative_equals_positive"] == 0
            and checks["fitb_invalid"] == 0
            and len(fitb) == len(tuples[split])
        )
    report["passes"] = bool(ok)
    return report


# ------------------------------------------------------------------------ builder
def construct_pog_clean(
    items_path: Path,
    outfits_path: Path,
    output_root: Path,
    *,
    download_manifest: Path | None = None,
    user_clicks: Path | None = None,
    category_map: Mapping[str, str] | None = None,
    category_map_name: str = "identity_cate_id",
    seed: int = CONSTRUCTION_SEED,
    target_seed: int = POG_TARGET_SEED,
    target_outfits: int = POG_TARGET_OUTFITS,
    min_category_groups: int = MIN_CATEGORY_GROUPS,
    ratios: Mapping[str, float] = SPLIT_RATIOS,
    image_equivalence: Mapping[str, str] | None = None,
    image_equivalence_info: Mapping | None = None,
    protocol_name: str | None = None,
) -> dict:
    """Build the POG clean protocol into a new directory and return its manifest.

    ``image_equivalence`` (v2-Clean, owner decision E): ``{item_id: group_id}`` of
    near-duplicate images; exact-image identities linked by a group are merged into one
    identity (indivisible packing node, distinct-image rule for questions, audit).
    ``None`` (default) reproduces the original construction byte for byte.

    ``download_manifest``: ``manifest.jsonl`` from ``pog_download.py``. When given, items
    absent from it are unavailable (their outfits are dropped) and exact-image identity
    is the SHA-256 of the image bytes; otherwise every target item is assumed available
    and identity falls back to the item ID.
    ``user_clicks``: JSON written from :func:`scan_user_clicks`; absent -> item-component
    groups are used for the cluster bootstrap.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must cover {SPLITS} and sum to one")

    target, with_url = load_target_outfits(items_path, outfits_path, target_seed, target_outfits)
    target_items = {item for _, items in target for item in items}
    cate_ids = load_item_categories(items_path, target_items)
    if category_map is None:
        semantic = {item: cate for item, cate in cate_ids.items()}
        category_map_sha = "identity"
    else:
        unmapped = sorted({cate for cate in cate_ids.values() if cate not in category_map})
        if unmapped:
            raise KeyError(
                f"Category map lacks {len(unmapped)} cate_id values, e.g. {unmapped[:3]}"
            )
        semantic = {item: str(category_map[cate]).strip() for item, cate in cate_ids.items()}
        category_map_sha = _canonical_json_sha256(dict(category_map))

    available: dict[str, dict] | None = None
    torn_lines = 0
    if download_manifest is not None:
        available, torn_lines = read_download_manifest(download_manifest)

    # ---- retention stages (each outfit is attributed to the first rule that drops it)
    stages: list[dict] = []
    dropped_by: dict[str, str] = {}
    kept: list[tuple[str, list[str]]] = []
    for outfit_id, items in target:
        if len(items) < 2 or len(set(items)) != len(items):
            dropped_by[outfit_id] = "fewer_than_two_or_repeated_items"
        elif any(item not in cate_ids or item not in with_url for item in items):
            dropped_by[outfit_id] = "item_without_category_or_url"
        elif available is not None and any(item not in available for item in items):
            dropped_by[outfit_id] = "item_image_unavailable"
        else:
            kept.append((outfit_id, items))
    seen: dict[tuple[str, ...], str] = {}
    pool: list[tuple[str, list[str]]] = []
    for outfit_id, items in kept:  # sorted by outfit ID -> smallest ID kept
        key = tuple(sorted(items))
        if key in seen:
            dropped_by[outfit_id] = "duplicate_item_tuple"
            continue
        seen[key] = outfit_id
        pool.append((outfit_id, items))
    stage_counts = Counter(dropped_by.values())
    remaining = len(target)
    for reason in (
        "fewer_than_two_or_repeated_items",
        "item_without_category_or_url",
        "item_image_unavailable",
        "duplicate_item_tuple",
    ):
        remaining -= stage_counts[reason]
        stages.append(
            {
                "rule": reason,
                "outfits_dropped": stage_counts[reason],
                "outfits_remaining": remaining,
            }
        )

    pool_items = {item for _, items in pool for item in items}
    if available is not None:
        item_to_group = {item: "sha256:" + available[item]["sha256"] for item in pool_items}
    else:
        item_to_group = {item: "item:" + item for item in pool_items}
    exact_identity = item_to_group
    if image_equivalence is not None:
        from repbench.data.image_equivalence import merge_identities

        item_to_group = merge_identities(exact_identity, image_equivalence)
    shared_image_groups = Counter(item_to_group.values())
    pool_components = _components(pool, item_to_group)

    assigned, packing = pack_outfits(pool, item_to_group, seed, ratios)
    packed = {split: sorted(pool[index] for index in assigned[split]) for split in SPLITS}
    for index in set(range(len(pool))) - {i for rows in assigned.values() for i in rows}:
        dropped_by[pool[index][0]] = "packing_conflict"
    remaining -= packing["conflict_outfits_discarded"]
    stages.append(
        {
            "rule": "packing_conflict",
            "outfits_dropped": packing["conflict_outfits_discarded"],
            "outfits_remaining": remaining,
        }
    )
    final, support_report = _enforce_category_support(
        packed, semantic, item_to_group, min_category_groups
    )
    support_dropped = sum(row["outfits_dropped"] for row in support_report.values())
    final_ids = {outfit_id for rows in final.values() for outfit_id, _ in rows}
    for rows in packed.values():
        for outfit_id, _ in rows:
            if outfit_id not in final_ids:
                dropped_by[outfit_id] = "insufficient_category_support"
    remaining -= support_dropped
    stages.append(
        {
            "rule": "insufficient_category_support",
            "outfits_dropped": support_dropped,
            "outfits_remaining": remaining,
        }
    )

    # ---- selection-effect audit: retained vs dropped target outfits
    degree = Counter(item for _, items in target for item in set(items))

    def _profile(rows: list[tuple[str, list[str]]]) -> dict:
        if not rows:
            return {"outfits": 0}
        degrees = [degree[item] for _, items in rows for item in items]
        cats = Counter(cate_ids.get(item, "<missing>") for _, items in rows for item in items)
        total = sum(cats.values())
        return {
            "outfits": len(rows),
            "length_counts": _distribution(len(items) for _, items in rows),
            "mean_length": round(float(np.mean([len(items) for _, items in rows])), 4),
            "mean_item_target_degree": round(float(np.mean(degrees)), 4),
            "median_item_target_degree": float(np.median(degrees)),
            "top_cate_id_shares": {
                cate: round(count / total, 5)
                for cate, count in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
            },
        }

    retained_rows = [row for row in target if row[0] in final_ids]
    discarded_rows = [row for row in target if row[0] not in final_ids]
    packing_discards = [row for row in target if dropped_by.get(row[0]) == "packing_conflict"]

    # ---- user groups for the cluster bootstrap
    clicks = None
    user_source = None
    if user_clicks is not None:
        clicks_doc = json.loads(Path(user_clicks).read_text(encoding="utf-8"))
        clicks = clicks_doc["outfits"]
        user_source = {k: v for k, v in clicks_doc.items() if k != "outfits"}

    metadata: dict[str, dict] = {}
    image_paths: dict[str, str] = {}
    manifest: dict = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "construction_seed": seed,
        "question_seeds": {split: seed + 10_000 * (i + 1) for i, split in enumerate(SPLITS)},
        "target_selection": {
            "function": "scripts/v2/pog_download.py:load_target",
            "rule": "random.Random(seed).sample(sorted (outfit_id, items) tuples, n)",
            "seed": target_seed,
            "requested_outfits": target_outfits,
            "selected_outfits": len(target),
            "selected_items": len(target_items),
            "target_outfit_ids_sha256": hashlib.sha256(
                ("\n".join(outfit_id for outfit_id, _ in target) + "\n").encode()
            ).hexdigest(),
        },
        "inputs": {
            "item_data": {"name": Path(items_path).name, "sha256": file_sha256(Path(items_path))},
            "outfit_data": {
                "name": Path(outfits_path).name,
                "sha256": file_sha256(Path(outfits_path)),
            },
            "download_manifest": None
            if download_manifest is None
            else {
                "name": Path(download_manifest).name,
                "sha256": file_sha256(Path(download_manifest)),
                "records": len(available),
                "invalid_or_torn_lines": torn_lines,
            },
            "user_clicks": None
            if user_clicks is None
            else {
                "name": Path(user_clicks).name,
                "sha256": file_sha256(Path(user_clicks)),
                "source": user_source,
            },
        },
        "settings": {
            "split_ratios": dict(ratios),
            "min_distinct_images_per_category_per_split": min_category_groups,
            "category_map": {"name": category_map_name, "sha256": category_map_sha},
            "image_identity": "sha256 of downloaded image bytes"
            if available is not None
            else "item_id (no download manifest: all target items assumed available)",
            "group_rule": "user:<smallest clicking user_id>, else unclicked:<outfit_id>"
            if clicks is not None
            else "item_component fallback (no user data): component:<split>:<smallest outfit_id>",
        },
        "construction_rules": [
            "Positive pool is the frozen 124k image-acquisition target (load_target, seed 20261118).",
            "Drop outfits with fewer than two items or a repeated item.",
            "Drop outfits containing an item without cate_id or pic_url.",
            "Drop outfits containing an item absent from the download manifest (if given).",
            "Keep the smallest outfit_id of duplicate sorted item-ID tuples.",
            "Treat image identity (SHA-256 of bytes, else item ID) as indivisible hypergraph nodes.",
            (
                "Traverse outfits in a seeded permutation; join the owning split; seed new outfits "
                "into the split with the largest deficit against the 70/10/20 ratio; discard outfits "
                "touching nodes owned by two splits."
            ),
            (
                "Within each split drop outfits in categories with fewer than the minimum distinct "
                "images (fixed point), so same-category questions can be generated."
            ),
            (
                "Generate one category/length-matched CP negative and one four-way same-category "
                "FITB question per outfit, within split only, with distinct image identities."
            ),
            "Never use user IDs as model features; they only define bootstrap clusters.",
        ],
        "provisional_decisions": [
            "Category map for same-category negatives defaults to identity on fine-grained cate_id.",
            "Packing uses proportional seeding instead of IQON fixed capacities (fixed image pool).",
            "Bootstrap user group = smallest user_id among users who clicked the outfit.",
            f"Categories with fewer than {min_category_groups} distinct images in a split are excluded there.",
        ],
        "retention": {
            "stages": stages,
            "final_outfits": len(final_ids),
            "final_fraction_of_target": round(len(final_ids) / len(target), 6) if target else 0.0,
            "selection_effect": {
                "target": _profile(target),
                "retained": _profile(retained_rows),
                "discarded": _profile(discarded_rows),
                "discarded_by_packing_conflict": _profile(packing_discards),
            },
        },
        "image_identity_groups_shared_by_multiple_items": sum(
            count > 1 for count in shared_image_groups.values()
        ),
        "pool_components": _component_summary(pool_components),
        "packing": packing,
        "category_support": support_report,
        "splits": {},
    }
    if protocol_name is not None:
        manifest["protocol"] = protocol_name
    if image_equivalence is not None:
        from repbench.data.image_equivalence import manifest_record

        manifest["settings"]["image_identity"] += (
            "; near-duplicate image-equivalence groups merged (near:<smallest identity>)"
        )
        manifest["construction_rules"].insert(
            6,
            "Merge exact-image identities linked by a near-duplicate image-equivalence group "
            "(v2-Clean) into one identity before packing, category support and questions.",
        )
        manifest["image_equivalence"] = manifest_record(
            image_equivalence, pool_items, image_equivalence_info
        )

    for split_index, split in enumerate(SPLITS):
        rows_in = final[split]
        rows = [
            {
                "set_id": outfit_id,
                "items": [
                    {
                        "item_id": item,
                        "index": index,
                        "semantic_category": semantic[item],
                        "cate_id": cate_ids[item],
                    }
                    for index, item in enumerate(items, start=1)
                ],
            }
            for outfit_id, items in rows_in
        ]
        split_items = {item for _, items in rows_in for item in items}
        for item in split_items:
            metadata[item] = {
                "semantic_category": semantic[item],
                "cate_id": cate_ids[item],
                "image_identity": item_to_group[item],
            }
            if image_equivalence is not None:
                metadata[item]["exact_image_identity"] = exact_identity[item]
            image_paths[item] = (
                available[item]["path"] if available is not None else f"images/{item[:2]}/{item}"
            )
        outfit_path = output_root / f"{split}.json"
        _write_json(outfit_path, rows)
        cp_lines, fitb_rows, diagnostics = generate_questions(
            rows,
            metadata,
            manifest["question_seeds"][split],
            {item: item_to_group[item] for item in split_items},
        )
        cp_path = output_root / f"compatibility_{split}.txt"
        cp_path.write_text("\n".join(cp_lines) + "\n", encoding="utf-8")
        fitb_path = output_root / f"fill_in_blank_{split}.json"
        _write_json(fitb_path, fitb_rows)

        split_components = _components(rows_in, item_to_group)
        if clicks is not None:
            groups = {
                outfit_id: (
                    "user:" + clicks[outfit_id]["min_user_id"]
                    if outfit_id in clicks
                    else "unclicked:" + outfit_id
                )
                for outfit_id, _ in rows_in
            }
        else:
            groups = {
                outfit_id: f"component:{split}:{component[0]}"
                for component in split_components
                for outfit_id in component
            }
        groups = dict(sorted(groups.items()))
        groups_path = output_root / f"group_ids_{split}.json"
        _write_json(groups_path, groups)
        group_sizes = sorted(Counter(groups.values()).values(), reverse=True)
        manifest["splits"][split] = {
            "outfits": len(rows),
            "unique_items": len(split_items),
            "unique_image_identities": len({item_to_group[item] for item in split_items}),
            "outfit_length_counts": _distribution(len(row["items"]) for row in rows),
            "semantic_categories": len({semantic[item] for item in split_items}),
            "item_components": _component_summary(split_components),
            "groups": {
                "groups": len(group_sizes),
                "largest_group_outfits": group_sizes[0] if group_sizes else 0,
                "largest_group_fraction": round(group_sizes[0] / len(rows), 6) if rows else 0.0,
                "singleton_groups": sum(size == 1 for size in group_sizes),
                "unclicked_outfits": sum(
                    value.startswith("unclicked:") for value in groups.values()
                ),
                "top_group_sizes": group_sizes[:10],
            },
            "question_generation": diagnostics,
            "sha256": {
                path.name: file_sha256(path)
                for path in (outfit_path, cp_path, fitb_path, groups_path)
            },
        }
    metadata_path = output_root / "polyvore_item_metadata.json"
    _write_json(metadata_path, dict(sorted(metadata.items())))
    paths_path = output_root / "image_paths.json"
    _write_json(paths_path, dict(sorted(image_paths.items())))
    retention_path = output_root / "retention_dropped_outfits.json"
    _write_json(retention_path, dict(sorted(dropped_by.items())))
    manifest["image_paths"] = {
        "file": paths_path.name,
        "extension_resolved": available is not None,
        "note": "POG images live at images/<id[:2]>/<id>.<png|jpeg|webp|gif> under the mirror root; "
        "PolyvoreDisjoint.image_path assumes images/<id>.jpg, so a POG-aware resolver is required.",
    }
    manifest["auxiliary_sha256"] = {
        path.name: file_sha256(path) for path in (metadata_path, paths_path, retention_path)
    }
    manifest["audit"] = audit_protocol(output_root)
    _write_json(output_root / "manifest.json", manifest)
    if not manifest["audit"]["passes"]:
        raise AssertionError(f"POG protocol audit failed: {manifest['audit']}")
    return manifest


def protocol_digest(protocol_dir: Path) -> dict[str, str]:
    """SHA-256 of every file in a built protocol directory (for the two-build check)."""
    return {
        path.name: file_sha256(path)
        for path in sorted(Path(protocol_dir).iterdir())
        if path.is_file()
    }
