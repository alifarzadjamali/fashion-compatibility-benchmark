"""Near-duplicate image-equivalence groups for the v2-Clean protocols (owner decision E).

``scripts/v2/image_equivalence.py`` computes a mapping ``{item_id: group_id}`` for items
whose images are model-free near duplicates. The clean-protocol builders accept that
mapping as ``image_equivalence`` and treat items sharing a group exactly like items
sharing an exact image: they become one indivisible unit for split packing (and, where
the builder uses image identity for question generation, one image identity).

``image_equivalence=None`` (the default everywhere) leaves every builder unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

NEAR_DUPLICATE_PREFIX = "near:"


def load_image_equivalence(path: Path) -> tuple[dict[str, str], dict]:
    """Read ``equivalence_groups.json``; return the mapping and its provenance."""
    path = Path(path)
    raw = path.read_bytes()
    mapping = json.loads(raw.decode("utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise ValueError(f"{path} must be a JSON object of item_id -> group_id strings")
    return mapping, {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def mapping_sha256(mapping: Mapping[str, str]) -> str:
    """Canonical content hash of a mapping (independent of file formatting)."""
    text = json.dumps(dict(sorted(mapping.items())), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def equivalence_groups(
    image_equivalence: Mapping[str, str], restrict_to: Iterable[str]
) -> list[list[str]]:
    """Groups (sorted lists, size >= 2) of the mapping restricted to ``restrict_to``."""
    keep = set(restrict_to)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item, group in image_equivalence.items():
        if item in keep:
            grouped[group].append(item)
    return sorted(sorted(items) for items in grouped.values() if len(items) > 1)


def merge_identities(
    item_to_identity: Mapping[str, str], image_equivalence: Mapping[str, str]
) -> dict[str, str]:
    """Join exact-image identities that an equivalence group links.

    Returns a new ``item -> identity`` mapping over the same items. Identity classes
    that no near-duplicate group touches keep their label; a merged class is labelled
    ``near:<smallest member identity>`` (deterministic, independent of input order).
    """
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    by_group: dict[str, list[str]] = defaultdict(list)
    for item, group in image_equivalence.items():
        if item in item_to_identity:
            by_group[group].append(item_to_identity[item])
    for identities in by_group.values():
        for identity in identities[1:]:
            left, right = find(identities[0]), find(identity)
            if left != right:
                parent[max(left, right)] = min(left, right)
    members: dict[str, set[str]] = defaultdict(set)
    for identity in set(item_to_identity.values()):
        members[find(identity)].add(identity)
    label = {}
    for identities in members.values():
        merged = len(identities) > 1
        for identity in identities:
            label[identity] = NEAR_DUPLICATE_PREFIX + min(identities) if merged else identity
    return {item: label[identity] for item, identity in item_to_identity.items()}


def groups_spanning_splits(
    split_items: Mapping[str, Iterable[str]], image_equivalence: Mapping[str, str]
) -> dict:
    """Audit: equivalence groups whose items occur in more than one split (must be 0)."""
    owners: dict[str, set[str]] = defaultdict(set)
    for split, items in split_items.items():
        for item in items:
            group = image_equivalence.get(item)
            if group is not None:
                owners[group].add(split)
    spanning = sorted(group for group, splits in owners.items() if len(splits) > 1)
    return {
        "groups_present": len(owners),
        "groups_spanning_splits": len(spanning),
        "examples": spanning[:20],
    }


def manifest_record(
    image_equivalence: Mapping[str, str], pool_items: Iterable[str], info: Mapping | None
) -> dict:
    """Provenance block stored in a v2-Clean manifest."""
    pool = set(pool_items)
    groups = equivalence_groups(image_equivalence, pool)
    return {
        **dict(info or {}),
        "mapping_sha256": mapping_sha256(image_equivalence),
        "mapping_items": len(image_equivalence),
        "pool_items_in_groups": sum(map(len, groups)),
        "pool_groups": len(groups),
        "largest_pool_group": max(map(len, groups), default=0),
    }
