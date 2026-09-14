import json
from pathlib import Path

import pandas as pd
import pytest

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_disjoint

PROTOCOL = Path("data/protocols/iqon3000_clean")
DATASET_ROOT = Path(".venv/iqon3000_clean")
pytestmark = pytest.mark.skipif(
    not (PROTOCOL / "test.json").is_file()
    or not Path(".venv/iqon_audit/catalog/image_audit_resolved.parquet").is_file(),
    reason="IQON3000 data are intentionally absent from the public source release",
)


def test_iqon_locked_sizes_and_item_disjointness():
    dataset = PolyvoreDisjoint(DATASET_ROOT, PROTOCOL, "iqon3000_clean")
    audit = inspect_disjoint(dataset)
    assert {split: row["outfits"] for split, row in audit["splits"].items()} == {
        "train": 86_800,
        "valid": 12_400,
        "test": 24_800,
    }
    assert audit["passes_item_disjointness"]
    assert audit["item_overlap_counts"] == {"train_valid": 0, "train_test": 0, "valid_test": 0}
    assert all(row["missing_images"] == 0 for row in audit["splits"].values())


def test_iqon_no_duplicate_outfits_or_exact_image_overlap():
    rows = {
        split: json.loads((PROTOCOL / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "valid", "test")
    }
    canonical = {
        split: {tuple(sorted(str(item["item_id"]) for item in row["items"])) for row in values}
        for split, values in rows.items()
    }
    assert all(len(canonical[split]) == len(rows[split]) for split in rows)
    images = pd.read_parquet(".venv/iqon_audit/catalog/image_audit_resolved.parquet").set_index("item_id")
    hashes = {}
    for split, values in rows.items():
        ids = {str(item["item_id"]) for row in values for item in row["items"]}
        hashes[split] = {
            "encoded": set(images.loc[list(ids)].encoded_sha256),
            "decoded": set(images.loc[list(ids)].decoded_rgb_sha256),
        }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert not canonical[left] & canonical[right]
        assert not hashes[left]["encoded"] & hashes[right]["encoded"]
        assert not hashes[left]["decoded"] & hashes[right]["decoded"]


def test_iqon_cp_and_fitb_structure_and_visual_candidate_uniqueness():
    images = pd.read_parquet(".venv/iqon_audit/catalog/image_audit_resolved.parquet").set_index("item_id")
    dataset = PolyvoreDisjoint(DATASET_ROOT, PROTOCOL, "iqon3000_clean")
    for split in dataset.SPLITS:
        cp = dataset.compatibility(split)
        assert sum(row.label == 1 for row in cp) == sum(row.label == 0 for row in cp)
        fitb = dataset.fitb(split)
        assert {len(row.candidate_item_ids) for row in fitb} == {4}
        assert all(len(set(row.candidate_item_ids)) == 4 for row in fitb)
        assert all(
            len(set(images.loc[list(row.candidate_item_ids)].decoded_rgb_sha256)) == 4
            for row in fitb
        )
