import json
from pathlib import Path

import pytest

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import validate_clean_protocol

pytestmark = pytest.mark.skipif(
    not Path("data/protocols/polyvore_d_clean/test.json").is_file()
    or not Path("data/raw/polyvore_outfits/polyvore_item_metadata.json").is_file(),
    reason="Polyvore data are intentionally absent from the public source release",
)


@pytest.fixture(scope="module")
def clean_report():
    root = Path("data/raw/polyvore_outfits")
    protocol = Path("data/protocols/polyvore_d_clean")
    return validate_clean_protocol(PolyvoreDisjoint(root, protocol, "polyvore_d_clean"))


def test_zero_cross_split_item_overlap(clean_report):
    assert clean_report["item_overlap_counts"] == {
        "train_valid": 0,
        "train_test": 0,
        "valid_test": 0,
    }


def test_no_duplicate_outfits_or_missing_images(clean_report):
    assert clean_report["global_duplicate_outfits"] == 0
    assert all(row["missing_images"] == 0 for row in clean_report["splits"].values())


def test_cp_negatives_are_balanced_controlled_and_unique(clean_report):
    for row in clean_report["clean_protocol_checks"].values():
        assert row["cp_balanced"]
        assert row["cp_category_and_length_matched"]
        assert row["cp_unique_negatives"]
        assert row["cp_negatives_not_positives"]


def test_fitb_candidate_structure(clean_report):
    for row in clean_report["clean_protocol_checks"].values():
        assert row["fitb_all_valid"]


def test_no_cross_split_exact_image_duplicates():
    audit = json.loads(
        Path("artifacts/clean_image_duplicate_audit.json").read_text(encoding="utf-8")
    )
    assert audit["passes_exact_image_disjointness"]
    assert all(
        row["cross_split_duplicate_groups"] == 0 for row in audit["findings"].values()
    )
