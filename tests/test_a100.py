import json
from pathlib import Path

import pytest

from repbench.data.a100 import AAT_DIMENSIONS

ROOT = Path("data/protocols/a100")
pytestmark = pytest.mark.skipif(
    not (ROOT / "lat.json").is_file() or not (ROOT / "aat.json").is_file(),
    reason="A100 archive is intentionally absent from the public source release",
)


def test_a100_indexing_and_candidate_counts():
    for task in ("aat", "lat"):
        rows = json.loads((ROOT / f"{task}.json").read_text())
        assert len(rows) == 100
        assert [row["question_num"] for row in rows] == list(range(1, 101))
        assert {len(row["answers"]) for row in rows} == {5}
        assert all(0 <= row["archive_gt_index"] < 5 for row in rows)


def test_lat_majority_policy_and_known_disagreements():
    rows = json.loads((ROOT / "lat.json").read_text())
    assert all(row["majority_gt_index"] == max(range(5), key=row["gt_distribution"].__getitem__) for row in rows)
    disagreements = [row["question_num"] for row in rows if row["archive_gt_index"] != row["majority_gt_index"]]
    assert disagreements == [92, 93, 94, 96, 98, 99, 100]


def test_aat_dimension_mapping_is_complete_and_nonoverlapping():
    rows = json.loads((ROOT / "aat.json").read_text())
    expected = {
        number: name
        for name, (lower, upper) in AAT_DIMENSIONS.items()
        for number in range(lower, upper + 1)
    }
    assert len(expected) == 100
    assert all(row["dimension"] == expected[row["question_num"]] for row in rows)
