import json

import pytest

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import validate_disjoint


def _write_split(root, split, set_id, item_id):
    directory = root / "disjoint"
    directory.mkdir(parents=True, exist_ok=True)
    row = [
        {
            "set_id": set_id,
            "items": [{"item_id": item_id, "index": 1}, {"item_id": f"{item_id}x", "index": 2}],
        }
    ]
    (directory / f"{split}.json").write_text(json.dumps(row), encoding="utf-8")
    (directory / f"compatibility_{split}.txt").write_text(
        f"1 {set_id}_1 {set_id}_2\n", encoding="utf-8"
    )
    fitb = [{"question": [f"{set_id}_1"], "answers": [f"{set_id}_2"]}]
    (directory / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb), encoding="utf-8")


def test_overlap_fails_closed(tmp_path):
    _write_split(tmp_path, "train", "1", "shared")
    _write_split(tmp_path, "valid", "2", "valid")
    _write_split(tmp_path, "test", "3", "shared")
    with pytest.raises(AssertionError):
        validate_disjoint(PolyvoreDisjoint(tmp_path), check_images=False)
