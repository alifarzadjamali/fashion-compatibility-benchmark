"""Synthetic POG-like fixture tests for the v2 POG clean-protocol builder."""

import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.v2.pog import (
    audit_protocol,
    construct_pog_clean,
    load_target_outfits,
    protocol_digest,
    scan_user_clicks,
)

CATEGORIES = ("top", "bottom", "shoe", "bag")
TARGET = 300
MIN_IMAGES = 5


def _item(category: str, index: int) -> str:
    return hashlib.md5(f"{category}-{index}".encode()).hexdigest()


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("pog_raw")
    rng = random.Random(7)
    items = {category: [_item(category, i) for i in range(120)] for category in CATEGORIES}
    with (root / "item_data.txt").open("w", encoding="utf-8") as stream:
        for category, ids in items.items():
            for item in ids:
                stream.write(f"{item},cate_{category},//img.example/{item}.png,标题, with comma\n")
    outfits = []
    for index in range(340):
        length = rng.choice((3, 4))
        chosen = [
            rng.choice(items[category][: 60 if index % 3 else 120])
            for category in CATEGORIES[:length]
        ]
        outfits.append((hashlib.md5(f"outfit-{index}".encode()).hexdigest(), chosen))
    outfits.append((hashlib.md5(b"dup").hexdigest(), list(outfits[0][1])))  # duplicate tuple
    outfits.append((hashlib.md5(b"single").hexdigest(), [items["top"][0]]))  # too short
    with (root / "outfit_data.txt").open("w", encoding="utf-8") as stream:
        for outfit_id, chosen in outfits:
            stream.write(f"{outfit_id},{';'.join(chosen)}\n")
    target, _ = load_target_outfits(
        root / "item_data.txt", root / "outfit_data.txt", 20261118, TARGET
    )
    target_ids = [outfit_id for outfit_id, _ in target]
    lines = []
    for index, outfit_id in enumerate(target_ids[: TARGET // 2]):
        for user in (f"u{index % 17:03d}", f"u{(index * 7) % 23:03d}"):
            lines.append(f"{user},{target[index][1][0]};{target[index][1][1]},{outfit_id}\n")
    lines.append("broken line\n")
    clicks = scan_user_clicks(lines, set(target_ids))
    (root / "clicks.json").write_text(json.dumps(clicks), encoding="utf-8")
    return root, target


def _build(raw, output, **kwargs):
    root, _ = raw
    kwargs.setdefault("min_category_groups", MIN_IMAGES)
    return construct_pog_clean(
        root / "item_data.txt", root / "outfit_data.txt", output, target_outfits=TARGET, **kwargs
    )


def _split_rows(protocol: Path) -> dict[str, list[dict]]:
    return {
        s: json.loads((protocol / f"{s}.json").read_text(encoding="utf-8"))
        for s in ("train", "valid", "test")
    }


def test_target_reuses_frozen_downloader_selection(raw):
    _, target = raw
    assert len(target) == TARGET
    assert target == sorted(target)


def test_indivisible_items_and_exact_images_and_reader_compatibility(raw, tmp_path):
    _, target = raw
    items = sorted({item for _, values in target for item in values})
    # Two distinct item IDs share identical image bytes: they must never be split.
    twin_a, twin_b = items[0], items[1]
    with (tmp_path / "manifest.jsonl").open("w") as stream:
        for item in items:
            sha = hashlib.sha256((twin_a if item == twin_b else item).encode()).hexdigest()
            stream.write(
                json.dumps(
                    {"item_id": item, "path": f"images/{item[:2]}/{item}.png", "sha256": sha}
                )
                + "\n"
            )
    manifest = _build(raw, tmp_path / "p", download_manifest=tmp_path / "manifest.jsonl")
    protocol = tmp_path / "p"
    assert manifest["audit"]["passes"]
    assert all(v == 0 for row in manifest["audit"]["cross_split"].values() for v in row.values())
    rows = _split_rows(protocol)
    owner = {}
    for split, values in rows.items():
        for row in values:
            for item in row["items"]:
                assert owner.setdefault(item["item_id"], split) == split
    present = [owner.get(twin_a), owner.get(twin_b)]
    assert None in present or present[0] == present[1]
    dataset = PolyvoreDisjoint(protocol, protocol, "pog_clean")
    for split in dataset.SPLITS:
        assert len(dataset.compatibility(split)) == 2 * len(dataset.outfits(split))
        assert len(dataset.fitb(split)) == len(dataset.outfits(split)) > 0
    paths = json.loads((protocol / "image_paths.json").read_text())
    assert set(paths) == set(owner) and all(p.endswith(".png") for p in paths.values())
    assert manifest["retention"]["stages"][0]["outfits_dropped"] >= 0
    total = sum(row["outfits"] for row in manifest["splits"].values())
    assert (
        total
        == manifest["retention"]["final_outfits"]
        == manifest["retention"]["stages"][-1]["outfits_remaining"]
    )


def test_two_builds_are_hash_identical_and_seed_matters(raw, tmp_path):
    _build(raw, tmp_path / "a", user_clicks=raw[0] / "clicks.json")
    _build(raw, tmp_path / "b", user_clicks=raw[0] / "clicks.json")
    assert protocol_digest(tmp_path / "a") == protocol_digest(tmp_path / "b")
    _build(raw, tmp_path / "c", user_clicks=raw[0] / "clicks.json", seed=11)
    assert (
        protocol_digest(tmp_path / "a")["train.json"]
        != protocol_digest(tmp_path / "c")["train.json"]
    )
    with pytest.raises(FileExistsError):
        _build(raw, tmp_path / "a")


def test_collision_audit_detects_negative_equal_to_positive(raw, tmp_path):
    _build(raw, tmp_path / "p")
    tampered = tmp_path / "t"
    shutil.copytree(tmp_path / "p", tampered)
    path = tampered / "compatibility_train.txt"
    lines = path.read_text().splitlines()
    positives = [line for line in lines if line.startswith("1 ")]
    first_negative = next(i for i, line in enumerate(lines) if line.startswith("0 "))
    lines[first_negative] = "0" + positives[0][1:]
    path.write_text("\n".join(lines) + "\n")
    report = audit_protocol(tampered)
    assert report["splits"]["train"]["cp_negative_equals_positive"] == 1
    assert not report["passes"]


def test_unavailable_items_drop_their_outfits(raw, tmp_path):
    _, target = raw
    items = sorted({item for _, values in target for item in values})
    missing = set(items[::25])
    with (tmp_path / "manifest.jsonl").open("w") as stream:
        for item in items:
            if item not in missing:
                stream.write(
                    json.dumps(
                        {
                            "item_id": item,
                            "path": f"images/{item[:2]}/{item}.webp",
                            "sha256": hashlib.sha256(item.encode()).hexdigest(),
                        }
                    )
                    + "\n"
                )
        stream.write('{"item_id": "torn')  # interrupted final line
    manifest = _build(raw, tmp_path / "p", download_manifest=tmp_path / "manifest.jsonl")
    expected = sum(
        any(item in missing for item in values) for _, values in target if len(values) >= 2
    )
    stage = {row["rule"]: row["outfits_dropped"] for row in manifest["retention"]["stages"]}
    assert stage["item_image_unavailable"] == expected > 0
    assert manifest["inputs"]["download_manifest"]["invalid_or_torn_lines"] == 1
    used = {
        item["item_id"]
        for rows in _split_rows(tmp_path / "p").values()
        for row in rows
        for item in row["items"]
    }
    assert not used & missing


def test_group_ids_user_rule_and_component_fallback(raw, tmp_path):
    root, _ = raw
    clicks = json.loads((root / "clicks.json").read_text())["outfits"]
    _build(raw, tmp_path / "u", user_clicks=root / "clicks.json")
    for split, rows in _split_rows(tmp_path / "u").items():
        groups = json.loads((tmp_path / "u" / f"group_ids_{split}.json").read_text())
        assert set(groups) == {row["set_id"] for row in rows}
        for outfit_id, group in groups.items():
            if outfit_id in clicks:
                assert group == "user:" + clicks[outfit_id]["min_user_id"]
            else:
                assert group == "unclicked:" + outfit_id
    manifest = _build(raw, tmp_path / "c")
    assert manifest["settings"]["group_rule"].startswith("item_component")
    for split, rows in _split_rows(tmp_path / "c").items():
        groups = json.loads((tmp_path / "c" / f"group_ids_{split}.json").read_text())
        assert set(groups) == {row["set_id"] for row in rows}
        assert all(value.startswith(f"component:{split}:") for value in groups.values())
        # outfits sharing an item share a component group
        by_item = {}
        for row in rows:
            for item in row["items"]:
                assert (
                    by_item.setdefault(item["item_id"], groups[row["set_id"]])
                    == groups[row["set_id"]]
                )


def test_scan_user_clicks_takes_smallest_user():
    lines = ["u9,a;b,o1\n", "u2,c,o1;o2\n", "u5,,o3\n", "bad\n"]
    result = scan_user_clicks(lines, {"o1", "o2"})
    assert result["outfits"] == {
        "o1": {"min_user_id": "u2", "click_lines": 2},
        "o2": {"min_user_id": "u2", "click_lines": 1},
    }
    assert result["scan_counts"]["malformed_lines"] == 1


def test_category_map_is_recorded_and_must_be_complete(raw, tmp_path):
    mapping = {f"cate_{c}": ("upper" if c == "top" else "other_" + c) for c in CATEGORIES}
    manifest = _build(raw, tmp_path / "m", category_map=mapping, category_map_name="test_map")
    assert manifest["settings"]["category_map"]["name"] == "test_map"
    assert manifest["settings"]["category_map"]["sha256"] != "identity"
    with pytest.raises(KeyError):
        _build(raw, tmp_path / "bad", category_map={"cate_top": "upper"})


def test_cli_double_build(raw, tmp_path):
    root, _ = raw
    command = [
        sys.executable,
        "-I",
        "scripts/v2/build_pog_protocol.py",
        "--items",
        str(root / "item_data.txt"),
        "--outfits",
        str(root / "outfit_data.txt"),
        "--output-a",
        str(tmp_path / "a"),
        "--output-b",
        str(tmp_path / "b"),
        "--report",
        str(tmp_path / "report.json"),
        "--target-outfits",
        str(TARGET),
        "--min-category-images",
        str(MIN_IMAGES),
    ]
    subprocess.run(
        command, check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "PASS" and report["builds_hash_identical"]
    rerun = subprocess.run(
        command, check=False, capture_output=True, cwd=Path(__file__).resolve().parents[1]
    )
    assert rerun.returncode != 0 and b"Refusing to overwrite" in rerun.stderr
