import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance


def _script():
    spec = importlib.util.spec_from_file_location(
        "near_dup", Path("scripts/v2/near_duplicate_audit.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ND = _script()


def pattern(seed, size=128):
    """Random low-frequency image: distinct seeds give unrelated pHashes."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8))
    return small.resize((size, size), Image.Resampling.BICUBIC)


def modified(image):
    noisy = np.asarray(ImageEnhance.Brightness(image).enhance(1.06), dtype=np.int16)
    noisy = noisy + np.random.default_rng(0).integers(-6, 7, noisy.shape)
    out = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    return out.resize((120, 120), Image.Resampling.BILINEAR)


def test_phash_identical_modified_different():
    base = pattern(1)
    assert ND.phash(base) == ND.phash(base.copy())
    assert ND.hamming(ND.phash(base), ND.phash(modified(base))) <= 4
    distances = [ND.hamming(ND.phash(pattern(1)), ND.phash(pattern(s))) for s in range(2, 30)]
    assert min(distances) > 10
    rgba = base.convert("RGBA")
    assert ND.phash(rgba) == ND.phash(base)
    assert 0 <= ND.phash(base) < 2**64


def test_multi_index_hashing_matches_brute_force():
    rng = np.random.default_rng(3)
    base = rng.integers(0, 2**63, 300, dtype=np.uint64) * np.uint64(2) + rng.integers(
        0, 2, 300, dtype=np.uint64
    )
    planted = []
    for value in base[:60]:
        flips = rng.choice(64, size=rng.integers(1, 7), replace=False)
        mask = 0
        for f in flips:
            mask |= 1 << int(f)
        planted.append(np.uint64(int(value) ^ mask))
    hashes = np.unique(np.concatenate([base, np.asarray(planted, dtype=np.uint64)]))
    for threshold in (0, 2, 4, 6):
        fast = {tuple(p) for p in ND.near_pairs(hashes, threshold).tolist()}
        brute = {
            (i, j)
            for i, j in itertools.combinations(range(len(hashes)), 2)
            if ND.hamming(hashes[i], hashes[j]) <= threshold
        }
        assert fast == brute
    assert sum(w for _, w in ND.band_layout(4)) == 64 and len(ND.band_layout(4)) == 5


def write_protocol(root):
    images = root / "images"
    images.mkdir()
    protocol = root / "protocol"
    protocol.mkdir()
    seeds = iter(range(100, 200))
    layout = {"train": ["a", "b"], "valid": ["c", "d"], "test": ["e", "f", "g"]}
    store = {}
    for split, sets in layout.items():
        outfits = []
        for set_id in sets:
            items = []
            for k in (1, 2):
                item = f"{set_id}{k}"
                pattern(next(seeds)).save(images / f"{item}.jpg", quality=95)
                store[item] = np.random.default_rng(len(store)).normal(size=8)
                items.append({"item_id": item, "index": k})
            outfits.append({"set_id": set_id, "items": items})
        (protocol / f"{split}.json").write_text(json.dumps(outfits))
    # Test outfit e: exact copy of train image a1. Outfit f: slightly modified valid c2.
    Image.open(images / "a1.jpg").save(images / "e1.jpg", quality=95)
    with Image.open(images / "c2.jpg") as image:
        modified(image.convert("RGB")).save(images / "f2.jpg", quality=90)
    # Embedding-only near duplicate: valid d1 is ~parallel to train b2.
    store["d1"] = store["b2"] + 0.01
    np.savez(
        root / "store.npz",
        item_ids=np.asarray(list(store)),
        embeddings=np.asarray(list(store.values()), dtype=np.float32),
    )
    return protocol, images


def test_end_to_end_audit(tmp_path):
    protocol, images = write_protocol(tmp_path)
    out = tmp_path / "audit"
    args = [
        "--protocol-root",
        str(protocol),
        "--dataset-name",
        "synthetic_clean",
        "--images-root",
        str(images),
        "--output-dir",
        str(out),
        "--workers",
        "0",
        "--embeddings",
        str(tmp_path / "store.npz"),
    ]
    ND.main(args)
    audit = json.loads((out / "audit.json").read_text())
    assert audit["phash"]["pair_counts"] == {"train-test": 1, "valid-test": 1, "train-valid": 0}
    assert audit["embedding"]["pair_counts"]["train-valid"] == 1
    assert audit["shared_item_ids"] == {"train-test": 0, "valid-test": 0, "train-valid": 0}
    affected = json.loads((out / "affected_test_outfits.json").read_text())
    assert affected["test_items"]["phash_train"] == ["e1"]
    assert affected["test_items"]["phash_valid"] == ["f2"]
    assert affected["test_outfits"]["phash_any"] == ["e", "f"]
    assert "g" not in affected["test_outfits"]["any_method"]
    pairs = (out / "near_duplicate_pairs.csv").read_text().splitlines()
    assert any(line.startswith("phash_hamming,a1,train,e1,test,0") for line in pairs)
    assert audit["runtime"]["total_seconds"] > 0
    with pytest.raises(FileExistsError):
        ND.main(args)
