"""v2-Clean near-duplicate image equivalence (owner decision E).

Covers scripts/v2/image_equivalence.py (signatures, exact multi-index candidate search,
pixel-level confirmation, union-find groups, CLI determinism/refusal), the optional
``image_equivalence`` parameter of the three clean builders (default byte-identical to
the pre-change code; groups never split), and scripts/v2/build_v2_clean.py.
"""

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_clean_fixtures as FX

from repbench.data import iqon
from repbench.data.clean_protocol import SPLITS, construct_clean_protocol
from repbench.data.image_equivalence import (
    groups_spanning_splits,
    merge_identities,
)
from repbench.v2.pog import construct_pog_clean

REPO = Path(__file__).resolve().parents[1]


def _script(name, module):
    if module in sys.modules:
        return sys.modules[module]
    spec = importlib.util.spec_from_file_location(module, REPO / "scripts" / "v2" / name)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module] = loaded
    spec.loader.exec_module(loaded)
    return loaded


IE = _script("image_equivalence.py", "v2_image_equivalence_under_test")
BUILD = _script("build_v2_clean.py", "v2_build_clean_under_test")


# ------------------------------------------------------------------ synthetic images


def product(colour=(200, 40, 40), shape="dress", size=240, seed=0):
    """A product-like photo: textured coloured shape on a white background."""
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    s = size / 240
    if shape == "dress":
        draw.polygon([(90 * s, 30 * s), (150 * s, 30 * s), (200 * s, 220 * s), (40 * s, 220 * s)], fill=colour)
    elif shape == "bag":
        draw.rectangle([40 * s, 90 * s, 200 * s, 200 * s], fill=colour)
        draw.arc([80 * s, 30 * s, 160 * s, 130 * s], 180, 360, fill=colour, width=int(8 * s))
    else:  # shoe
        draw.ellipse([20 * s, 120 * s, 220 * s, 190 * s], fill=colour)
        draw.rectangle([150 * s, 60 * s, 190 * s, 150 * s], fill=colour)
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.int16)
    mask = pixels.min(axis=-1) < 250
    noise = rng.integers(-25, 26, pixels.shape[:2])[..., None]
    pixels = np.where(mask[..., None], pixels + noise, pixels)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def jpeg_bytes(image, quality=90):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def transparent_png(colour=(200, 40, 40)):
    """Same dress on a transparent (black RGB) background: must equal the white one."""
    opaque = product(colour)
    alpha = Image.fromarray(((np.asarray(opaque).min(axis=-1) < 250) * 255).astype(np.uint8))
    rgba = Image.new("RGBA", opaque.size, (0, 0, 0, 0))
    rgba.paste(opaque, mask=alpha)
    return png_bytes(rgba)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("equivalence_images")
    base = product()
    files = {
        "a_original": jpeg_bytes(base),
        "b_exact_copy": jpeg_bytes(base),
        "c_reencoded_resized": jpeg_bytes(base.resize((300, 300), Image.Resampling.BICUBIC), 70),
        "d_transparent_png": transparent_png(),
        "e_navy_variant": jpeg_bytes(product(colour=(30, 40, 110))),
        "f_black_variant": jpeg_bytes(product(colour=(25, 25, 25))),
        "g_other_item_bag": jpeg_bytes(product(colour=(200, 40, 40), shape="bag", seed=3)),
        "h_other_item_shoe": jpeg_bytes(product(colour=(40, 160, 60), shape="shoe", seed=4)),
        "i_corrupt": b"not an image",
    }
    lines = []
    for name, data in files.items():
        path = root / f"{name}.img"
        path.write_bytes(data)
        lines.append(f"{name}\t{path}")
    (root / "items.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _signatures(library):
    return IE.compute_signatures(IE.tsv_pool(library / "items.tsv"), workers=0)


# ------------------------------------------------------------------ the rule


def test_rule_on_synthetic_cases(library):
    signatures, failures = _signatures(library)
    assert [f["item_id"] for f in failures] == ["i_corrupt"]
    groups = IE.equivalence(signatures)["groups"]
    same = {k for k, v in groups.items() if v == "a_original"}
    assert same == {"a_original", "b_exact_copy", "c_reencoded_resized", "d_transparent_png"}
    for other in ("e_navy_variant", "f_black_variant", "g_other_item_bag", "h_other_item_shoe"):
        assert groups.get(other, other) != "a_original"
    assert groups.get("e_navy_variant") is None or groups["e_navy_variant"] != groups.get(
        "f_black_variant"
    )


def test_colour_variant_is_a_phash_candidate_but_rejected_by_pixels(library):
    signatures, _ = _signatures(library)
    ids = list(signatures["item_ids"])
    result = IE.equivalence(signatures)
    pairs = {(ids[a], ids[b]): k for k, (a, b) in enumerate(result["pairs"])}
    key = ("a_original", "e_navy_variant")
    assert key in pairs, "same shape, other colour must reach the pixel check"
    assert not result["confirmed"][pairs[key]]
    assert result["metrics"]["colour_dab"][pairs[key]] > IE.DEFAULT_COLOUR


def test_candidate_pairs_match_brute_force():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 2**63, 40, dtype=np.uint64)
    hashes = list(base)
    for value in base[:15]:  # near and identical copies
        flip = np.uint64(1) << np.uint64(int(rng.integers(0, 64)))
        flip2 = np.uint64(1) << np.uint64(int(rng.integers(0, 64)))
        hashes += [value, value ^ flip, value ^ flip ^ flip2]
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs, distance, _ = IE.candidate_pairs(hashes, 2)
    brute = {
        (i, j)
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if (int(hashes[i]) ^ int(hashes[j])).bit_count() <= 2
    }
    assert {tuple(map(int, p)) for p in pairs} == brute
    for (i, j), d in zip(pairs, distance, strict=True):
        assert d == (int(hashes[i]) ^ int(hashes[j])).bit_count()


def test_union_groups_transitive_with_min_id():
    ids = np.asarray(["z", "b", "c", "a", "q"])
    assert IE.union_groups(ids, [(0, 1), (1, 2), (3, 4)]) == {
        "a": "a", "b": "b", "c": "b", "q": "a", "z": "b"
    }
    assert IE.union_groups(ids, []) == {}


def test_signature_cache_roundtrip(library, tmp_path):
    signatures, failures = _signatures(library)
    IE.save_signatures(tmp_path / "s.npz", signatures, failures)
    with pytest.raises(FileExistsError):
        IE.save_signatures(tmp_path / "s.npz", signatures, failures)
    loaded, loaded_failures = IE.load_signatures(tmp_path / "s.npz")
    assert loaded_failures == failures
    assert IE.input_sha256(loaded) == IE.input_sha256(signatures)
    assert IE.equivalence(loaded)["groups"] == IE.equivalence(signatures)["groups"]


def test_cli_outputs_deterministic_and_refuses_overwrite(library, tmp_path):
    common = ["--dataset", "tsv", "--items-tsv", str(library / "items.tsv")]
    IE.main(common + ["--output-dir", str(tmp_path / "a"), "--workers", "2"])
    IE.main(common + ["--output-dir", str(tmp_path / "b"), "--workers", "0"])
    for name in ("equivalence_groups.json", "candidate_pairs.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    report = json.loads((tmp_path / "a" / "equivalence_report.json").read_text())
    assert report["thresholds"]["phash_hamming_max"] == 2
    assert report["pairs"]["candidates"] >= report["pairs"]["confirmed"] > 0
    assert report["groups"]["size_distribution"] == {"4": 1}
    assert len(report["unreadable_images"]) == 1
    assert len(report["input_sha256"]) == 64
    with pytest.raises(FileExistsError):
        IE.main(common + ["--output-dir", str(tmp_path / "a")])


def test_polyvore_pool_from_zip_and_raw_root(tmp_path):
    raw = FX.make_polyvore_raw(tmp_path / "raw")
    from_root = IE.polyvore_pool(raw_root=raw)
    archive = tmp_path / "polyvore_outfits.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                stream.write(path, "dataset/polyvore_outfits/" + str(path.relative_to(raw)))
        stream.writestr("dataset/polyvore_outfits/nondisjoint/train.json", "[]")
    from_zip = IE.polyvore_pool(polyvore_zip=archive)
    assert [r[0] for r in from_root] == [r[0] for r in from_zip] and len(from_zip) == 171
    a, _ = IE.compute_signatures(from_root[:20], workers=0)
    b, _ = IE.compute_signatures(from_zip[:20], workers=0)
    assert IE.input_sha256(a) == IE.input_sha256(b)


def test_pog_pool_checks_manifest_hashes(tmp_path):
    inputs = FX.make_pog_raw(tmp_path / "pog", with_images=True)
    pool = IE.pog_pool(
        inputs["items_path"], inputs["outfits_path"], inputs["download_manifest"], tmp_path / "pog",
        target_outfits=FX.POG_TARGET,
    )
    assert pool and all(expected is not None for _, _, expected in pool)
    victim = Path(pool[0][1][1])
    victim.write_bytes(png_bytes(product()))  # bytes no longer match the manifest
    _, failures = IE.compute_signatures(pool[:5], workers=0)
    assert [f["item_id"] for f in failures] == [pool[0][0]]
    assert failures[0]["error"].startswith("ChecksumMismatch")


# ------------------------------------------------------------------ helpers


def test_merge_identities_and_spanning_audit():
    exact = {"a": "s1", "b": "s1", "c": "s2", "d": "s3", "e": "s4"}
    merged = merge_identities(exact, {"b": "g", "c": "g", "x": "g", "e": "h"})
    assert merged == {"a": "near:s1", "b": "near:s1", "c": "near:s1", "d": "s3", "e": "s4"}
    assert merge_identities(exact, {}) == exact
    audit = groups_spanning_splits({"train": ["a", "b"], "test": ["c"]}, {"b": "g", "c": "g"})
    assert audit["groups_spanning_splits"] == 1


# ------------------------------------------------------------------ builders: default


GOLDEN = {  # produced by the pre-change builders (git HEAD b2e71a8) on the fixtures
    "polyvore": {
        "compatibility_test.txt": "ca97e53ec6bff3600f4807af34cc78a9a0a129274930eee19b51bf0ee42cb524",
        "compatibility_train.txt": "5a2bf73269d811d36aebffea5f930c41f1d4e70f6c16d9dba5ac3f0dc8e7f16d",
        "compatibility_valid.txt": "993429b0bd4d708d6565d24b7704da4bf31a7b14c3f9de1cd12421881f5b29e6",
        "duplicates_removed.json": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "fill_in_blank_test.json": "46b29ad231269519253d695ab75a6dbb04d515f11931c5a4afbadc206e1ff904",
        "fill_in_blank_train.json": "559f80f808b49b46378389a3c0a00d17f5acee9a241da861b9158e9dabc04da2",
        "fill_in_blank_valid.json": "90c251db980bc988dc761d7626fd3354483563c42a84fd4eb4c02dde1015ca32",
        "manifest.json": "757b36e8f61a3fec7749472e0bc224efc3e067b8090957f42ee8ba91384b8802",
        "test.json": "d6bcaef34f9505fca9893f3164e56f83fcb9401287315e575dda09dcb9fc69c6",
        "train.json": "14060aa4f6cd70b029fe09603c50c8bc1013f919c3bfc59b9cde6dffd2f0cfa4",
        "valid.json": "5ef03b56b26027228c27e74a504b200cc45e82885a8fd164c7c9eba2913b11e8",
    },
    "iqon": {
        "compatibility_test.txt": "c47472040b7659b8286d4ccae627c042f9386d5e1c4f3b1d4de7badeb5754f09",
        "compatibility_train.txt": "698815828e617942b96d17407ed2feda2e69dcba34e2981d0409245915485b51",
        "compatibility_valid.txt": "fce66a22952e7722c967d9f94851df37d71e2e019a5d284075c6e6766b09ebaf",
        "fill_in_blank_test.json": "325f7ae4691383b47a683356ac622bd4666131aae892da09248cbc3d50519360",
        "fill_in_blank_train.json": "e007c25d0d6d9ca82ee410481313c2e0781db9980273b5b151a086e19abf21b9",
        "fill_in_blank_valid.json": "bcec2fcfbe8f85d899f59246fea420c6301b6f02744b962aa0542b22cad53b77",
        "group_ids_test.json": "cedbc3e1e43fc18f354974a293c44317710049c686ee8585a0562820d59b946a",
        "group_ids_train.json": "758beee0580c7985a98cdd9ac587d401079e084564b15721b9280cb3875b499c",
        "group_ids_valid.json": "97a9216d8c5e9f6ca8c67768f9bf66f779efed734fa08ff442902281da6f4684",
        "test.json": "9b3eff2092b902d414a809be205de7db187a28a763bb433d51eb3ebc8a0803ce",
        "train.json": "3dd4bd252963cded0ce5f771605f5f7136ee24608317db880350a16befab95f2",
        "valid.json": "a268fbcccd11e6eac34b6a16e3bb075410ca008b070f7db1957a383424b11a9c",
        "manifest.json(without source_archive)": "8145aaff333152f07bda911d7e4ad16a3ddc8521c861973a563c9e1367213b5d",
    },
    "pog": {
        "compatibility_test.txt": "2dbbbdd59cbe5fc97ec74a6410527610e37b7612f1b9991eb754536f850484f1",
        "compatibility_train.txt": "46a4b58c91aaa6c865a8b1a975e91bef0a171b55df9a800c573ea85ce4f60a2b",
        "compatibility_valid.txt": "1a498137c775ad64d73937b142231ea37e3fb20879e7308fe38e34884d27a54f",
        "fill_in_blank_test.json": "a726e2a413de2eb69bfd2e5c748b9470bbbb099d336dda9059dc1632c68623e1",
        "fill_in_blank_train.json": "799c89109c70744bebc275b9801c484aad3a9830d4b4a855ca4ed9f160254a50",
        "fill_in_blank_valid.json": "4d9d1ba3c5168c694b8ead84434d7fe82911f3a26bd9fcb0fe50e9bb0412e870",
        "group_ids_test.json": "87011506a55395bb75f8e29a0377db60bdddc3ac8364672716c5f4130d60b3f9",
        "group_ids_train.json": "35625e85ad92b89abd399bc5239f9365b693cee5c5d52d013e506efc0f14bec0",
        "group_ids_valid.json": "b9e1b33b0fa720c38a0c43753c29ef592e8444b61b51146958d115ee67c84e69",
        "image_paths.json": "f48901dce0d8062764e99578d665fc49cddb43fac4d909ef3dc97f8e99ac9102",
        "manifest.json": "f5d43f24c2a82078fcd037a0189cb43da385b6eae81a97d8c1b16d3f43321698",
        "polyvore_item_metadata.json": "1d69972233b07b609b9a9adb2f59eaf31a283f4f3c46111da194121693d85605",
        "retention_dropped_outfits.json": "ce4e5f2d75ee1c11c2641ad7a4bfb6105b3f325f975d19a4ef795179af250b91",
        "test.json": "72f93468289f6de277fd165431d0586646858843bfe53b171aa78b61f18937f1",
        "train.json": "bd8364e81020e6514f650c6e49c46724d8e6b892fde8c477eb4e8e4353103269",
        "valid.json": "4e808dc4d75e3ec11cb24be123a6abf83d2bad46585d511cd53ddfeca02825da",
    },
}


def _pog_build(inputs, output, **kwargs):
    return construct_pog_clean(
        inputs["items_path"], inputs["outfits_path"], output,
        download_manifest=inputs["download_manifest"], target_outfits=FX.POG_TARGET,
        min_category_groups=5, **kwargs,
    )


def test_polyvore_default_output_unchanged(tmp_path):
    raw = FX.make_polyvore_raw(tmp_path / "raw")
    construct_clean_protocol(raw, tmp_path / "out")
    assert FX.file_digests(tmp_path / "out") == GOLDEN["polyvore"]


def test_iqon_default_output_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(iqon, "TARGET_COUNTS", FX.IQON_TARGETS)
    inputs = FX.make_iqon_raw(tmp_path / "raw")
    iqon.construct_iqon_clean(output_root=tmp_path / "out", **inputs)
    assert FX.iqon_digest(tmp_path / "out") == GOLDEN["iqon"]


def test_pog_default_output_unchanged(tmp_path):
    inputs = FX.make_pog_raw(tmp_path / "raw")
    _pog_build(inputs, tmp_path / "out")
    assert FX.file_digests(tmp_path / "out") == GOLDEN["pog"]


# ------------------------------------------------------------------ builders: groups


def _owners(protocol):
    owner = {}
    for split in SPLITS:
        for row in json.loads((protocol / f"{split}.json").read_text(encoding="utf-8")):
            for item in row["items"]:
                assert owner.setdefault(str(item["item_id"]), split) == split
    return owner


def test_polyvore_equivalence_joins_units(tmp_path):
    raw = FX.make_polyvore_raw(tmp_path / "raw")
    _, bottoms, shoes = FX.polyvore_ids()
    plain = construct_clean_protocol(raw, tmp_path / "plain")
    owner = _owners(tmp_path / "plain")
    # Pick items that the default build puts in three different splits.
    by_split = {s: sorted(i for i in bottoms + shoes if owner.get(i) == s) for s in SPLITS}
    group = [by_split["train"][-1], by_split["valid"][0], by_split["test"][0]]
    mapping = {item: group[0] for item in group} | {"999999": "999999"}
    info = {"file": "groups.json", "sha256": "0" * 64}
    manifest = construct_clean_protocol(
        raw, tmp_path / "v2", image_equivalence=mapping, image_equivalence_info=info,
        protocol_name="polyvore_v2_clean",
    )
    owner_v2 = _owners(tmp_path / "v2")
    assert len({owner_v2[item] for item in group}) == 1
    assert manifest["protocol"] == "polyvore_v2_clean" and plain["protocol"] == "polyvore_d_clean"
    record = manifest["image_equivalence"]
    assert record["pool_groups"] == 1 and record["pool_items_in_groups"] == 3
    assert record["sha256"] == "0" * 64 and len(record["mapping_sha256"]) == 64
    assert sum(manifest["splits"][s]["outfits"] for s in SPLITS) == 60


def test_iqon_equivalence_joins_units(tmp_path, monkeypatch):
    monkeypatch.setattr(iqon, "TARGET_COUNTS", FX.IQON_TARGETS)
    inputs = FX.make_iqon_raw(tmp_path / "raw")
    iqon.construct_iqon_clean(output_root=tmp_path / "plain", **inputs)
    owner = _owners(tmp_path / "plain")
    by_split = {s: sorted(i for i in owner if owner[i] == s and i.startswith("i1_")) for s in SPLITS}
    group = [by_split["train"][0], by_split["valid"][0], by_split["test"][0]]
    manifest = iqon.construct_iqon_clean(
        output_root=tmp_path / "v2", image_equivalence={i: "g" for i in group},
        protocol_name="iqon3000_v2_clean", **inputs,
    )
    owner_v2 = _owners(tmp_path / "v2")
    present = [i for i in group if i in owner_v2]
    assert len(present) >= 2 and len({owner_v2[i] for i in present}) == 1
    assert manifest["protocol"] == "iqon3000_v2_clean"
    assert manifest["image_equivalence"]["pool_groups"] == 1
    assert all(row["image_equivalence_nodes"] == 0 for row in manifest["split_overlap"].values())
    assert {s: manifest["splits"][s]["outfits"] for s in SPLITS} == FX.IQON_TARGETS


def test_pog_equivalence_joins_units_and_audit_sees_groups(tmp_path):
    inputs = FX.make_pog_raw(tmp_path / "raw")
    _pog_build(inputs, tmp_path / "plain")
    owner = _owners(tmp_path / "plain")
    by_split = {s: sorted(i for i in owner if owner[i] == s) for s in SPLITS}
    group = [by_split["train"][0], by_split["valid"][0], by_split["test"][0]]
    manifest = _pog_build(
        inputs, tmp_path / "v2", image_equivalence={i: "g" for i in group},
        protocol_name="pog_v2_clean",
    )
    owner_v2 = _owners(tmp_path / "v2")
    present = [i for i in group if i in owner_v2]
    assert len(present) >= 2 and len({owner_v2[i] for i in present}) == 1
    assert manifest["audit"]["passes"] and manifest["protocol"] == "pog_v2_clean"
    metadata = json.loads((tmp_path / "v2" / "polyvore_item_metadata.json").read_text())
    assert len({metadata[i]["image_identity"] for i in present}) == 1
    assert all(metadata[i]["image_identity"].startswith("near:") for i in present)
    assert all("exact_image_identity" in row for row in metadata.values())


# ------------------------------------------------------------------ build_v2_clean


def _near_duplicate_polyvore(tmp_path):
    """Fixture raw root whose images include re-encoded copies across historical splits."""
    raw = FX.make_polyvore_raw(tmp_path / "raw")
    _, bottoms, _ = FX.polyvore_ids()
    twins = ((bottoms[12], bottoms[44], "dress"), (bottoms[13], bottoms[50], "bag"))
    for k, (source, copy, shape) in enumerate(twins):
        image = product(colour=(200, 40 + 100 * k, 40), shape=shape, seed=10 + k)
        (raw / "images" / f"{source}.jpg").write_bytes(jpeg_bytes(image))
        (raw / "images" / f"{copy}.jpg").write_bytes(
            jpeg_bytes(image.resize((200, 200), Image.Resampling.BICUBIC), 75)
        )
    return raw, [(bottoms[12], bottoms[44]), (bottoms[13], bottoms[50])]


def test_build_v2_clean_polyvore_end_to_end(tmp_path):
    raw, twins = _near_duplicate_polyvore(tmp_path)
    eq = tmp_path / "eq"
    IE.main(["--dataset", "polyvore", "--raw-root", str(raw), "--output-dir", str(eq), "--workers", "0"])
    groups = json.loads((eq / "equivalence_groups.json").read_text())
    for a, b in twins:
        assert groups[a] == groups[b]
    out = tmp_path / "v2"
    BUILD.main([
        "--dataset", "polyvore_v2_clean", "--raw-root", str(raw), "--output", str(out),
        "--image-equivalence", str(eq / "equivalence_groups.json"),
        "--equivalence-report", str(eq / "equivalence_report.json"),
    ])
    report = json.loads((out / "build_report.json").read_text())
    assert report["status"] == "PASS" and report["builds_hash_identical"]
    assert report["audits"]["image_equivalence_not_split"]["groups_spanning_splits"] == 0
    assert report["image_equivalence"]["thresholds"]["phash_hamming_max"] == 2
    manifest = json.loads((out / "build_a" / "manifest.json").read_text())
    assert manifest["protocol"] == "polyvore_v2_clean"
    assert manifest["image_equivalence"]["input_sha256"] == json.loads(
        (eq / "equivalence_report.json").read_text())["input_sha256"]
    owner = _owners(out / "build_a")
    for a, b in twins:
        assert owner[a] == owner[b]
    with pytest.raises(FileExistsError):
        BUILD.main([
            "--dataset", "polyvore_v2_clean", "--raw-root", str(raw), "--output", str(out),
            "--image-equivalence", str(eq / "equivalence_groups.json"),
            "--equivalence-report", str(eq / "equivalence_report.json"),
        ])


def test_build_v2_clean_rejects_mismatched_equivalence(tmp_path):
    raw, _ = _near_duplicate_polyvore(tmp_path)
    eq = tmp_path / "eq"
    IE.main(["--dataset", "polyvore", "--raw-root", str(raw), "--output-dir", str(eq), "--workers", "0"])
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({"1": "1", "2": "1"}))
    with pytest.raises(ValueError, match="does not match"):
        BUILD.main([
            "--dataset", "polyvore_v2_clean", "--raw-root", str(raw), "--output", str(tmp_path / "x"),
            "--image-equivalence", str(tampered),
            "--equivalence-report", str(eq / "equivalence_report.json"),
        ])
    with pytest.raises(ValueError, match="not 'pog'"):
        BUILD.main([
            "--dataset", "pog_v2_clean", "--items", "i", "--outfits", "o",
            "--download-manifest", "m", "--output", str(tmp_path / "y"),
            "--image-equivalence", str(eq / "equivalence_groups.json"),
            "--equivalence-report", str(eq / "equivalence_report.json"),
        ])
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_build_v2_clean_pog_and_iqon_end_to_end(tmp_path, monkeypatch):
    inputs = FX.make_pog_raw(tmp_path / "pog", with_images=True)
    eq = tmp_path / "pog_eq"
    IE.main([
        "--dataset", "tsv", "--items-tsv", str(_pog_tsv(inputs, tmp_path)),
        "--output-dir", str(eq), "--workers", "0",
    ])
    report = json.loads((eq / "equivalence_report.json").read_text())
    report["dataset"] = "pog"  # tsv pool of the same images; relabel for the builder check
    (eq / "pog_report.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys.modules["repbench.v2.pog"], "POG_TARGET_OUTFITS", FX.POG_TARGET)
    monkeypatch.setattr(sys.modules["repbench.v2.pog"], "MIN_CATEGORY_GROUPS", 5)
    out = tmp_path / "pog_v2"
    real = construct_pog_clean

    def small(*args, **kwargs):
        kwargs.setdefault("target_outfits", FX.POG_TARGET)
        kwargs.setdefault("min_category_groups", 5)
        return real(*args, **kwargs)

    monkeypatch.setattr(sys.modules["repbench.v2.pog"], "construct_pog_clean", small)
    BUILD.main([
        "--dataset", "pog_v2_clean", "--items", str(inputs["items_path"]),
        "--outfits", str(inputs["outfits_path"]), "--download-manifest", str(inputs["download_manifest"]),
        "--output", str(out), "--image-equivalence", str(eq / "equivalence_groups.json"),
        "--equivalence-report", str(eq / "pog_report.json"),
    ])
    assert json.loads((out / "build_report.json").read_text())["status"] == "PASS"

    monkeypatch.setattr(iqon, "TARGET_COUNTS", FX.IQON_TARGETS)
    iq = FX.make_iqon_raw(tmp_path / "iqon")
    mapping = {"i1_3": "i1_3", "i1_40": "i1_3", "i2_5": "i2_5", "i2_60": "i2_5"}
    (tmp_path / "iq_groups.json").write_text(json.dumps(mapping))
    iq_report = {
        "kind": "v2_image_equivalence_groups", "dataset": "iqon", "rule": "r",
        "thresholds": {}, "signature_version": "v", "input_sha256": "0" * 64,
        "input_items_hashed": 0, "unreadable_images": [], "pairs": {}, "groups": {},
        "outputs_sha256": {"equivalence_groups.json": hashlib.sha256(
            (tmp_path / "iq_groups.json").read_bytes()).hexdigest()},
    }
    (tmp_path / "iq_report.json").write_text(json.dumps(iq_report))
    BUILD.main([
        "--dataset", "iqon3000_v2_clean", "--catalog", str(iq["catalog_dir"]),
        "--source-audit", str(iq["source_audit_path"]), "--archive", str(iq["archive_path"]),
        "--output", str(tmp_path / "iq_v2"), "--image-equivalence", str(tmp_path / "iq_groups.json"),
        "--equivalence-report", str(tmp_path / "iq_report.json"),
    ])
    iq_build = json.loads((tmp_path / "iq_v2" / "build_report.json").read_text())
    assert iq_build["status"] == "PASS"
    assert iq_build["splits"] == FX.IQON_TARGETS


def _pog_tsv(inputs, tmp_path):
    lines = []
    root = Path(inputs["download_manifest"]).parent
    for line in Path(inputs["download_manifest"]).read_text().splitlines():
        record = json.loads(line)
        lines.append(f"{record['item_id']}\t{root / record['path']}")
    path = tmp_path / "pog.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path
