import numpy as np
from PIL import Image

from repbench.v2.images import MAX_PIXELS, QWEN_MAX_PIXELS, load_rgb, to_rgb_on_white


def test_transparent_background_becomes_white(tmp_path):
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    rgba.putpixel((4, 4), (200, 10, 10, 255))
    path = tmp_path / "x.png"
    rgba.save(path)
    out = np.asarray(load_rgb(path))
    assert out.shape == (8, 8, 3)
    assert (out[0, 0] == 255).all() and tuple(out[4, 4]) == (200, 10, 10)


def test_palette_transparency_and_plain_rgb():
    pal = Image.new("P", (4, 4), 0)
    pal.putpalette([0, 0, 0, 9, 9, 9] + [0] * 762)
    pal.info["transparency"] = 0
    assert (np.asarray(to_rgb_on_white(pal)) == 255).all()
    rgb = Image.new("RGB", (4, 4), (12, 34, 56))
    assert (np.asarray(to_rgb_on_white(rgb)) == np.asarray(rgb)).all()


def test_uniform_pixel_budget():
    assert MAX_PIXELS == 384 * 384 == QWEN_MAX_PIXELS == 144 * 32 * 32


def test_store_loader_merges_parts(tmp_path):
    import hashlib
    import json

    from repbench.v2.stores import load_embedding_store

    for part, ids in (("trainvalid", ["a", "b"]), ("test", ["b", "c"])):
        folder = tmp_path / part
        folder.mkdir()
        vectors = {"a": [1, 0], "b": [0, 1], "c": [1, 1]}
        np.save(folder / "items.npy", np.asarray([vectors[i] for i in ids], dtype=np.float32))
        (folder / "item_ids.json").write_text(json.dumps(ids))
        digest = hashlib.sha256((folder / "items.npy").read_bytes()).hexdigest()
        (folder / "metadata.json").write_text(json.dumps({"items_sha256": digest}))
    store, _ = load_embedding_store(tmp_path)
    assert sorted(store) == ["a", "b", "c"]
