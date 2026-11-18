"""Read-only helpers shared by the v2 baselines and audits: embedding stores, images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repbench.v2.state import sha256

SPLITS = ("train", "valid", "test")


def load_embedding_store(path):
    """Return (dict item_id -> float32 vector, provenance).

    Accepted layouts:
    - v2 extraction directory: ``items.npy`` + ``item_ids.json`` (scripts/v2/extract_v2.py);
    - v1/G3 cache directory: ``{split}.npy`` + ``{split}_item_ids.json`` per split;
    - ``.npz`` file with arrays ``item_ids`` and ``embeddings``.
    Items present in several files must have identical vectors.
    """
    path = Path(path)
    sources = []
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            sources.append(([str(i) for i in data["item_ids"]], data["embeddings"]))
        files = [path]
    elif (path / "items.npy").is_file():
        ids = json.loads((path / "item_ids.json").read_text())
        sources.append((ids, np.load(path / "items.npy", mmap_mode="r")))
        files = [path / "items.npy", path / "item_ids.json"]
    else:
        files = []
        for split in SPLITS:
            values = path / f"{split}.npy"
            if values.is_file():
                ids = json.loads((path / f"{split}_item_ids.json").read_text())
                sources.append((ids, np.load(values, mmap_mode="r")))
                files += [values, path / f"{split}_item_ids.json"]
        if not sources:
            raise FileNotFoundError(f"No embedding store found at {path}")
    store = {}
    for ids, values in sources:
        if values.ndim != 2 or len(ids) != len(values):
            raise ValueError(f"Embedding store shape mismatch in {path}")
        for item, row in zip(ids, values, strict=True):
            vector = np.asarray(row, dtype=np.float32)
            previous = store.get(str(item))
            if previous is not None and not np.array_equal(previous, vector):
                raise ValueError(f"Item {item} has conflicting embeddings across files")
            store[str(item)] = vector
    dims = {len(v) for v in store.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")
    for vector in store.values():
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite embedding in store")
    provenance = {"path": str(path), "sha256": {str(f): sha256(f) for f in files}}
    return store, provenance


def image_resolver(images_root, protocol):
    """Same rule as scripts/v2/extract_v2.py: protocol image_paths.json, else {id}.jpg."""
    images_root, protocol = Path(images_root), Path(protocol)
    mapping_path = protocol / "image_paths.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        return lambda item: images_root / mapping[item]
    return lambda item: images_root / f"{item}.jpg"


def protocol_hashes(protocol, names=None):
    protocol = Path(protocol)
    if names is None:
        names = sorted(p.name for p in protocol.iterdir() if p.is_file())
    return {name: sha256(protocol / name) for name in names if (protocol / name).is_file()}
