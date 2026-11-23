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
    if path.is_dir() and not (path / "items.npy").exists() and not any(path.glob("*_item_ids.json")):
        parts = sorted(p for p in path.iterdir() if (p / "items.npy").exists())
        if parts:  # Phase 2 layout: <model>/trainvalid/ + <model>/test/ (or all/)
            merged, provenance = {}, {"parts": {}}
            for part in parts:
                store, prov = load_embedding_store(part)
                provenance["parts"][part.name] = prov
                for item, vector in store.items():
                    if item in merged and not np.array_equal(merged[item], vector):
                        raise ValueError(f"Conflicting vectors for {item} across {path}")
                    merged[item] = vector
            return merged, provenance
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


def store_directories(folder):
    """Every v2 extraction directory (``items.npy``) under ``folder``: the folder itself
    and/or its immediate subdirectories (e.g. ``trainvalid/`` and ``test/``), sorted."""
    folder = Path(folder)
    found = [folder] if (folder / "items.npy").is_file() else []
    if folder.is_dir():
        found += sorted(p for p in folder.iterdir() if p.is_dir() and (p / "items.npy").is_file())
    if not found:
        raise FileNotFoundError(f"No v2 embedding store (items.npy) in {folder} or its subdirectories")
    return found


def load_model_store(root, dataset, model, *, verify_checksum=True):
    """Merge all store directories of one (dataset, model) into item_id -> float32 vector.

    Each directory needs ``items.npy``, ``item_ids.json`` and ``metadata.json``; with
    ``verify_checksum`` the ``items_sha256`` recorded in metadata must match. An item that
    occurs in several directories must have bit-identical vectors (else ValueError).
    Returns (store, provenance) with per-directory hashes.
    """
    folder = Path(root) / dataset / model
    store, provenance = {}, {"dataset": dataset, "model": model, "directories": {}}
    for directory in store_directories(folder):
        meta_path = directory / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        digest = sha256(directory / "items.npy")
        if verify_checksum:
            if "items_sha256" not in meta:
                raise ValueError(f"{directory} lacks metadata.json items_sha256")
            if digest != meta["items_sha256"]:
                raise ValueError(f"Embedding checksum mismatch in {directory}")
        part, _ = load_embedding_store(directory)
        for item, vector in part.items():
            previous = store.get(item)
            if previous is not None and not np.array_equal(previous, vector):
                raise ValueError(f"Item {item} has conflicting embeddings across {folder} stores")
            store[item] = vector
        provenance["directories"][str(directory.relative_to(folder)) or "."] = {
            "items_sha256": digest, "items": len(part)}
    dims = {len(v) for v in store.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions across {folder} stores: {dims}")
    provenance["items"] = len(store)
    return store, provenance
