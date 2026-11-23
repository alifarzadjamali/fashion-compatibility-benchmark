"""Model-free near-duplicate image equivalence groups for v2-Clean (owner decision E).

Rule (identical for Polyvore, IQON3000 and POG; no roster encoder is involved). Two
images of a dataset's whole candidate item pool are equivalent when ALL hold:

  1. pHash: their 64-bit DCT pHash (``near_duplicate_audit.phash``; transparent
     images composited onto white) differ in at most ``--hamming`` bits (default 2).
     Pairs are found exactly with multi-index hashing (``near_duplicate_audit.near_pairs``).
  2. colour: the CIELAB chromatic distance (Delta a*b*, D65) between the two images'
     mean foreground colours is at most ``--colour-threshold`` (default 2.0).
  3. colour/shape, local: the 95th percentile, over the union foreground cells of
     their 16x16 thumbnails, of the per-cell CIELAB Delta E (CIE76) is at most
     ``--structure-threshold`` (default 15.0); a different shape or a locally
     different colour (strap, stone, print) produces large cell differences.
  4. optional: aspect ratios agree, |log(ar_a / ar_b)| <= log(1 + ``--aspect-tolerance``)
     (off by default: every Polyvore image is 300x300, so it could not be calibrated).

Pixel signature: the white-composited image is area-averaged (PIL BOX) to a 32x32 RGB
thumbnail (aspect ratio ignored); 16x16 = 2x2 means of it. Foreground = pixels whose
minimum channel is below ``--foreground-level`` (default 245); for (2) each image's own
foreground, for (3) the union of both. An image with no foreground counts as white
(two blank images are equivalent).

Confirmed pairs are closed transitively with union-find; the group ID is the smallest
item ID of the group (deterministic). Thresholds were calibrated on a labelled seeded
Polyvore sample: research/v2/readiness/2026-11-19-image-equivalence/.

Outputs (in a new ``--output-dir``; nothing is overwritten):
  equivalence_groups.json  {item_id: group_id} for items in groups of size >= 2
  equivalence_report.json  thresholds, pair counts before/after confirmation, group
                           size distribution, input hash, runtime, unreadable images
  candidate_pairs.csv      every pHash candidate pair with all metrics and the verdict
``--save-signatures`` stores pHashes and thumbnails (.npz, ~3 kB per image; never
commit) so thresholds can be re-applied later with ``--signatures``.

Image files are untrusted: each is decoded in a worker, failures are recorded per item
(not fatal), and expected SHA-256 hashes (IQON image audit, POG manifest) are checked.

Pools (all items of all outfits before splitting):
  polyvore  items of <raw-root>/disjoint/{train,valid,test}.json, images/<id>.jpg
            (or the same files read straight from the original zip: --polyvore-zip)
  iqon      items of catalog outfits.parquet in outfits whose items all have a
            decodable image (construct_iqon_clean usable-outfit rule); bytes read from
            --archive via image_audit_resolved.parquet zip_entry, checked against
            encoded_sha256 (identical bytes to the extracted images/<id>.jpg)
  pog       items of the frozen target outfits kept by construct_pog_clean's
            pre-packing rules; bytes at <images-root>/<manifest path>, checked against
            the download-manifest SHA-256
  tsv       --items-tsv lines "item_id<TAB>image path"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import multiprocessing
import sys
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))  # works under python -I


def _load_near_duplicate_audit():
    name = "repbench_v2_near_duplicate_audit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / "near_duplicate_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ND = _load_near_duplicate_audit()

THUMB = 32
POOL = 2  # 32x32 -> 16x16 for the structure check
DEFAULT_HAMMING = 2
DEFAULT_COLOUR = 2.0
DEFAULT_STRUCTURE = 15.0
DEFAULT_FOREGROUND_LEVEL = 245
DEFAULT_ASPECT_TOLERANCE = None
SIGNATURE_VERSION = "phash64-dct+thumb32-rgb-box-white-v1"

# ------------------------------------------------------------------ signatures

_ZIPS: dict[str, zipfile.ZipFile] = {}


def _read_source(source):
    """source: ("file", path) or ("zip", archive, entry)."""
    if source[0] == "file":
        return Path(source[1]).read_bytes()
    if source[0] == "zip":
        archive = _ZIPS.get(source[1])
        if archive is None:
            archive = _ZIPS[source[1]] = zipfile.ZipFile(source[1])
        return archive.read(source[2])
    raise ValueError(f"Unknown image source kind {source[0]!r}")


def signature_of_image(image):
    """(phash, width, height, 32x32x3 uint8 thumbnail) of a decoded PIL image."""
    value = ND.phash(image)
    rgb = ND._flatten(image).convert("RGB")
    thumb = np.asarray(rgb.resize((THUMB, THUMB), Image.Resampling.BOX), dtype=np.uint8)
    return value, rgb.size[0], rgb.size[1], thumb


def signature_task(task):
    """Worker: (source, expected_sha256|None) -> tuple or error string. Never raises."""
    source, expected = task
    try:
        raw = _read_source(source)
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            return f"ChecksumMismatch: expected {expected}, got {digest}"
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            value, width, height, thumb = signature_of_image(image)
        return digest, value, width, height, thumb.tobytes()
    except Exception as exc:  # noqa: BLE001 - untrusted input, recorded per item
        return f"{type(exc).__name__}: {exc}"


def compute_signatures(items, workers=8, chunksize=64, progress_every=50_000):
    """items: iterable of (item_id, source, expected_sha256|None).

    Returns (signatures: dict of arrays sorted by item ID, failures: list of dicts).
    """
    items = sorted(items, key=lambda row: row[0])
    ids = [row[0] for row in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Item IDs in the pool must be unique")
    tasks = [(row[1], row[2]) for row in items]
    results = []
    if workers > 0:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for k, result in enumerate(pool.imap(signature_task, tasks, chunksize=chunksize)):
                results.append(result)
                if progress_every and (k + 1) % progress_every == 0:
                    print(f"signatures: {k + 1:,}/{len(tasks):,}", flush=True)
    else:
        results = [signature_task(task) for task in tasks]
    failures = [
        {"item_id": item_id, "error": result}
        for item_id, result in zip(ids, results, strict=True)
        if isinstance(result, str)
    ]
    ok = [k for k, result in enumerate(results) if not isinstance(result, str)]
    thumbs = np.frombuffer(b"".join(results[k][4] for k in ok), dtype=np.uint8)
    signatures = {
        "item_ids": np.asarray([ids[k] for k in ok], dtype=str),
        "sha256": np.asarray([results[k][0] for k in ok], dtype="U64"),
        "phash": np.asarray([results[k][1] for k in ok], dtype=np.uint64),
        "width": np.asarray([results[k][2] for k in ok], dtype=np.int64),
        "height": np.asarray([results[k][3] for k in ok], dtype=np.int64),
        "thumbs": thumbs.reshape(len(ok), THUMB, THUMB, 3),
    }
    return signatures, failures


def input_sha256(signatures):
    """Content hash of the hashed pool: sorted lines 'item_id<TAB>image SHA-256'."""
    digest = hashlib.sha256()
    for item_id, value in zip(signatures["item_ids"], signatures["sha256"], strict=True):
        digest.update(f"{item_id}\t{value}\n".encode())
    return digest.hexdigest()


def save_signatures(path, signatures, failures):
    with Path(path).open("xb") as stream:
        np.savez(
            stream,
            version=np.asarray(SIGNATURE_VERSION),
            failures=np.asarray(json.dumps(failures)),
            **signatures,
        )


def load_signatures(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["version"]) != SIGNATURE_VERSION:
            raise ValueError(f"Signature cache version {data['version']} != {SIGNATURE_VERSION}")
        keys = ("item_ids", "sha256", "phash", "width", "height", "thumbs")
        signatures = {key: data[key] for key in keys}
        failures = json.loads(str(data["failures"]))
    return signatures, failures


# ------------------------------------------------------------------ candidates


def candidate_pairs(phashes, threshold, max_pairs=200_000_000):
    """All item-index pairs (i < j) with pHash Hamming <= threshold (exact).

    Identical hashes are collapsed and near pairs of distinct hashes are found with
    multi-index hashing; both are expanded to item pairs. Returns (pairs (n, 2) int64
    sorted, hamming (n,) int64, stats).
    """
    phashes = np.asarray(phashes, dtype=np.uint64)
    unique, inverse = np.unique(phashes, return_inverse=True)
    members = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    starts = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    mih = {}
    near = ND.near_pairs(unique, threshold, stats=mih)
    identical_pairs = int(np.sum(counts * (counts - 1) // 2))
    near_item_pairs = int(np.sum(counts[near[:, 0]] * counts[near[:, 1]])) if len(near) else 0
    stats = {
        "unique_hashes": len(unique),
        "largest_identical_hash_group": int(counts.max()) if len(counts) else 0,
        "identical_hash_item_pairs": identical_pairs,
        "near_unique_hash_pairs": len(near),
        "near_hash_item_pairs": near_item_pairs,
        "multi_index_bands": ND.band_layout(threshold),
        "multi_index_stats": mih,
    }
    total = identical_pairs + near_item_pairs
    if total > max_pairs:
        raise RuntimeError(f"{total:,} candidate pairs exceed max_pairs={max_pairs:,}: {stats}")
    left, right, dist = [], [], []
    for u in np.flatnonzero(counts > 1):
        group = members[starts[u] : starts[u] + counts[u]]
        a, b = np.triu_indices(len(group), k=1)
        left.append(group[a])
        right.append(group[b])
        dist.append(np.zeros(len(a), dtype=np.int64))
    for u, v in near:
        gu = members[starts[u] : starts[u] + counts[u]]
        gv = members[starts[v] : starts[v] + counts[v]]
        a, b = np.meshgrid(gu, gv, indexing="ij")
        left.append(a.ravel())
        right.append(b.ravel())
        dist.append(np.full(a.size, ND.hamming(unique[u], unique[v]), dtype=np.int64))
    if not left:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64), stats
    i, j = np.concatenate(left), np.concatenate(right)
    pairs = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1).astype(np.int64)
    hamming = np.concatenate(dist)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order], hamming[order], stats


# ------------------------------------------------------------------ pixel checks


def srgb_to_lab(rgb):
    """sRGB (0-255, ... x 3) -> CIELAB (D65), float64."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = c @ matrix.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _foreground_ab(thumbs, level, chunk=8192):
    """Mean CIELAB (a*, b*) over each image's foreground; white (0, 0) if none."""
    out = np.zeros((len(thumbs), 2))
    for start in range(0, len(thumbs), chunk):
        block = thumbs[start : start + chunk]
        lab = srgb_to_lab(block)
        mask = block.min(axis=-1) < level
        counts = mask.sum(axis=(1, 2))
        sums = (lab[..., 1:] * mask[..., None]).sum(axis=(1, 2))
        out[start : start + chunk] = np.where(
            counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], 0.0
        )
    return out


def _pooled(thumbs):
    n, size = len(thumbs), THUMB // POOL
    return thumbs.astype(np.float32).reshape(n, size, POOL, size, POOL, 3).mean(axis=(2, 4))


def pixel_metrics(pairs, signatures, level=DEFAULT_FOREGROUND_LEVEL, chunk=8192):
    """Per pair: colour (Delta a*b* of foreground means), structure (95th percentile of
    per-cell CIELAB Delta E over the union foreground of the 16x16 thumbnails), and two
    descriptive extras (16x16 foreground RGB MAD, whole 32x32 RGB MAD) plus
    |log aspect-ratio ratio|. Only items that occur in pairs are processed."""
    n = len(pairs)
    out = {
        "colour_dab": np.zeros(n),
        "structure_de95": np.zeros(n),
        "fg_mad16": np.zeros(n),
        "thumb_mad32": np.zeros(n),
        "abs_log_aspect": np.zeros(n),
    }
    if n == 0:
        return out
    used, local = np.unique(pairs, return_inverse=True)
    local = local.reshape(pairs.shape)
    thumbs = signatures["thumbs"][used]
    ab = _foreground_ab(thumbs, level)
    pooled = _pooled(thumbs)
    lab16 = srgb_to_lab(pooled)
    aspect = np.log(signatures["width"][used] / signatures["height"][used])
    a, b = local[:, 0], local[:, 1]
    out["colour_dab"] = np.hypot(*(ab[a] - ab[b]).T)
    out["abs_log_aspect"] = np.abs(aspect[a] - aspect[b])
    for start in range(0, n, chunk):
        sa, sb = a[start : start + chunk], b[start : start + chunk]
        ta, tb = thumbs[sa].astype(np.int16), thumbs[sb].astype(np.int16)
        out["thumb_mad32"][start : start + chunk] = (
            np.abs(ta - tb).reshape(len(sa), -1).mean(axis=1)
        )
        pa, pb = pooled[sa], pooled[sb]
        mask = (pa.min(axis=-1) < level) | (pb.min(axis=-1) < level)
        diff = np.abs(pa - pb).mean(axis=-1)
        counts = mask.sum(axis=(1, 2))
        out["fg_mad16"][start : start + chunk] = np.where(
            counts > 0, (diff * mask).sum(axis=(1, 2)) / np.maximum(counts, 1), 0.0
        )
        delta_e = np.linalg.norm(lab16[sa] - lab16[sb], axis=-1)
        delta_e = np.where(mask, delta_e, np.nan).reshape(len(sa), -1)
        empty = counts == 0
        delta_e[empty] = 0.0  # two blank (all-white) thumbnails agree
        out["structure_de95"][start : start + chunk] = np.nanpercentile(delta_e, 95, axis=1)
    return out


def confirm(metrics, colour, structure, aspect_tolerance=None):
    ok = (metrics["colour_dab"] <= colour) & (metrics["structure_de95"] <= structure)
    if aspect_tolerance is not None:
        ok &= metrics["abs_log_aspect"] <= np.log1p(aspect_tolerance) + 1e-12
    return ok


def union_groups(item_ids, pairs):
    """{item_id: group_id} for items in confirmed groups of size >= 2 (group = min ID)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members = {}
    for index in sorted(parent):
        members.setdefault(find(index), []).append(str(item_ids[index]))
    mapping = {}
    for group in members.values():
        group_id = min(group)
        for item in group:
            mapping[item] = group_id
    return dict(sorted(mapping.items()))


def equivalence(
    signatures,
    hamming=DEFAULT_HAMMING,
    colour=DEFAULT_COLOUR,
    structure=DEFAULT_STRUCTURE,
    aspect_tolerance=DEFAULT_ASPECT_TOLERANCE,
    foreground_level=DEFAULT_FOREGROUND_LEVEL,
    max_pairs=200_000_000,
):
    pairs, distance, stats = candidate_pairs(signatures["phash"], hamming, max_pairs)
    metrics = pixel_metrics(pairs, signatures, foreground_level)
    ok = confirm(metrics, colour, structure, aspect_tolerance)
    groups = union_groups(signatures["item_ids"], pairs[ok])
    return {
        "pairs": pairs,
        "hamming": distance,
        "metrics": metrics,
        "confirmed": ok,
        "groups": groups,
        "stats": stats,
    }


def group_size_distribution(groups):
    sizes = Counter(Counter(groups.values()).values())
    return {str(size): count for size, count in sorted(sizes.items())}


# ------------------------------------------------------------------ pools


def _zip_prefix(archive, suffix):
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one zip entry ending in {suffix!r}, found {len(matches)}")
    return matches[0][: -len(suffix)]


def polyvore_pool(raw_root=None, polyvore_zip=None):
    """All items of the historical disjoint outfits (the polyvore_d_clean pool)."""
    if (raw_root is None) == (polyvore_zip is None):
        raise ValueError("Give exactly one of raw_root / polyvore_zip")
    item_ids = set()
    if raw_root is not None:
        for split in ND.SPLITS:
            text = (Path(raw_root) / "disjoint" / f"{split}.json").read_text(encoding="utf-8")
            item_ids |= {str(item["item_id"]) for row in json.loads(text) for item in row["items"]}
        return [
            (item, ("file", str(Path(raw_root) / "images" / f"{item}.jpg")), None)
            for item in sorted(item_ids)
        ]
    archive_path = str(Path(polyvore_zip).resolve())
    with zipfile.ZipFile(archive_path) as archive:
        prefix = _zip_prefix(archive, "/disjoint/train.json")
        for split in ND.SPLITS:
            rows = json.loads(archive.read(f"{prefix}/disjoint/{split}.json"))
            item_ids |= {str(item["item_id"]) for row in rows for item in row["items"]}
    return [
        (item, ("zip", archive_path, f"{prefix}/images/{item}.jpg"), None)
        for item in sorted(item_ids)
    ]


def iqon_pool(catalog, archive):
    import pandas as pd

    outfits = pd.read_parquet(Path(catalog) / "outfits.parquet", columns=["item_ids"])
    images = pd.read_parquet(
        Path(catalog) / "image_audit_resolved.parquet",
        columns=["item_id", "zip_entry", "encoded_sha256"],
    )
    images["item_id"] = images.item_id.astype(str)
    valid = set(images.item_id)
    item_ids = set()
    for values in outfits.item_ids:
        values = [str(item) for item in values]
        if all(item in valid for item in values):  # construct_iqon_clean usable-outfit rule
            item_ids.update(values)
    rows = images[images.item_id.isin(item_ids)].drop_duplicates("item_id").set_index("item_id")
    archive_path = str(Path(archive).resolve())
    return [
        (
            item,
            ("zip", archive_path, str(rows.at[item, "zip_entry"])),
            str(rows.at[item, "encoded_sha256"]),
        )
        for item in sorted(item_ids)
    ]


def pog_pool(items_path, outfits_path, download_manifest, images_root, target_seed=None,
             target_outfits=None):
    from repbench.v2 import pog

    target, with_url = pog.load_target_outfits(
        Path(items_path),
        Path(outfits_path),
        pog.POG_TARGET_SEED if target_seed is None else target_seed,
        pog.POG_TARGET_OUTFITS if target_outfits is None else target_outfits,
    )
    needed = {item for _, items in target for item in items}
    categories = pog.load_item_categories(Path(items_path), needed)
    available, _ = pog.read_download_manifest(Path(download_manifest))
    item_ids = set()
    for _, items in target:  # construct_pog_clean pre-packing retention rules
        if len(items) < 2 or len(set(items)) != len(items):
            continue
        if any(i not in categories or i not in with_url or i not in available for i in items):
            continue
        item_ids.update(items)
    root = Path(images_root)
    return [
        (item, ("file", str(root / available[item]["path"])), available[item]["sha256"])
        for item in sorted(item_ids)
    ]


def tsv_pool(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item, _, image = line.partition("\t")
            rows.append((item, ("file", image), None))
    return rows


# ------------------------------------------------------------------ CLI


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def thresholds_of(args):
    return {
        "phash_hamming_max": args.hamming,
        "colour_delta_ab_max": args.colour_threshold,
        "structure_delta_e_p95_max": args.structure_threshold,
        "foreground_min_channel_below": args.foreground_level,
        "aspect_tolerance": args.aspect_tolerance,
    }


def run(args):
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    timing = {}
    if args.signatures:
        signatures, failures = load_signatures(args.signatures)
        signature_source = {"cache": Path(args.signatures).name}
    else:
        if args.dataset == "polyvore":
            pool = polyvore_pool(args.raw_root, args.polyvore_zip)
        elif args.dataset == "iqon":
            pool = iqon_pool(args.catalog, args.archive)
        elif args.dataset == "pog":
            pool = pog_pool(args.items, args.outfits, args.download_manifest, args.images_root)
        else:
            pool = tsv_pool(args.items_tsv)
        timing["pool_seconds"] = round(time.perf_counter() - started, 2)
        print(f"pool: {len(pool):,} items", flush=True)
        t0 = time.perf_counter()
        signatures, failures = compute_signatures(pool, workers=args.workers)
        timing["signature_seconds"] = round(time.perf_counter() - t0, 2)
        signature_source = {"pool_items": len(pool)}
        if args.save_signatures:
            save_signatures(args.save_signatures, signatures, failures)
    t0 = time.perf_counter()
    result = equivalence(
        signatures,
        args.hamming,
        args.colour_threshold,
        args.structure_threshold,
        args.aspect_tolerance,
        args.foreground_level,
        args.max_pairs,
    )
    timing["match_seconds"] = round(time.perf_counter() - t0, 2)
    ids = signatures["item_ids"]
    groups_path = out / "equivalence_groups.json"
    with groups_path.open("x", encoding="utf-8") as stream:
        json.dump(result["groups"], stream, indent=0, sort_keys=True)
        stream.write("\n")
    metrics = result["metrics"]
    names = ("colour_dab", "structure_de95", "fg_mad16", "thumb_mad32", "abs_log_aspect")
    with (out / "candidate_pairs.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_a", "item_b", "hamming", *names, "confirmed"])
        for k, (a, b) in enumerate(result["pairs"]):
            writer.writerow(
                [ids[a], ids[b], int(result["hamming"][k])]
                + [f"{metrics[name][k]:.4f}" for name in names]
                + [int(result["confirmed"][k])]
            )
    by_distance = {}
    for d in sorted(set(result["hamming"].tolist())):
        mask = result["hamming"] == d
        by_distance[str(d)] = {
            "candidates": int(mask.sum()),
            "confirmed": int((mask & result["confirmed"]).sum()),
        }
    sizes = Counter(result["groups"].values())
    timing["total_seconds"] = round(time.perf_counter() - started, 2)
    report = {
        "kind": "v2_image_equivalence_groups",
        "dataset": args.dataset,
        "rule": "pHash Hamming <= h AND foreground-mean Delta a*b* <= c AND p95 of 16x16 "
        "per-cell Delta E over the union foreground <= s" + ("" if args.aspect_tolerance is None else " AND aspect within tolerance"),
        "thresholds": thresholds_of(args),
        "signature_version": SIGNATURE_VERSION,
        "phash": "64-bit DCT pHash (scripts/v2/near_duplicate_audit.py), alpha on white",
        "thumbnail": f"{THUMB}x{THUMB} RGB, PIL BOX resample of the white-composited image",
        "signature_source": signature_source,
        "input_items_hashed": len(ids),
        "input_sha256": input_sha256(signatures),
        "input_sha256_definition": "sha256 over sorted lines 'item_id<TAB>image_bytes_sha256\\n'",
        "unreadable_images": failures,
        "candidate_stats": result["stats"],
        "pairs": {
            "candidates": len(result["pairs"]),
            "confirmed": int(result["confirmed"].sum()),
            "rejected": int((~result["confirmed"]).sum()),
            "by_hamming": by_distance,
        },
        "groups": {
            "items_in_groups": len(result["groups"]),
            "groups": len(sizes),
            "largest_group": max(sizes.values()) if sizes else 0,
            "size_distribution": group_size_distribution(result["groups"]),
        },
        "outputs_sha256": {
            groups_path.name: _file_sha256(groups_path),
            "candidate_pairs.csv": _file_sha256(out / "candidate_pairs.csv"),
        },
        "workers": args.workers,
        "runtime": timing,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    with (out / "equivalence_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["polyvore", "iqon", "pog", "tsv"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, help="polyvore: raw root with disjoint/, images/")
    parser.add_argument("--polyvore-zip", type=Path, help="polyvore: polyvore_outfits.zip")
    parser.add_argument("--catalog", type=Path, help="iqon: catalog dir (parquet files)")
    parser.add_argument("--archive", type=Path, help="iqon: source image archive (zip)")
    parser.add_argument("--items", type=Path, help="pog: item_data.txt")
    parser.add_argument("--outfits", type=Path, help="pog: outfit_data.txt")
    parser.add_argument("--download-manifest", type=Path, help="pog: manifest.jsonl")
    parser.add_argument("--images-root", type=Path, help="pog: mirror root holding images/")
    parser.add_argument("--items-tsv", type=Path, help="tsv: item_id<TAB>image path")
    parser.add_argument("--hamming", type=int, default=DEFAULT_HAMMING)
    parser.add_argument("--colour-threshold", type=float, default=DEFAULT_COLOUR)
    parser.add_argument("--structure-threshold", type=float, default=DEFAULT_STRUCTURE)
    parser.add_argument("--foreground-level", type=int, default=DEFAULT_FOREGROUND_LEVEL)
    parser.add_argument("--aspect-tolerance", type=float, default=DEFAULT_ASPECT_TOLERANCE)
    parser.add_argument("--workers", type=int, default=8, help="0 = serial")
    parser.add_argument("--max-pairs", type=int, default=200_000_000)
    parser.add_argument("--save-signatures", type=Path, help="write a .npz signature cache")
    parser.add_argument("--signatures", type=Path, help="reuse a .npz signature cache")
    args = parser.parse_args(argv)
    if not args.signatures:
        need = {
            "polyvore": [],
            "iqon": ["catalog", "archive"],
            "pog": ["items", "outfits", "download_manifest", "images_root"],
            "tsv": ["items_tsv"],
        }[args.dataset]
        missing = [name for name in need if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            parser.error(f"--dataset {args.dataset} needs {flags}")
        if args.dataset == "polyvore" and (args.raw_root is None) == (args.polyvore_zip is None):
            parser.error("--dataset polyvore needs exactly one of --raw-root / --polyvore-zip")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.save_signatures and args.save_signatures.exists():
        raise FileExistsError(f"Refusing to overwrite {args.save_signatures}")
    report = run(args)
    print(json.dumps({k: report[k] for k in ("thresholds", "pairs", "groups", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
