"""Cross-split near-duplicate audit for one protocol directory (HANDOFF_V2 §6.8).

Methods:
- 64-bit DCT perceptual hash (pHash), implemented with PIL + numpy following the
  imagehash.phash algorithm (grayscale, Lanczos resize to 32x32, 2-D DCT-II, top-left
  8x8 block, bit = coefficient > block median). Near duplicate: Hamming <= threshold
  (default 4). Scales by multi-index hashing: the 64 bits are split into
  threshold + 1 bands, so by pigeonhole any pair within the threshold agrees exactly on
  at least one band; only band-bucket collisions are verified. Identical hashes are
  collapsed first, so large clusters of identical images (blank backgrounds) cost
  O(unique hashes), not O(images^2).
- Optional embedding cosine > threshold (default 0.95) from an embedding store,
  computed blockwise (exact, all cross-split pairs; CPU or CUDA matmul).

Reports cross-split pair counts (train-test, valid-test, train-valid), the affected
test items and test outfits (for the exclusion sensitivity analysis), item ids shared
across splits, unreadable images, and runtime. Outputs go to a new directory:
  audit.json, near_duplicate_pairs.csv (listing capped; counts exact),
  affected_test_outfits.json, phashes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "valid", "test")
SPLIT_PAIRS = (("train", "test"), ("valid", "test"), ("train", "valid"))
HASH_SIZE, HIGHFREQ = 8, 4
_N = HASH_SIZE * HIGHFREQ
_DCT = np.cos(np.pi * np.outer(np.arange(_N), 2 * np.arange(_N) + 1) / (2 * _N))
_POPCOUNT8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


# ------------------------------------------------------------------------- hashing


def _flatten(image):
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return image


def phash(image):
    """64-bit DCT perceptual hash as a Python int (bit 63 = coefficient [0, 0])."""
    gray = _flatten(image).convert("L").resize((_N, _N), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT @ pixels @ _DCT.T
    low = dct[:HASH_SIZE, :HASH_SIZE]
    bits = (low > np.median(low)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_file(path):
    try:
        with Image.open(path) as image:
            image.load()
            return phash(image)
    except Exception as exc:  # noqa: BLE001 - recorded per image, audit continues
        return f"{type(exc).__name__}: {exc}"


def popcount(values):
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values).astype(np.int64)
    return _POPCOUNT8[values.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.int64)


def hamming(a, b):
    return int(popcount(np.asarray([int(a) ^ int(b)], dtype=np.uint64))[0])


def band_layout(threshold):
    bands = threshold + 1
    if not 1 <= bands <= 64:
        raise ValueError("Hamming threshold must be in [0, 63]")
    widths = [64 // bands + (1 if i < 64 % bands else 0) for i in range(bands)]
    layout, shift = [], 64
    for width in widths:
        shift -= width
        layout.append((shift, width))
    return layout


def near_pairs(unique_hashes, threshold, block=1024, stats=None):
    """All index pairs i < j of distinct hashes within the Hamming threshold.

    Multi-index hashing; exact (equivalent to all-pairs), verified per bucket. Cost is
    sum over buckets of size^2 / 2; `stats` (optional dict) receives the largest bucket
    and the number of verified comparisons so skewed hash distributions are visible.
    """
    if stats is not None:
        stats.update(largest_bucket=0, verified_comparisons=0)
    hashes = np.asarray(unique_hashes, dtype=np.uint64)
    n = len(hashes)
    found = []
    for shift, width in band_layout(threshold):
        keys = (hashes >> np.uint64(shift)) & np.uint64((1 << width) - 1)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                continue
            members = order[start:end]
            if stats is not None:
                stats["largest_bucket"] = max(stats["largest_bucket"], len(members))
                stats["verified_comparisons"] += len(members) * (len(members) - 1) // 2
            for r0 in range(0, len(members), block):
                rows = members[r0 : r0 + block]
                for c0 in range(r0, len(members), block):
                    cols = members[c0 : c0 + block]
                    xor = hashes[rows][:, None] ^ hashes[cols][None, :]
                    dist = popcount(xor.ravel()).reshape(xor.shape)
                    r, c = np.nonzero(dist <= threshold)
                    i, j = rows[r], cols[c]
                    keep = i != j
                    i, j = i[keep], j[keep]
                    low, high = np.minimum(i, j), np.maximum(i, j)
                    found.append(low.astype(np.int64) * n + high)
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    codes = np.unique(np.concatenate(found))
    return np.stack([codes // n, codes % n], axis=1)


# ------------------------------------------------------------------------- protocol


def split_items(protocol, dataset_name):
    """Items referenced per split by outfits, CP examples and FITB questions."""
    from repbench.data.polyvore import PolyvoreDisjoint

    data = PolyvoreDisjoint(protocol.parent, protocol, dataset_name)
    items, outfits = {}, {}
    for split in SPLITS:
        found = set()
        outfits[split] = {o.set_id: o.item_ids for o in data.outfits(split)}
        for ids in outfits[split].values():
            found |= set(ids)
        for loader in (data.compatibility, data.fitb):
            try:
                rows = loader(split)
            except FileNotFoundError:
                continue
            for row in rows:
                if hasattr(row, "item_ids"):
                    found |= set(row.item_ids)
                else:
                    found |= set(row.question_item_ids) | set(row.candidate_item_ids)
        items[split] = sorted(found)
    return items, outfits


# ------------------------------------------------------------------------- audit


def phash_audit(entries, hashes, threshold):
    """entries: list of (item_id, split); hashes: uint64 per entry.

    Returns counts per split pair, flagged entries per (target split, source split),
    and near-duplicate (entry_a, entry_b, distance) generator inputs.
    """
    split_index = {s: k for k, s in enumerate(SPLITS)}
    unique, inverse = np.unique(np.asarray(hashes, dtype=np.uint64), return_inverse=True)
    counts = np.zeros((len(unique), len(SPLITS)), dtype=np.int64)
    entry_split = np.asarray([split_index[s] for _, s in entries])
    np.add.at(counts, (inverse, entry_split), 1)
    mih_stats = {}
    pairs = near_pairs(unique, threshold, stats=mih_stats)
    # reach[u, s] = some image of split s has hash within threshold of unique hash u.
    reach = counts > 0
    if len(pairs):
        np.logical_or.at(reach, pairs[:, 0], counts[pairs[:, 1]] > 0)
        np.logical_or.at(reach, pairs[:, 1], counts[pairs[:, 0]] > 0)
    pair_counts = {}
    for a, b in SPLIT_PAIRS:
        ia, ib = split_index[a], split_index[b]
        total = int(np.sum(counts[:, ia] * counts[:, ib]))  # identical hashes
        if len(pairs):
            ci, cj = counts[pairs[:, 0]], counts[pairs[:, 1]]
            total += int(np.sum(ci[:, ia] * cj[:, ib] + ci[:, ib] * cj[:, ia]))
        pair_counts[f"{a}-{b}"] = total
    flagged = {}
    for target, source in (("test", "train"), ("test", "valid"), ("valid", "train")):
        mask = (entry_split == split_index[target]) & reach[inverse, split_index[source]]
        flagged[(target, source)] = {entries[k][0] for k in np.flatnonzero(mask)}
    return {
        "unique": unique,
        "inverse": inverse,
        "pairs": pairs,
        "pair_counts": pair_counts,
        "flagged": flagged,
        "identical_hash_groups_cross_split": int(np.sum((counts > 0).sum(axis=1) > 1)),
        "multi_index_stats": mih_stats,
    }


def list_phash_pairs(entries, result, limit):
    """Yield item-level cross-split pairs (capped)."""
    by_unique = {}
    for k, u in enumerate(result["inverse"]):
        by_unique.setdefault(int(u), []).append(k)
    unique = result["unique"]
    emitted = 0
    candidates = [(u, u) for u, ks in by_unique.items() if len({entries[k][1] for k in ks}) > 1]
    candidates += [(int(i), int(j)) for i, j in result["pairs"]]
    for i, j in candidates:
        distance = hamming(unique[i], unique[j])
        for a in by_unique.get(i, []):
            for b in by_unique.get(j, []):
                if (i == j and a >= b) or entries[a][1] == entries[b][1]:
                    continue
                if emitted >= limit:
                    return
                emitted += 1
                yield entries[a], entries[b], distance


def embedding_audit(store, items, threshold, device="cpu", block=4096, limit=100_000):
    import torch

    flagged, counts, listed = {}, {}, []
    matrices = {}
    for split in SPLITS:
        ids = [i for i in items[split] if i in store]
        if ids:
            m = torch.from_numpy(np.stack([store[i] for i in ids]).astype(np.float32))
            matrices[split] = (ids, torch.nn.functional.normalize(m, dim=1).to(device))
        else:
            matrices[split] = ([], None)
    missing = {s: len(items[s]) - len(matrices[s][0]) for s in SPLITS}
    for source, target in SPLIT_PAIRS:
        query_ids, query = matrices[target]
        ref_ids, ref = matrices[source]
        total, hits = 0, set()
        if query is not None and ref is not None:
            for start in range(0, len(query_ids), block):
                sims = query[start : start + block] @ ref.T
                rows, cols = torch.nonzero(sims > threshold, as_tuple=True)
                total += len(rows)
                for r in torch.unique(rows).tolist():
                    hits.add(query_ids[start + r])
                if len(listed) < limit:
                    values = sims[rows, cols].tolist()
                    for r, c, v in zip(rows.tolist(), cols.tolist(), values, strict=True):
                        if len(listed) >= limit:
                            break
                        listed.append(((ref_ids[c], source), (query_ids[start + r], target), v))
        counts[f"{source}-{target}"] = total
        flagged[(target, source)] = hits
    return counts, flagged, listed, missing


def affected_outfits(outfits, flagged_items):
    return sorted(s for s, ids in outfits.items() if flagged_items.intersection(ids))


def run_audit(args):
    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    protocol = args.protocol_root.resolve()
    from repbench.v2.state import sha256, write_json
    from repbench.v2.stores import image_resolver, protocol_hashes

    items, outfits = split_items(protocol, args.dataset_name)
    entries = [(item, split) for split in SPLITS for item in items[split]]
    resolve = image_resolver(args.images_root, protocol)
    timing = {"read_protocol_seconds": time.perf_counter() - started}
    t0 = time.perf_counter()
    paths = [resolve(item) for item, _ in entries]
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(phash_file, paths, chunksize=256))
    else:
        results = [phash_file(path) for path in paths]
    timing["phash_seconds"] = time.perf_counter() - t0
    failures = [
        {"item_id": e[0], "split": e[1], "error": r}
        for e, r in zip(entries, results, strict=True)
        if isinstance(r, str)
    ]
    ok = [k for k, r in enumerate(results) if not isinstance(r, str)]
    hashed_entries = [entries[k] for k in ok]
    hashes = np.asarray([results[k] for k in ok], dtype=np.uint64)
    with (out / "phashes.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "split", "phash_hex"])
        for (item, split), value in zip(hashed_entries, hashes, strict=True):
            writer.writerow([item, split, f"{int(value):016x}"])
    t0 = time.perf_counter()
    ph = phash_audit(hashed_entries, hashes, args.hamming)
    timing["phash_match_seconds"] = time.perf_counter() - t0
    with (out / "near_duplicate_pairs.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "item_a", "split_a", "item_b", "split_b", "value"])
        listed = 0
        for a, b, distance in list_phash_pairs(hashed_entries, ph, args.max_listed_pairs):
            writer.writerow(["phash_hamming", a[0], a[1], b[0], b[1], distance])
            listed += 1
        phash_listing_truncated = listed >= args.max_listed_pairs
        emb = None
        if args.embeddings:
            from repbench.v2.stores import load_embedding_store

            t0 = time.perf_counter()
            store, provenance = load_embedding_store(args.embeddings)
            counts, flagged, pairs, missing = embedding_audit(
                store, items, args.cosine, device=args.device, limit=args.max_listed_pairs
            )
            timing["embedding_seconds"] = time.perf_counter() - t0
            for a, b, value in pairs:
                writer.writerow(["embedding_cosine", a[0], a[1], b[0], b[1], f"{value:.6f}"])
            emb = {
                "store": provenance,
                "threshold": f"cosine > {args.cosine}",
                "pair_counts": counts,
                "items_without_embedding": missing,
                "listing_truncated": len(pairs) >= args.max_listed_pairs,
                "flagged": flagged,
            }
    shared = {f"{a}-{b}": len(set(items[a]) & set(items[b])) for a, b in SPLIT_PAIRS}
    affected = {}
    for target, source in (("test", "train"), ("test", "valid")):
        affected[f"phash_{source}"] = sorted(ph["flagged"][(target, source)])
        if emb:
            affected[f"embedding_{source}"] = sorted(emb["flagged"][(target, source)])
    union_items = set().union(*(set(v) for v in affected.values())) if affected else set()
    test_outfits = outfits["test"]
    outfit_lists = {k: affected_outfits(test_outfits, set(v)) for k, v in affected.items()}
    outfit_lists["phash_any"] = affected_outfits(
        test_outfits, set(affected["phash_train"]) | set(affected["phash_valid"])
    )
    if emb:
        outfit_lists["embedding_any"] = affected_outfits(
            test_outfits, set(affected["embedding_train"]) | set(affected["embedding_valid"])
        )
    outfit_lists["any_method"] = affected_outfits(test_outfits, union_items)
    write_json(
        out / "affected_test_outfits.json",
        {"test_items": {k: v for k, v in affected.items()}, "test_outfits": outfit_lists},
    )
    timing["total_seconds"] = time.perf_counter() - started
    summary = {
        "kind": "cross_split_near_duplicate_audit",
        "dataset_name": args.dataset_name,
        "protocol_root": str(protocol),
        "protocol_sha256": protocol_hashes(protocol),
        "images_root": str(args.images_root),
        "items_per_split": {s: len(items[s]) for s in SPLITS},
        "outfits_per_split": {s: len(outfits[s]) for s in SPLITS},
        "shared_item_ids": shared,
        "unreadable_images": failures,
        "phash": {
            "algorithm": "64-bit DCT pHash (imagehash.phash algorithm), PIL Lanczos 32x32",
            "threshold": f"hamming <= {args.hamming}",
            "multi_index_bands": band_layout(args.hamming),
            "multi_index_stats": ph["multi_index_stats"],
            "unique_hashes": len(ph["unique"]),
            "near_unique_hash_pairs": len(ph["pairs"]),
            "identical_hash_groups_cross_split": ph["identical_hash_groups_cross_split"],
            "pair_counts": ph["pair_counts"],
            "flagged_valid_items_from_train": len(ph["flagged"][("valid", "train")]),
            "listing_truncated": phash_listing_truncated,
        },
        "embedding": None if emb is None else {k: v for k, v in emb.items() if k != "flagged"},
        "affected_test_items": {k: len(v) for k, v in affected.items()},
        "affected_test_outfits": {k: len(v) for k, v in outfit_lists.items()},
        "affected_test_outfit_fraction": {
            k: len(v) / max(1, len(test_outfits)) for k, v in outfit_lists.items()
        },
        "pairs_csv_sha256": sha256(out / "near_duplicate_pairs.csv"),
        "workers": args.workers,
        "runtime": timing,
        "finished_utc": datetime.now(UTC).isoformat(),
        "notes": "Pair counts are image-pair counts (exact); listing capped at "
        f"{args.max_listed_pairs} per method. Affected test outfits contain >= 1 test item "
        "with a near duplicate in train or valid.",
    }
    write_json(out / "audit.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hamming", type=int, default=4)
    parser.add_argument("--embeddings", type=Path, help="Optional embedding store")
    parser.add_argument("--cosine", type=float, default=0.95)
    parser.add_argument("--device", default="cpu", help="Device for the cosine matmul")
    parser.add_argument("--workers", type=int, default=8, help="0 = serial")
    parser.add_argument("--max-listed-pairs", type=int, default=200_000)
    args = parser.parse_args(argv)
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "phash": summary["phash"]["pair_counts"],
                "embedding": (summary["embedding"] or {}).get("pair_counts"),
                "affected_test_outfits": summary["affected_test_outfits"],
                "seconds": summary["runtime"]["total_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
