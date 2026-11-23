"""v2 embedding extraction (uniform bf16 + SDPA via V2Encoder), resumable.

Embeds every item referenced by the requested splits (outfits, CP lines, FITB
questions/candidates) of a Polyvore-format protocol. Output per (dataset, model):
`items.npy` (float32, L2-normalized as returned by the encoder), `item_ids.json`,
`metadata.json` (hashes, revision, rates). Completed outputs are verified, never
overwritten; an interrupted run resumes from `state.json` if inputs are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def referenced_items(protocol, splits, dataset_name):
    from repbench.data.polyvore import PolyvoreDisjoint

    data = PolyvoreDisjoint(protocol.parent, protocol, dataset_name)
    items = set()
    for split in splits:
        items |= data.item_ids(split)
        for example in data.compatibility(split):
            items |= set(example.item_ids)
        for question in data.fitb(split):
            items |= set(question.question_item_ids) | set(question.candidate_item_ids)
    return sorted(items)


def image_resolver(images_root, protocol):
    mapping_path = protocol / "image_paths.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        return lambda item: images_root / mapping[item]
    return lambda item: images_root / f"{item}.jpg"


def main():
    import torch

    from repbench.utils.seeds import seed_everything
    from repbench.v2.encoders import V2Encoder
    from repbench.v2.registry import V2_ENCODERS
    from repbench.v2.state import require_budget, require_gate, sha256, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--reuse-from", type=Path, nargs="*", default=[],
                        help="verified earlier stores (same model+revision) whose vectors are copied; "
                             "only valid when decoding is unchanged for these images (JPEG datasets)")
    args = parser.parse_args()
    splits = args.splits.split(",")
    if "test" in splits:
        frozen = Path("configs/v2_protocol_hash.json")
        if not frozen.exists():
            raise PermissionError("Test-split extraction is Phase 2 only (after G7 freeze)")
        record = json.loads(frozen.read_text())
        if sha256(record["file"]) != record["sha256"]:
            raise PermissionError("Frozen protocol file changed since freeze; refusing")
    require_gate(args.g2, "G2")
    require_budget(600)
    spec = V2_ENCODERS[args.model]
    ids = referenced_items(args.protocol, splits, args.dataset)
    resolve = image_resolver(args.images_root, args.protocol)
    identity = {
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "model": args.model,
        "revision": spec.revision,
        "splits": splits,
        "batch_size": args.batch_size,
        "lock": sha256("uv.lock"),
        "reuse_from": sorted(str(f) for f in args.reuse_from),
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        meta = json.loads((out / "metadata.json").read_text())
        if meta["signature"] != signature or sha256(out / "items.npy") != meta["items_sha256"]:
            raise ValueError("Completed extraction does not match inputs; refusing to overwrite")
        print(f"Reusing verified {args.dataset}/{args.model}")
        return
    seed_everything(20260912)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    encoder = V2Encoder(spec)
    load_seconds = time.perf_counter() - started
    state_path, partial = out / "state.json", out / "items.partial.npy"
    if (out / "items.npy").exists() and not partial.exists():
        raise FileExistsError(f"{out}: items.npy without metadata.json (interrupted finalisation); "
                              "inspect, then remove the incomplete store and re-run")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["signature"] != signature:
            raise ValueError("Partial extraction is from different inputs")
        array = np.lib.format.open_memmap(partial, mode="r+")
    else:
        state = {"signature": signature, "offset": 0, "seconds": 0.0}
        array = None

    reused = {}
    for folder in args.reuse_from:
        meta = json.loads((folder / "metadata.json").read_text())
        if meta["model"] != args.model or meta["revision"] != spec.revision:
            raise ValueError(f"Reuse store {folder} is a different model/revision")
        if sha256(folder / "items.npy") != meta["items_sha256"]:
            raise ValueError(f"Reuse store {folder} failed checksum")
        old_ids = json.loads((folder / "item_ids.json").read_text())
        old_values = np.load(folder / "items.npy", mmap_mode="r")
        for k, item in enumerate(old_ids):
            reused.setdefault(item, (old_values, k))
    reuse_count = sum(i in reused for i in ids)
    if array is None and reuse_count:
        dim = next(iter(reused.values()))[0].shape[1]
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(ids), dim))
    if reuse_count and state["offset"] == 0 and not state.get("reused"):
        for k, item in enumerate(ids):
            if item in reused:
                values, row = reused[item]
                array[k] = values[row]
        array.flush()
        state["reused"] = reuse_count
    todo = [k for k, item in enumerate(ids) if item not in reused]
    print(f"{args.dataset}/{args.model}: reused {reuse_count}, to extract {len(todo)}", flush=True)

    def prepare(batch):
        return encoder.prepare([resolve(ids[k]) for k in batch])

    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        for offset in range(state["offset"], len(todo), args.chunk):
            require_budget(300)
            chunk = todo[offset : offset + args.chunk]
            batches = [chunk[i : i + args.batch_size] for i in range(0, len(chunk), args.batch_size)]
            t0 = time.perf_counter()
            outputs = []
            for inputs in pool.map(prepare, batches):
                outputs.append(encoder.forward(encoder.to_device(inputs)).float().cpu().numpy())
            values = np.concatenate(outputs)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite embeddings")
            if array is None:
                array = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.float32, shape=(len(ids), values.shape[1])
                )
            array[chunk] = values
            array.flush()
            state["offset"] = offset + len(chunk)
            state["seconds"] += time.perf_counter() - t0
            state_path.with_suffix(".tmp").write_text(json.dumps(state))
            state_path.with_suffix(".tmp").replace(state_path)
            print(f"{args.dataset}/{args.model}: {state['offset']}/{len(todo)} new", flush=True)
    dim = array.shape[1]
    del array
    partial.rename(out / "items.npy")
    write_json(out / "item_ids.json", ids)
    write_json(
        out / "metadata.json",
        {
            "signature": signature,
            "identity": identity,
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint": spec.checkpoint,
            "revision": spec.revision,
            "dtype": "bfloat16",
            "attention": encoder.attention_method,
            "items": len(ids),
            "reused_items": reuse_count,
            "reused_from": [str(f) for f in args.reuse_from],
            "embedding_dim": dim,
            "load_seconds": round(load_seconds, 2),
            "extract_seconds": round(state["seconds"], 2),
            "images_per_second": round(len(todo) / max(state["seconds"], 1e-9), 2),
            "allocator_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "items_sha256": sha256(out / "items.npy"),
            "item_ids_sha256": sha256(out / "item_ids.json"),
        },
    )
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
