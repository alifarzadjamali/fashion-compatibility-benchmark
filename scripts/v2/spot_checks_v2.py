"""Supplementary spot checks promised before the freeze (post-run, reporting only).

precision : bf16 (as run) vs fp32 embeddings for a few encoders (HANDOFF §5; decisions
            log: GR-Lite SDPA substitution "must receive bf16/fp32 spot checks").
qwen_pixels: Qwen3-VL-Embedding at the uniform 384x384 cap (as run) vs the provider
            default pixel budget (1800 visual tokens), decision F "provider-default spot
            check on a seeded subsample reported in the supplement".

Seeded subsample of TRAIN/VALID items (no test data). Also re-embeds with the as-run
setting and compares to the stored Phase 2 vectors (determinism check). Reports per-item
cosine, Spearman of the pairwise-similarity structure, and top-10 neighbour overlap.
Changes nothing in the frozen pipeline; outputs one JSON per check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def similarity_agreement(a, b, k=10):
    from scipy.stats import spearmanr

    sa, sb = a @ a.T, b @ b.T
    iu = np.triu_indices(len(a), 1)
    rho = float(spearmanr(sa[iu], sb[iu]).statistic)
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    na, nb = np.argsort(-sa, 1)[:, :k], np.argsort(-sb, 1)[:, :k]
    overlap = float(np.mean([len(set(x) & set(y)) / k for x, y in zip(na, nb)]))
    cos = np.sum(a * b, 1)
    return {"cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()),
            "cosine_p01": float(np.quantile(cos, 0.01)),
            "pairwise_similarity_spearman": rho, "top10_neighbour_overlap": overlap}


def embed(encoder, paths, batch=16):
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            inputs = encoder.to_device(encoder.prepare(paths[i:i + batch]))
            out.append(encoder.forward(inputs).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def qwen_tokens(encoder, paths):
    tokens = []
    for p in paths:
        grid = encoder.prepare([p])["image_grid_thw"]
        tokens.append(int(grid.prod(-1).sum()) // 4)
    return tokens


def main():
    import torch

    from repbench.utils.seeds import seed_everything
    from repbench.v2 import images
    from repbench.v2.encoders import V2Encoder
    from repbench.v2.registry import V2_ENCODERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["precision", "qwen_pixels"], required=True)
    ap.add_argument("--dataset", action="append", required=True,
                    help="name=images_root=protocol_dir=store_dir_root (repeatable)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_v2 import image_resolver, referenced_items

    for model in args.models.split(","):
        spec = V2_ENCODERS[model]
        for ds in args.dataset:
            name, images_root, protocol, stores = ds.split("=")
            target = args.out / f"{args.check}_{name}_{model}.json"
            if target.exists():
                print("skip (exists)", target)
                continue
            store = Path(stores) / name / model / "trainvalid"
            if not store.exists():  # Qwen stores hold all splits in one file
                store = Path(stores) / name / model / "all"
            ids = json.loads((store / "item_ids.json").read_text())
            stored = np.load(store / "items.npy", mmap_mode="r")
            allowed = set(referenced_items(Path(protocol), ["train", "valid"], name))
            eligible = np.array([i for i, item in enumerate(ids) if item in allowed])
            rng = np.random.default_rng(args.seed)
            pick = np.sort(rng.choice(eligible, size=min(args.n, len(eligible)), replace=False))
            resolve = image_resolver(Path(images_root), Path(protocol))
            paths = [resolve(ids[i]) for i in pick]
            ref = np.asarray(stored[pick], dtype=np.float64)
            record = {"check": args.check, "dataset": name, "model": model,
                      "revision": spec.revision, "n_items": len(pick), "seed": args.seed,
                      "sample": "seeded subsample of TRAIN/VALID items (no test data)"}
            t0 = time.time()
            seed_everything(args.seed)
            if args.check == "precision":
                enc = V2Encoder(spec, dtype=torch.bfloat16)
                bf16 = embed(enc, paths)
                del enc
                torch.cuda.empty_cache()
                enc = V2Encoder(spec, dtype=torch.float32)
                fp32 = embed(enc, paths)
                del enc
                record["bf16_vs_fp32"] = similarity_agreement(bf16, fp32)
                record["bf16_rerun_vs_stored"] = similarity_agreement(bf16, ref)
            else:
                enc = V2Encoder(spec, dtype=torch.bfloat16)
                capped = embed(enc, paths)
                cap_tokens = qwen_tokens(enc, paths)
                original = images.QWEN_MAX_PIXELS
                images.QWEN_MAX_PIXELS = 1800 * 32 * 32  # provider default upper bound
                try:
                    default = embed(enc, paths, batch=4)
                    def_tokens = qwen_tokens(enc, paths)
                finally:
                    images.QWEN_MAX_PIXELS = original
                del enc
                record["capped_vs_provider_default"] = similarity_agreement(capped, default)
                record["capped_rerun_vs_stored"] = similarity_agreement(capped, ref)
                record["visual_tokens"] = {
                    "capped_mean": float(np.mean(cap_tokens)), "default_mean": float(np.mean(def_tokens)),
                    "share_items_changed_by_cap": float(np.mean(np.array(cap_tokens) != np.array(def_tokens)))}
            torch.cuda.empty_cache()
            record["seconds"] = round(time.time() - t0, 1)
            target.write_text(json.dumps(record, indent=2))
            print(json.dumps(record))


if __name__ == "__main__":
    main()
