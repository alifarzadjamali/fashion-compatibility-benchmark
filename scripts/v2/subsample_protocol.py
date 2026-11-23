"""Deterministic subsample of a clean protocol for the G5 dry run (train/valid only).

Keeps a seeded fraction of CP positive/negative *pairs* (the i-th negative is paired
with the i-th positive, as constructed) and of FITB questions, plus every outfit
referenced by a kept key so `PolyvoreDisjoint` can resolve `setid_index` keys.
The test split is never read or written. Group-id files are carried over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def subsample_split(source, target, split, fraction, seed):
    lines = [
        line.split()
        for line in (source / f"compatibility_{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    positive = [line for line in lines if line[0] == "1"]
    negative = [line for line in lines if line[0] == "0"]
    if len(positive) != len(negative):
        raise ValueError("Expected balanced paired CP construction")
    rng = random.Random(f"{seed}:{split}")
    keep = sorted(rng.sample(range(len(positive)), max(2, round(fraction * len(positive)))))
    cp = [positive[i] for i in keep] + [negative[i] for i in keep]
    fitb = json.loads((source / f"fill_in_blank_{split}.json").read_text())
    fitb_keep = sorted(rng.sample(range(len(fitb)), max(2, round(fraction * len(fitb)))))
    questions = [fitb[i] for i in fitb_keep]
    keys = {key for line in cp for key in line[1:]}
    keys |= {key for q in questions for key in q["question"] + q["answers"]}
    sets = {key.split("_", 1)[0] for key in keys}
    outfits = [o for o in json.loads((source / f"{split}.json").read_text()) if str(o["set_id"]) in sets]
    target.mkdir(parents=True, exist_ok=True)
    (target / f"compatibility_{split}.txt").write_text("".join(" ".join(l) + "\n" for l in cp))
    (target / f"fill_in_blank_{split}.json").write_text(json.dumps(questions))
    (target / f"{split}.json").write_text(json.dumps(outfits))
    groups = source / f"group_ids_{split}.json"
    if groups.exists():
        mapping = json.loads(groups.read_text())
        (target / f"group_ids_{split}.json").write_text(
            json.dumps({s: mapping[s] for s in sorted(sets) if s in mapping})
        )
    return {"cp_pairs": len(keep), "fitb": len(questions), "outfits": len(outfits)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    counts = {s: subsample_split(args.source, args.output, s, args.fraction, args.seed)
              for s in ("train", "valid")}
    for extra in ("image_paths.json",):
        if (args.source / extra).exists():
            (args.output / extra).write_bytes((args.source / extra).read_bytes())
    manifest = {
        "purpose": "G5 dry-run subsample; train/valid only; test never read",
        "source": str(args.source),
        "source_manifest_sha256": hashlib.sha256((args.source / "manifest.json").read_bytes()).hexdigest()
        if (args.source / "manifest.json").exists() else None,
        "fraction": args.fraction,
        "seed": args.seed,
        "counts": counts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
