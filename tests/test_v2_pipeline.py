"""End-to-end smoke test: synthetic protocol + embeddings -> pipeline -> tables -> checker."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_protocol(folder: Path, rng: random.Random, n_outfits: int, prefix: str):
    """Outfits of 3 items; compatible items share a latent style. Polyvore key format."""
    outfits, cp, fitb, groups = [], [], [], {}
    for k in range(n_outfits):
        set_id = f"{prefix}{k}"
        style = k % 4
        items = [{"item_id": f"{set_id}x{j}s{style}", "index": j + 1} for j in range(3)]
        outfits.append({"set_id": set_id, "items": items})
        groups[set_id] = f"user{k % 7}"
    pos, neg = [], []
    for k, outfit in enumerate(outfits):
        sid = outfit["set_id"]
        pos.append(f"1 {sid}_1 {sid}_2 {sid}_3")
        other = outfits[(k + 1) % len(outfits)]["set_id"]
        neg.append(f"0 {sid}_1 {other}_2 {sid}_3")
        wrong = [outfits[(k + d) % len(outfits)]["set_id"] for d in (1, 2, 3)]
        fitb.append({"question": [f"{sid}_1", f"{sid}_2"], "blank_position": 3,
                     "answers": [f"{sid}_3"] + [f"{w}_3" for w in wrong]})
    folder.mkdir(parents=True, exist_ok=True)
    split = folder.name
    return outfits, pos + neg, fitb, groups, split


def build(tmp: Path):
    rng = random.Random(0)
    protocol = tmp / "synthetic_clean"
    for split, n in (("train", 160), ("valid", 60)):
        outfits, cp, fitb, groups, _ = write_protocol(protocol / split, rng, n, split[:2])
        (protocol / f"{split}.json").write_text(json.dumps(outfits))
        (protocol / f"compatibility_{split}.txt").write_text("\n".join(cp) + "\n")
        (protocol / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb))
        (protocol / f"group_ids_{split}.json").write_text(json.dumps(groups))
    items = sorted({i["item_id"] for s in ("train", "valid")
                    for o in json.loads((protocol / f"{s}.json").read_text()) for i in o["items"]})
    centers = np.random.default_rng(1).normal(size=(4, 160))
    for model, noise in (("resnet50", 3.0), ("clip_vitl14_336", 2.5), ("siglip2_so400m_384", 2.0)):
        gen = np.random.default_rng(len(model))
        values = np.stack([centers[int(i[-1])] + noise * gen.normal(size=160) for i in items])
        folder = tmp / "emb" / "synthetic_clean" / model
        folder.mkdir(parents=True)
        np.save(folder / "items.npy", values.astype(np.float32))
        (folder / "item_ids.json").write_text(json.dumps(items))
        digest = hashlib.sha256((folder / "items.npy").read_bytes()).hexdigest()
        (folder / "metadata.json").write_text(json.dumps({"items_sha256": digest}))
    return protocol


@pytest.mark.skipif(
    sys.platform == "win32", reason="Linux-only: requires the fork multiprocessing context"
)
def test_pipeline_end_to_end(tmp_path):
    protocol = build(tmp_path)
    results = tmp_path / "results"
    run = [sys.executable, str(ROOT / "scripts/v2/pipeline.py"),
           "--datasets", f"synthetic_clean={protocol}=two_way",
           "--models", "resnet50,clip_vitl14_336,siglip2_so400m_384",
           "--embeddings", str(tmp_path / "emb"), "--results", str(results),
           "--settings", "native,pca128", "--learners", "logistic,xgboost",
           "--replicates", "200", "--workers", "2"]
    subprocess.run(run, check=True, cwd=ROOT)
    metrics = (results / "metrics.csv").read_text()
    assert "native" in metrics and "abs_product_sum" in metrics
    for name in ("failures.csv", "pairwise.csv", "rank_intervals.csv", "equivalence.csv", "calibration.csv",
                 "friedman.json", "selection.json", "manifest.json"):
        assert (results / name).stat().st_size > 0
    # Signal ordering: least-noisy encoder should beat the noisiest on native/logistic.
    import pandas as pd

    frame = pd.read_csv(results / "metrics.csv")
    primary = frame[(frame.setting == "native") & (frame.learner == "logistic")
                    & (frame.metric == "cp_auc")].set_index("model").value
    assert primary["siglip2_so400m_384"] > primary["resnet50"]
    generated = tmp_path / "generated"
    subprocess.run([sys.executable, str(ROOT / "scripts/v2/make_tables_v2.py"),
                    "--results", str(results), "--out", str(generated)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/v2/check_paper_numbers.py"),
                    "--generated", str(generated)], check=True, cwd=ROOT)
    # Test-split evaluation is refused before the freeze.
    refused = subprocess.run(run[:-2] + ["--eval-split", "test"], cwd=ROOT, capture_output=True)
    assert refused.returncode != 0 and b"frozen" in refused.stderr
