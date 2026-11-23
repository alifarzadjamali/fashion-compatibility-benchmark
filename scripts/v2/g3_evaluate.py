"""Narrow G3 exception: unchanged v1 PCA-256 ResNet50/CLIP reproduction only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.registry import PRIMARY_ENCODERS
from repbench.utils.seeds import seed_everything
from repbench.v2.state import require_budget, require_gate, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--dataset", choices=["polyvore_d_clean", "iqon3000_clean"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reconstruction-check", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet50", "clip_vitl14_336"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_gate(args.g2, "G2")
    require_budget(1800)
    check = json.loads(args.reconstruction_check.read_text())
    if check["status"] != "PASS" or check["dataset"] != args.dataset:
        raise ValueError("Revised G3 reconstruction has not passed")
    protocol = args.reconstruction_check.resolve().parent / "rebuild_a"
    for name, digest in check["hashes"][0].items():
        if sha256(protocol / name) != digest:
            raise ValueError("Rebuilt protocol changed since reconstruction check")
    embedding_root = args.embedding_root.resolve()
    spec = PRIMARY_ENCODERS[args.model]
    metadata_hashes = {}
    for split in ("train", "valid", "test"):
        cache = embedding_root / args.dataset / args.model
        metadata_path = cache / f"{split}_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["checkpoint_revision"] != spec.revision:
            raise ValueError("Embedding revision differs from frozen v1")
        if args.model == "clip_vitl14_336" and metadata.get("inference_dtype") != "torch.float16":
            raise ValueError("G3 requires original v1 fp16 CLIP, not v2 bf16")
        if (
            args.model == "resnet50"
            and metadata.get("inference_dtype", "torch.float32") != "torch.float32"
        ):
            raise ValueError("G3 requires original v1 fp32 ResNet50")
        for file_name, field in [
            (f"{split}.npy", "embedding_sha256"),
            (f"{split}_item_ids.json", "item_ids_sha256"),
        ]:
            if sha256(cache / file_name) != metadata[field]:
                raise ValueError("Embedding cache checksum mismatch")
        metadata_hashes[split] = sha256(metadata_path)
    reference_path = Path("configs/v1_published_reference.json")
    reference = json.loads(reference_path.read_text())
    expected = reference["models"][args.model]
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The v1 writer uses relative artifacts/. Isolate its entire working directory
    # so no completed v1 output can be touched, even if the writer has fixed paths.
    scripts = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts))
    from run_benchmark import run_representation

    seed_everything(20260912)
    dataset = PolyvoreDisjoint(raw_root, protocol, args.dataset)
    previous = Path.cwd()
    start = time.perf_counter()
    try:
        os.chdir(output)
        actual = run_representation(
            dataset,
            args.model,
            20260912,
            "pca",
            "v2_g3_legacy",
            pca_components=256,
            embedding_root=embedding_root,
            embedding_dataset_key=args.dataset,
        )
    finally:
        os.chdir(previous)
    # Metric names are taken from the existing v1 runner, not reimplemented here.
    corpus = "polyvore" if args.dataset == "polyvore_d_clean" else "iqon"
    deltas = {
        "cp_auc": float(actual["cp_auc"]) - expected[f"{corpus}_cp_auc"],
        "fitb": float(actual["fitb_acc"]) - expected[f"{corpus}_fitb"],
    }
    passed = abs(deltas["cp_auc"]) <= 0.002 and abs(deltas["fitb"]) <= 0.005
    write_json(
        output / "reproduction_check.json",
        {
            "gate_part": "G3 revised published-metric reproduction",
            "dataset": args.dataset,
            "model": args.model,
            "status": "PASS" if passed else "FAIL",
            "deltas_from_published": deltas,
            "metrics": actual,
            "elapsed_seconds": time.perf_counter() - start,
            "embedding_metadata_hashes": metadata_hashes,
            "original_split_identity": "UNVERIFIABLE",
            "reference_sha256": sha256(reference_path),
        },
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
