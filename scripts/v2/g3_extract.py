"""Resumable extraction for the authorized v1-only reproduction exception."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

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
    parser.add_argument("--model", choices=["resnet50", "clip_vitl14_336"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-images", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    require_gate(args.g2, "G2")
    require_budget(900)
    if args.chunk_images < 1 or args.batch_size < 1:
        raise ValueError("Chunk and batch sizes must be positive")
    check = json.loads(args.reconstruction_check.read_text())
    if check["status"] != "PASS" or check["dataset"] != args.dataset:
        raise ValueError("Revised G3 reconstruction has not passed")
    protocol = args.reconstruction_check.resolve().parent / "rebuild_a"
    for name, digest in check["hashes"][0].items():
        if sha256(protocol / name) != digest:
            raise ValueError("Reconstruction files changed")
    dataset = PolyvoreDisjoint(args.raw_root, protocol, args.dataset)
    spec = PRIMARY_ENCODERS[args.model]
    seed_everything(20260912)
    load_start = time.perf_counter()
    if args.model == "resnet50":
        from repbench.encoders.torchvision_encoder import ResNet50Encoder

        encoder = ResNet50Encoder()
    else:
        from repbench.encoders.hf_encoder import HuggingFaceImageEncoder

        encoder = HuggingFaceImageEncoder(
            spec.model_key,
            spec.checkpoint,
            spec.revision,
            cuda_dtype=spec.cuda_dtype,
            use_safetensors=spec.use_safetensors,
        )
    load_seconds = time.perf_counter() - load_start
    destination = args.output_root / args.dataset / args.model
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        ids = sorted(dataset.item_ids(split))
        identity = {
            "ids": ids,
            "spec": vars(spec),
            "batch_size": args.batch_size,
            "protocol": sha256(protocol / "manifest.json"),
            "lock": sha256("uv.lock"),
        }
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        complete = destination / f"{split}_metadata.json"
        values = destination / f"{split}.npy"
        ids_path = destination / f"{split}_item_ids.json"
        state_path = destination / f"{split}_g3_state.json"
        if complete.exists():
            metadata = json.loads(complete.read_text())
            if (
                metadata.get("g3_signature") != signature
                or sha256(values) != metadata["embedding_sha256"]
                or sha256(ids_path) != metadata["item_ids_sha256"]
            ):
                raise ValueError("Completed G3 cache failed verification; do not overwrite")
            print(f"Reusing verified {args.model}/{split}", flush=True)
            continue
        if values.exists() or ids_path.exists():
            raise FileExistsError("Unfinalized outputs require inspection; refusing overwrite")
        partial = destination / f"{split}.partial.npy"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["signature"] != signature:
                raise ValueError("Partial cache is from different inputs/settings")
            array = np.lib.format.open_memmap(partial, mode="r+")
            if array.shape != (len(ids), spec.embedding_dim):
                raise ValueError("Partial cache has wrong shape")
        else:
            if partial.exists():
                raise FileExistsError("Orphan partial cache; inspect before retrying")
            state = {
                "signature": signature,
                "offset": 0,
                "elapsed_seconds": 0.0,
                "peak_vram_mb": 0.0,
            }
            array = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32, shape=(len(ids), spec.embedding_dim)
            )
        for offset in range(state["offset"], len(ids), args.chunk_images):
            require_budget(300)
            if (destination / "STOP").exists():
                raise SystemExit("Graceful stop requested; partial cache retained")
            batch_ids = ids[offset : offset + args.chunk_images]
            output = encoder.encode(
                [dataset.image_path(i) for i in batch_ids], batch_ids, args.batch_size
            )
            if not np.isfinite(output.embeddings).all():
                raise ValueError("Non-finite legacy embeddings")
            array[offset : offset + len(batch_ids)] = output.embeddings
            array.flush()
            state["offset"] = offset + len(batch_ids)
            state["elapsed_seconds"] += output.metadata["elapsed_seconds"]
            state["peak_vram_mb"] = max(
                state["peak_vram_mb"], output.metadata["peak_vram_mb"] or 0.0
            )
            state["last_metadata"] = output.metadata
            scratch = state_path.with_suffix(".tmp")
            scratch.write_text(json.dumps(state, indent=2))
            scratch.replace(state_path)
            print(f"{args.model}/{split}: {state['offset']}/{len(ids)} images", flush=True)
        del array
        partial.rename(values)
        write_json(ids_path, ids)
        metadata = state["last_metadata"]
        metadata.update(
            g3_signature=signature,
            dataset_key=args.dataset,
            protocol_manifest_sha256=identity["protocol"],
            batch_size=args.batch_size,
            elapsed_seconds=state["elapsed_seconds"],
            peak_vram_mb=state["peak_vram_mb"],
            images_per_second=len(ids) / state["elapsed_seconds"],
            load_seconds=load_seconds,
            embedding_sha256=sha256(values),
            item_ids_sha256=sha256(ids_path),
            inference_dtype=str(next(encoder.model.parameters()).dtype),
        )
        write_json(complete, metadata)


if __name__ == "__main__":
    main()
