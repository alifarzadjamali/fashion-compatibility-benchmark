from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.hf_encoder import HuggingFaceImageEncoder
from repbench.encoders.open_clip_encoder import MarqoFashionSigLIPEncoder
from repbench.encoders.registry import get_encoder_spec
from repbench.encoders.torchvision_encoder import ResNet50Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dataset-key", default="historical_polyvore_d")
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--skip-valid-existing", action="store_true")
    args = parser.parse_args()
    dataset = PolyvoreDisjoint(args.root, args.protocol_dir, args.dataset_key)
    spec = get_encoder_spec(args.model_key)
    if args.trust_remote_code and not spec.trust_remote_code:
        raise ValueError("Remote code may only be enabled for an audited registry entry")
    if spec.backend == "torchvision":
        encoder = ResNet50Encoder()
    elif spec.backend == "open_clip_pinned":
        encoder = MarqoFashionSigLIPEncoder(spec.checkpoint, spec.revision)
    else:
        encoder = HuggingFaceImageEncoder(
            spec.model_key,
            spec.checkpoint,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            use_safetensors=spec.use_safetensors,
            cuda_dtype=spec.cuda_dtype,
        )
    for split in dataset.SPLITS:
        item_ids = sorted(dataset.item_ids(split))
        if args.limit:
            item_ids = item_ids[: args.limit]
        destination = args.output_root / args.dataset_key / args.model_key
        destination.mkdir(parents=True, exist_ok=True)
        values_path = destination / f"{split}.npy"
        ids_path = destination / f"{split}_item_ids.json"
        metadata_path = destination / f"{split}_metadata.json"
        if args.skip_valid_existing and values_path.is_file() and ids_path.is_file() and metadata_path.is_file():
            existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            values = np.load(values_path, mmap_mode="r")
            if existing_ids == item_ids and values.shape == (len(item_ids), spec.embedding_dim) and np.isfinite(values).all():
                print(f"{args.model_key}: reusing validated {split} cache", flush=True)
                continue
            raise ValueError(f"Existing cache failed immutable validation: {destination}/{split}")
        paths = [dataset.image_path(item_id) for item_id in item_ids]
        print(f"{args.model_key}: extracting {split} ({len(item_ids)} images)", flush=True)
        output = encoder.encode(paths, item_ids, args.batch_size)
        np.save(values_path, output.embeddings)
        ids_path.write_text(
            json.dumps(output.item_ids), encoding="utf-8"
        )
        output.metadata["extraction_date"] = datetime.now(UTC).isoformat()
        output.metadata["dataset_key"] = args.dataset_key
        output.metadata["batch_size"] = args.batch_size
        output.metadata["protocol_manifest_sha256"] = (
            hashlib.sha256((args.protocol_dir / "manifest.json").read_bytes()).hexdigest()
            if args.protocol_dir else None
        )
        output.metadata["embedding_sha256"] = hashlib.sha256(values_path.read_bytes()).hexdigest()
        output.metadata["item_ids_sha256"] = hashlib.sha256(ids_path.read_bytes()).hexdigest()
        metadata_path.write_text(
            json.dumps(output.metadata, indent=2), encoding="utf-8"
        )
        print(
            f"{args.model_key}: finished {split} at "
            f"{output.metadata['images_per_second']:.2f} images/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
