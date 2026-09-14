"""Extract the locked seven frozen representations once for external A100."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from repbench.encoders.hf_encoder import HuggingFaceImageEncoder
from repbench.encoders.open_clip_encoder import MarqoFashionSigLIPEncoder
from repbench.encoders.registry import PRIMARY_MODEL_KEYS, get_encoder_spec
from repbench.encoders.torchvision_encoder import ResNet50Encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path(".venv/a100/images"))
    parser.add_argument("--output", type=Path, default=Path(".venv/a100/embeddings"))
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODEL_KEYS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-valid-existing", action="store_true")
    args = parser.parse_args()
    item_ids = sorted(path.stem for path in args.images.glob("*.jpg"))
    paths = [args.images / f"{item_id}.jpg" for item_id in item_ids]
    if len(item_ids) != 1663:
        raise ValueError(f"Unexpected A100 referenced image count: {len(item_ids)}")
    for model_key in args.models:
        spec = get_encoder_spec(model_key)
        destination = args.output / model_key
        values_path = destination / "all.npy"
        ids_path = destination / "all_item_ids.json"
        metadata_path = destination / "all_metadata.json"
        if args.skip_valid_existing and all(path.is_file() for path in (values_path, ids_path, metadata_path)):
            values = np.load(values_path)
            if values.shape == (len(item_ids), spec.embedding_dim) and json.loads(ids_path.read_text()) == item_ids and np.isfinite(values).all():
                print(f"{model_key}: reusing validated A100 cache", flush=True)
                continue
            raise ValueError(f"Invalid existing A100 cache: {model_key}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite A100 cache: {destination}")
        destination.mkdir(parents=True)
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
        result = encoder.encode(paths, item_ids, args.batch_size)
        np.save(values_path, result.embeddings)
        ids_path.write_text(json.dumps(result.item_ids), encoding="utf-8")
        result.metadata.update(
            {
                "dataset": "a100_external",
                "extraction_date": datetime.now(UTC).isoformat(),
                "embedding_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
                "item_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            }
        )
        metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
        print(f"{model_key}: extracted {len(item_ids)} A100 images", flush=True)


if __name__ == "__main__":
    main()
