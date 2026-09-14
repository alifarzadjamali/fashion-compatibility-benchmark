from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.hf_encoder import HuggingFaceImageEncoder
from repbench.encoders.registry import PRIMARY_ENCODERS


def extract(model_key: str, dtype: str, paths: list[Path], ids: list[str]) -> np.ndarray:
    spec = PRIMARY_ENCODERS[model_key]
    encoder = HuggingFaceImageEncoder(
        model_key,
        spec.checkpoint,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        use_safetensors=spec.use_safetensors,
        cuda_dtype=dtype,
    )
    output = encoder.encode(paths, ids, batch_size=len(paths)).embeddings
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main() -> None:
    dataset = PolyvoreDisjoint("data/raw/polyvore_outfits")
    ids = sorted(dataset.item_ids("test"))[:16]
    paths = [dataset.image_path(item_id) for item_id in ids]
    rows = []
    for model_key in ("dinov3_vitl16", "gr_lite"):
        reduced = extract(model_key, "bfloat16", paths, ids)
        reference = extract(model_key, "float32", paths, ids)
        cosine = np.sum(reduced * reference, axis=1)
        rows.append(
            {
                "model_key": model_key,
                "samples": len(ids),
                "reduced_precision": "bfloat16",
                "reference_precision": "float32",
                "all_finite": bool(np.isfinite(reduced).all()),
                "mean_cosine_similarity": float(cosine.mean()),
                "min_cosine_similarity": float(cosine.min()),
                "max_absolute_coordinate_difference": float(
                    np.max(np.abs(reduced - reference))
                ),
            }
        )
    Path("artifacts/dinofamily_precision_audit.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
