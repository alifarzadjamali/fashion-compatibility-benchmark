"""Repeatable, same-hardware image-encoder and downstream compute profile."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".venv/cache/huggingface").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".venv/cache/torch").resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.flop_counter import FlopCounterMode

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.hf_encoder import HuggingFaceImageEncoder
from repbench.encoders.open_clip_encoder import MarqoFashionSigLIPEncoder
from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS
from repbench.encoders.torchvision_encoder import ResNet50Encoder
from repbench.features.outfit_features import outfit_feature, outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import LogisticCompatibility

BATCH_SIZE = 64
WARMUP_REPEATS = 2
TIMING_REPEATS = 5
SEED = 20260912


class ImageForward(torch.nn.Module):
    def __init__(self, encoder, model_key: str):
        super().__init__()
        self.encoder = encoder
        self.model_key = model_key

    def forward(self, pixels):
        if self.model_key == "resnet50":
            return self.encoder.model(pixels).flatten(1)
        if self.model_key == "marqo_fashionsiglip":
            return self.encoder.model.encode_image(pixels, normalize=False)
        model = self.encoder.model
        if hasattr(model, "get_image_features"):
            output = model.get_image_features(pixel_values=pixels)
            return output if isinstance(output, torch.Tensor) else output.pooler_output
        output = model(pixel_values=pixels)
        if getattr(output, "image_embeds", None) is not None:
            return output.image_embeds
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]


def make_encoder(model_key: str):
    spec = PRIMARY_ENCODERS[model_key]
    if spec.backend == "torchvision":
        return ResNet50Encoder()
    if spec.backend == "open_clip_pinned":
        return MarqoFashionSigLIPEncoder(spec.checkpoint, spec.revision)
    return HuggingFaceImageEncoder(
        spec.model_key,
        spec.checkpoint,
        spec.revision,
        trust_remote_code=spec.trust_remote_code,
        use_safetensors=spec.use_safetensors,
        cuda_dtype=spec.cuda_dtype,
    )


def prepared_pixels(encoder, model_key: str, path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if model_key in {"resnet50", "marqo_fashionsiglip"}:
            pixels = encoder.transform(rgb).unsqueeze(0)
        else:
            pixels = encoder.processor(images=[rgb], return_tensors="pt")["pixel_values"]
    dtype = next(encoder.model.parameters()).dtype
    return pixels.to(encoder.device, dtype=dtype)


def active_parameters(encoder, model_key: str) -> int:
    if model_key == "resnet50":
        modules = [encoder.model]
    elif model_key == "marqo_fashionsiglip":
        modules = [encoder.model.visual]
    elif hasattr(encoder.model, "vision_model"):
        modules = [encoder.model.vision_model]
        if isinstance(getattr(encoder.model, "visual_projection", None), torch.nn.Module):
            modules.append(encoder.model.visual_projection)
    else:
        modules = [encoder.model]
    unique = {id(parameter): parameter for module in modules for parameter in module.parameters()}
    return sum(parameter.numel() for parameter in unique.values())


def encoder_profile(dataset: PolyvoreDisjoint, model_key: str) -> dict:
    encoder = make_encoder(model_key)
    item_ids = sorted(dataset.item_ids("test"))[:BATCH_SIZE]
    paths = [dataset.image_path(item_id) for item_id in item_ids]
    for _ in range(WARMUP_REPEATS):
        encoder.encode(paths, item_ids, BATCH_SIZE)
    repetitions = []
    peak = 0.0
    for _ in range(TIMING_REPEATS):
        output = encoder.encode(paths, item_ids, BATCH_SIZE)
        repetitions.append(output.metadata["elapsed_seconds"])
        peak = max(peak, output.metadata["peak_vram_mb"] or 0.0)

    pixels = prepared_pixels(encoder, model_key, paths[0])
    wrapper = ImageForward(encoder, model_key).eval()
    try:
        with torch.inference_mode(), FlopCounterMode(wrapper, display=False) as flop_counter:
            wrapper(pixels)
        flops = int(flop_counter.get_total_flops())
        flop_status = "PyTorch FlopCounterMode forward estimate"
    except Exception as exc:  # noqa: BLE001 - preserve all other measurements if profiling fails
        flops = np.nan
        flop_status = f"unavailable: {type(exc).__name__}: {exc}"
    result = {
        "representation": model_key,
        "batch_size": BATCH_SIZE,
        "warmup_repeats": WARMUP_REPEATS,
        "timing_repeats": TIMING_REPEATS,
        "inference_precision": str(next(encoder.model.parameters()).dtype),
        "active_image_parameters": active_parameters(encoder, model_key),
        "repeated_elapsed_mean_seconds": float(np.mean(repetitions)),
        "repeated_elapsed_sd_seconds": float(np.std(repetitions, ddof=1)),
        "repeated_images_per_second": BATCH_SIZE / float(np.mean(repetitions)),
        "repeated_latency_ms_per_image": 1000 * float(np.mean(repetitions)) / BATCH_SIZE,
        "repeated_peak_vram_mb": peak,
        "estimated_flops_per_image": flops,
        "estimated_gflops_per_image": flops / 1e9 if np.isfinite(flops) else np.nan,
        "flop_count_method": flop_status,
    }
    del wrapper, pixels, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_split(cache: Path, split: str):
    item_ids = json.loads((cache / f"{split}_item_ids.json").read_text(encoding="utf-8"))
    return item_ids, np.load(cache / f"{split}.npy").astype(np.float32, copy=False)


def downstream_profile(dataset: PolyvoreDisjoint, model_key: str) -> dict:
    cache = Path("data/embeddings/historical_polyvore_d") / model_key
    split_data = {split: load_split(cache, split) for split in dataset.SPLITS}
    pca = TrainOnlyPCA(256, seed=SEED)
    started = time.perf_counter()
    transformed = pca.fit_transform_splits(
        *(split_data[split][1] for split in dataset.SPLITS)
    )
    pca_seconds = time.perf_counter() - started
    embeddings = {
        split: dict(zip(split_data[split][0], transformed[index], strict=True))
        for index, split in enumerate(dataset.SPLITS)
    }
    cp = {split: dataset.compatibility(split) for split in dataset.SPLITS}
    x = {
        split: outfit_matrix([row.item_ids for row in cp[split]], embeddings[split])
        for split in dataset.SPLITS
    }
    y = {split: np.asarray([row.label for row in cp[split]]) for split in dataset.SPLITS}
    scaler = StandardScaler().fit(x["train"])
    x = {split: scaler.transform(value) for split, value in x.items()}
    classifier = LogisticCompatibility(seed=SEED).fit(
        x["train"], y["train"], x["valid"], y["valid"]
    )
    outfit_ids = dataset.outfits("test")[0].item_ids
    repeats = 1000
    started = time.perf_counter()
    for _ in range(repeats):
        feature = outfit_feature(outfit_ids, embeddings["test"])[None, :]
        classifier.predict_proba(scaler.transform(feature))
    scoring_ms = 1000 * (time.perf_counter() - started) / repeats
    return {
        "representation": model_key,
        "pca_fit_transform_seconds": pca_seconds,
        "end_to_end_outfit_scoring_ms": scoring_ms,
        "timed_outfit_items": len(outfit_ids),
    }


def main() -> None:
    dataset = PolyvoreDisjoint(Path("data/raw/polyvore_outfits"))
    access = json.loads(Path("artifacts/checkpoint_access_audit.json").read_text(encoding="utf-8"))
    sizes = {
        row["model_key"]: row["weight_size"] / 2**20 for row in access["models"]
    }
    audit = pd.read_csv("artifacts/model_audit.csv").set_index("model_key")
    prior = pd.read_csv("artifacts/primary_results.csv").set_index("representation")
    rows = []
    for model_key in PRIMARY_MODEL_KEYS:
        print(f"Profiling {model_key}", flush=True)
        measured = encoder_profile(dataset, model_key)
        downstream = downstream_profile(dataset, model_key)
        rows.append(
            {
                **measured,
                **downstream,
                "checkpoint_parameters": int(audit.loc[model_key, "parameter_count"]),
                "native_embedding_dim": int(audit.loc[model_key, "native_embedding_dimension"]),
                "input_resolution": int(audit.loc[model_key, "official_input_resolution"]),
                "checkpoint_size_mb": sizes[model_key],
                "full_extraction_images_per_second": prior.loc[
                    model_key, "extract_images_per_second"
                ],
                "full_extraction_latency_ms_per_image": 1000
                / prior.loc[model_key, "extract_images_per_second"],
                "full_extraction_seconds": prior.loc[model_key, "extraction_seconds"],
                "full_extraction_peak_vram_mb": prior.loc[model_key, "peak_vram_mb"],
                "embedding_storage_mb": prior.loc[model_key, "embedding_storage_mb"],
                "classifier_train_seconds": prior.loc[model_key, "train_seconds"],
            }
        )
        pd.DataFrame(rows).to_csv("artifacts/efficiency_profile.partial.csv", index=False)
    pd.DataFrame(rows).to_csv("artifacts/efficiency_profile.csv", index=False)
    Path("artifacts/efficiency_environment.json").write_text(
        json.dumps(
            {
                "hardware": "NVIDIA GeForce RTX 5070 Ti 16 GB",
                "gpu_reported": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "batch_size": BATCH_SIZE,
                "timing_scope": "decode + official preprocessing + frozen image forward + CPU copy",
                "flop_definition": (
                    "FLOPs reported by torch.utils.flop_counter.FlopCounterMode for one image; "
                    "multiply-add conventions follow PyTorch operator mappings."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
