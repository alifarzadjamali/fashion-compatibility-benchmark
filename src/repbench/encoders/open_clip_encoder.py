from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from open_clip import add_model_config, create_model_and_transforms
from PIL import Image

os.environ.setdefault("HF_HOME", str(Path(".venv/cache/huggingface").resolve()))

from .base import EncoderOutput, FrozenImageEncoder


class MarqoFashionSigLIPEncoder(FrozenImageEncoder):
    def __init__(self, checkpoint: str, revision: str, device: str | None = None):
        self.model_key = "marqo_fashionsiglip"
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights_path = hf_hub_download(
            checkpoint, "open_clip_model.safetensors", revision=revision
        )
        local_config = Path("configs/encoders/marqo_fashionsiglip_open_clip.json")
        add_model_config(local_config)
        precision = "fp16" if self.device.type == "cuda" else "fp32"
        self.model, _, self.transform = create_model_and_transforms(
            local_config.stem,
            pretrained=weights_path,
            precision=precision,
            device=self.device,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            image_interpolation="bicubic",
            image_resize_mode="squash",
        )
        self.model.eval().requires_grad_(False)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def encode(self, paths: Sequence[Path], item_ids: Sequence[str], batch_size: int = 32):
        if len(paths) != len(item_ids):
            raise ValueError("paths and item_ids differ in length")
        chunks = []
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            for offset in range(0, len(paths), batch_size):
                images = []
                for path in paths[offset : offset + batch_size]:
                    try:
                        with Image.open(path) as image:
                            images.append(self.transform(image.convert("RGB")))
                    except Exception as exc:
                        raise RuntimeError(f"Failed image: {path}") from exc
                pixels = torch.stack(images).to(self.device, dtype=model_dtype)
                features = self.model.encode_image(pixels, normalize=True)
                chunks.append(features.float().cpu().numpy())
        elapsed = time.perf_counter() - started
        embeddings = np.concatenate(chunks).astype(np.float32)
        return EncoderOutput(
            list(item_ids),
            embeddings,
            {
                "model_key": self.model_key,
                "checkpoint": self.checkpoint,
                "checkpoint_revision": self.revision,
                "parameter_count": self.parameter_count,
                "embedding_dim": embeddings.shape[1],
                "input_resolution": 224,
                "dtype": "float32",
                "device": str(self.device),
                "normalization": "l2",
                "trust_remote_code": False,
                "elapsed_seconds": elapsed,
                "images_per_second": len(paths) / elapsed,
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else None
                ),
            },
        )
