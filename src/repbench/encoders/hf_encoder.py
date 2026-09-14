from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

os.environ.setdefault("HF_HOME", str(Path(".venv/cache/huggingface").resolve()))

from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from .base import EncoderOutput, FrozenImageEncoder


class _GRLiteProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("GR-Lite processor only supports PyTorch tensors")
        return {"pixel_values": torch.stack([self.transform(image) for image in images])}


class HuggingFaceImageEncoder(FrozenImageEncoder):
    def __init__(
        self,
        model_key: str,
        checkpoint: str,
        revision: str,
        device: str | None = None,
        trust_remote_code: bool = False,
        use_safetensors: bool | None = True,
        cuda_dtype: str = "float16",
    ):
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.trust_remote_code = trust_remote_code
        if model_key == "gr_lite":
            self.processor = _GRLiteProcessor()
        else:
            use_fast = model_key == "dinov3_vitl16"
            try:
                self.processor = AutoProcessor.from_pretrained(
                    checkpoint,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                )
            except (ImportError, ValueError, TypeError):
                self.processor = AutoImageProcessor.from_pretrained(
                    checkpoint,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                )
        if self.device.type == "cuda":
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[cuda_dtype]
        else:
            dtype = torch.float32
        load_kwargs = {
            "revision": revision,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if use_safetensors is not None:
            load_kwargs["use_safetensors"] = use_safetensors
        self.model = AutoModel.from_pretrained(checkpoint, **load_kwargs).eval().to(self.device)
        self.model.requires_grad_(False)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.input_resolution = getattr(
            getattr(self.model.config, "vision_config", self.model.config), "image_size", None
        )

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
                            images.append(image.convert("RGB"))
                    except Exception as exc:
                        raise RuntimeError(f"Failed image: {path}") from exc
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {
                    key: value.to(
                        self.device,
                        dtype=model_dtype if torch.is_floating_point(value) else value.dtype,
                    )
                    for key, value in inputs.items()
                }
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                    if not isinstance(features, torch.Tensor):
                        features = features.pooler_output
                    outputs = None
                else:
                    outputs = self.model(**inputs)
                if (
                    outputs is not None
                    and hasattr(outputs, "image_embeds")
                    and outputs.image_embeds is not None
                ):
                    features = outputs.image_embeds
                elif (
                    outputs is not None
                    and hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    features = outputs.pooler_output
                elif outputs is not None:
                    features = outputs.last_hidden_state[:, 0]
                chunks.append(torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy())
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
                "input_resolution": self.input_resolution,
                "dtype": "float32",
                "inference_dtype": str(model_dtype),
                "device": str(self.device),
                "normalization": "l2",
                "trust_remote_code": self.trust_remote_code,
                "elapsed_seconds": elapsed,
                "images_per_second": len(paths) / elapsed,
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else None
                ),
            },
        )
