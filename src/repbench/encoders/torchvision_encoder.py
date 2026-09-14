from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_HOME", str(Path(".venv/cache/torch").resolve()))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from .base import EncoderOutput, FrozenImageEncoder


class _Images(Dataset):
    def __init__(self, paths: Sequence[Path], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed image: {path}") from exc


class ResNet50Encoder(FrozenImageEncoder):
    model_key = "resnet50"

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=self.weights)
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(self.device)
        self.model.requires_grad_(False)
        self.transform = self.weights.transforms()
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def encode(self, paths: Sequence[Path], item_ids: Sequence[str], batch_size: int = 64):
        if len(paths) != len(item_ids):
            raise ValueError("paths and item_ids differ in length")
        loader = DataLoader(
            _Images(paths, self.transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
        )
        chunks = []
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for images in loader:
                features = self.model(images.to(self.device, non_blocking=True)).flatten(1)
                features = torch.nn.functional.normalize(features, dim=1)
                chunks.append(features.cpu().numpy().astype(np.float32))
        elapsed = time.perf_counter() - started
        return EncoderOutput(
            list(item_ids),
            np.concatenate(chunks),
            {
                "model_key": self.model_key,
                "checkpoint": str(self.weights),
                "checkpoint_revision": "11ad3fa6",
                "parameter_count": self.parameter_count,
                "embedding_dim": 2048,
                "input_resolution": 224,
                "dtype": "float32",
                "device": str(self.device),
                "normalization": "l2",
                "dataloader_workers": min(8, os.cpu_count() or 1),
                "elapsed_seconds": elapsed,
                "images_per_second": len(paths) / elapsed,
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else None
                ),
            },
        )
