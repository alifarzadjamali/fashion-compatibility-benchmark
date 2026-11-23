"""Booking-local data/cache paths, independent of virtual environments."""

from __future__ import annotations

import os
from pathlib import Path


def data_path(*parts: str) -> Path:
    return Path(os.environ.get("REPBENCH_DATA_ROOT", "data")).expanduser().joinpath(*parts)


def configure_cache_environment() -> None:
    root = Path(os.environ.get("REPBENCH_CACHE_ROOT", str(data_path("cache")))).expanduser()
    os.environ.setdefault("HF_HOME", str((root / "huggingface").resolve()))
    os.environ.setdefault("TORCH_HOME", str((root / "torch").resolve()))
