from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EncoderOutput:
    item_ids: list[str]
    embeddings: np.ndarray
    metadata: dict


class FrozenImageEncoder(ABC):
    model_key: str

    @abstractmethod
    def encode(
        self, paths: Sequence[Path], item_ids: Sequence[str], batch_size: int
    ) -> EncoderOutput:
        raise NotImplementedError
