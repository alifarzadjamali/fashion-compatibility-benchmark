"""Fixed, order-invariant pairwise outfit representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np


def outfit_feature(item_ids: Sequence[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    if len(item_ids) < 2:
        raise ValueError("An outfit must contain at least two items")
    vectors = [np.asarray(embeddings[item], dtype=np.float32) for item in item_ids]
    if len({vector.shape for vector in vectors}) != 1:
        raise ValueError("All item embeddings must have the same shape")
    pair_features = [np.concatenate((np.abs(a - b), a * b)) for a, b in combinations(vectors, 2)]
    return np.mean(pair_features, axis=0, dtype=np.float32)


def outfit_matrix(
    outfits: Sequence[Sequence[str]], embeddings: Mapping[str, np.ndarray]
) -> np.ndarray:
    return np.stack([outfit_feature(outfit, embeddings) for outfit in outfits])
