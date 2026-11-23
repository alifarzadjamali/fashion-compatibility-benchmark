"""V2 operator sensitivity; the original v1 operator remains unchanged."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def pair_outfit_feature(item_ids, embeddings, *, operator="abs_product"):
    if operator not in {"abs_product", "abs_product_sum"}:
        raise ValueError("Unregistered pair operator")
    if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
        raise ValueError("Outfit must contain at least two distinct items")
    vectors = [np.asarray(embeddings[i], dtype=np.float32) for i in item_ids]
    if any(v.ndim != 1 or v.shape != vectors[0].shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError("Item vectors must be finite and have the same width")
    width = len(vectors[0]) * (3 if operator == "abs_product_sum" else 2)
    total = np.zeros(width, dtype=np.float32)
    count = 0
    for left, right in combinations(vectors, 2):
        blocks = [np.abs(left - right), left * right]
        if operator == "abs_product_sum":
            blocks.append(left + right)
        total += np.concatenate(blocks)
        count += 1
    return total / count


def pair_outfit_matrix(outfits, embeddings, *, operator="abs_product"):
    return np.stack(
        [pair_outfit_feature(items, embeddings, operator=operator) for items in outfits]
    )
