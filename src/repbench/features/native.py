"""Train-only standardization and L2 normalization at native feature width."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize


class TrainOnlyNative:
    def fit(self, train):
        self._validate(train)
        self.scaler_ = StandardScaler().fit(train)
        self.n_train_samples_ = len(train)
        return self

    def transform(self, values):
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Fit on training items before transformation")
        self._validate(values)
        return normalize(self.scaler_.transform(values), norm="l2").astype(np.float32)

    @staticmethod
    def _validate(values):
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Embeddings must be a nonempty finite rank-2 array")
