"""Leakage-resistant controlled representation bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


@dataclass
class TrainOnlyPCA:
    n_components: int = 256
    seed: int = 20260912
    standardize: bool = True

    def fit(self, train: np.ndarray) -> TrainOnlyPCA:
        self._validate(train)
        if train.shape[0] < self.n_components or train.shape[1] < self.n_components:
            raise ValueError("PCA-256 requires at least 256 training items and raw dimensions")
        self.scaler_ = StandardScaler() if self.standardize else None
        fitted = self.scaler_.fit_transform(train) if self.scaler_ else train
        self.pca_ = PCA(
            n_components=self.n_components, svd_solver="randomized", random_state=self.seed
        )
        self.pca_.fit(fitted)
        self.n_train_samples_ = train.shape[0]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pca_"):
            raise RuntimeError("fit must be called before transform")
        self._validate(values)
        scaled = self.scaler_.transform(values) if self.scaler_ else values
        return normalize(self.pca_.transform(scaled), norm="l2").astype(np.float32)

    def fit_transform_splits(self, train: np.ndarray, valid: np.ndarray, test: np.ndarray):
        self.fit(train)
        return self.transform(train), self.transform(valid), self.transform(test)

    @property
    def explained_variance_ratio(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum())

    @staticmethod
    def _validate(values: np.ndarray) -> None:
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Embeddings must be a finite rank-2 array")
