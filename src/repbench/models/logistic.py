from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SHARED_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


class LogisticCompatibility:
    def __init__(self, c_grid=SHARED_C_GRID, seed: int = 20260912):
        self.c_grid = tuple(c_grid)
        self.seed = seed

    def fit(self, x_train, y_train, x_valid, y_valid):
        best = None
        self.validation_curve_ = []
        for c_value in self.c_grid:
            model = LogisticRegression(
                C=c_value, max_iter=2000, solver="lbfgs", random_state=self.seed
            )
            model.fit(x_train, y_train)
            auc = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
            self.validation_curve_.append({"c": float(c_value), "validation_auc": float(auc)})
            if best is None or auc > best[0]:
                best = (auc, c_value, model)
        self.validation_auc_, self.best_c_, self.model_ = best
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(features)[:, 1]
