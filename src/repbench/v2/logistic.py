"""V2 grid with explicit, validation-only edge extension; v1 stays unchanged."""

from __future__ import annotations

import numpy as np

from repbench.models.logistic import LogisticCompatibility

V2_C_GRID = tuple(10.0**power for power in range(-6, 5))  # 1e-6..1e4 (owner decision D)


class GridEdgeError(ValueError):
    pass


class V2LogisticCompatibility(LogisticCompatibility):
    def __init__(self, c_grid=V2_C_GRID, seed=20260912, max_extensions=4):
        super().__init__(c_grid=c_grid, seed=seed)
        if not self.c_grid or tuple(sorted(set(self.c_grid))) != self.c_grid:
            raise ValueError("C grid must be nonempty, unique and increasing")
        if not all(np.isfinite(c) and c > 0 for c in self.c_grid):
            raise ValueError("C grid must contain finite positive values")
        if max_extensions < 0:
            raise ValueError("max_extensions must be nonnegative")
        self.max_extensions = max_extensions

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.grid_extensions_ = []
        for attempt in range(self.max_extensions + 1):
            super().fit(x_train, y_train, x_valid, y_valid)
            at_low = self.best_c_ == min(self.c_grid)
            at_high = self.best_c_ == max(self.c_grid)
            if not (at_low or at_high):
                return self
            if attempt == self.max_extensions:
                raise GridEdgeError(
                    "Validation-selected C remains at grid edge; revise before freeze"
                )
            addition = min(self.c_grid) / 10 if at_low else max(self.c_grid) * 10
            self.grid_extensions_.append({"selected_c": self.best_c_, "added_c": addition})
            self.c_grid = tuple(sorted((*self.c_grid, addition)))
        raise AssertionError("unreachable")
