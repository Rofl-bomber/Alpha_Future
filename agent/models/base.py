"""Общий интерфейс моделей раннера.

fit(X_tr, y_tr, X_va, y_va, cat_cols) -> self
predict(X) -> (n,) для регрессии, (n, K) вероятности для классификации
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Параметры, которые выставляет харнесс и которые нельзя переопределять из гипотез.
RESERVED = {
    "n_jobs", "nthread", "num_threads", "thread_count", "random_state", "random_seed", "seed",
    "verbose", "verbosity", "silent", "logging_level", "num_class", "allow_writing_files",
    "train_dir", "enable_categorical", "device", "task_type", "gpu_id",
}


class ModelBase:
    name = "base"
    defaults: dict = {}

    def __init__(self, params: dict, spec, n_threads: int = 1, seed: int = 42):
        self.spec = spec
        self.is_clf = spec.is_clf
        self.n_classes = spec.n_classes
        self.n_threads = n_threads
        self.seed = seed
        self.params = {**self.defaults, **{k: v for k, v in (params or {}).items() if k not in RESERVED}}
        self.importances: dict[str, float] | None = None
        self.best_iteration: int | None = None

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    # --- утилиты
    def expand_proba(self, p: np.ndarray, classes) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = np.stack([1 - p, p], axis=1)
        classes = list(np.asarray(classes).astype(int))
        if len(classes) == self.n_classes and classes == list(range(self.n_classes)):
            return p
        out = np.zeros((p.shape[0], self.n_classes))
        for j, c in enumerate(classes):
            out[:, c] = p[:, j]
        return out


def to_numeric_frame(X: pd.DataFrame, cat_cols: list[str], fill: float | None = None) -> pd.DataFrame:
    """Категории -> коды (NaN = -1 или NaN), остальное как есть."""
    Z = X.copy()
    for c in cat_cols:
        codes = Z[c].cat.codes.astype(float) if isinstance(Z[c].dtype, pd.CategoricalDtype) else \
            pd.Categorical(Z[c]).codes.astype(float)
        codes[codes < 0] = np.nan
        Z[c] = codes
    Z = Z.astype(float)
    if fill is not None:
        Z = Z.fillna(fill)
    return Z
