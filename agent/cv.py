"""Фолды задаются один раз за прогон и общие для всех экспериментов."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class Folds:
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_id: np.ndarray
    scheme: str

    def __len__(self) -> int:
        return len(self.splits)

    def __iter__(self):
        return iter(self.splits)


def make_folds(spec, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> Folds:
    n = len(y)
    idx = np.arange(n)
    if spec.is_clf:
        strat = np.asarray(y).copy()
        counts = np.bincount(strat, minlength=max(spec.n_classes, 1))
        rare = np.where((counts > 0) & (counts < n_folds))[0]
        if len(rare):
            strat[np.isin(strat, rare)] = -1  # редкие классы стратифицируем вместе
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx, strat))
        scheme = f"StratifiedKFold({n_folds}, seed={seed})"
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx))
        scheme = f"KFold({n_folds}, seed={seed})"
    fold_id = np.empty(n, dtype=int)
    for k, (_, va) in enumerate(splits):
        fold_id[va] = k
    return Folds(splits=splits, fold_id=fold_id, scheme=scheme)
