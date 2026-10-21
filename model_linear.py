"""Линейные модели (Ridge / LogisticRegression) и общий препроцессор для моделей без деревьев.

Препроцессор: числовые — медианная импутация, флаги пропусков, обрезка хвостов, масштабирование;
категории — one-hot до 30 уровней, остальные — частота. Подгоняется только на train фолда.
"""

import numpy as np
import pandas as pd

import task_metrics

FAMILY = "linear"

DEFAULTS = {"alpha": 1.0, "C": 1.0, "max_iter": 400}
RANGES = {"alpha": (float, 1e-4, 1e4), "C": (float, 1e-4, 1e3), "max_iter": (int, 50, 5000)}


def space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


class Preprocessor:
    def __init__(self, quantile=False, max_levels=30):
        self.quantile = quantile
        self.max_levels = max_levels

    def fit(self, X: pd.DataFrame):
        self.cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.nums = [c for c in X.columns if c not in self.cats and pd.api.types.is_numeric_dtype(X[c])]
        self.levels, self.freqs = {}, {}
        for c in self.cats:
            vc = X[c].astype(str).value_counts()
            if len(vc) <= self.max_levels:
                self.levels[c] = list(vc.index)
            else:
                self.freqs[c] = (vc / vc.sum()).to_dict()
        N = X[self.nums].astype(float)
        self.median = N.median()
        self.lo = N.quantile(0.001)
        self.hi = N.quantile(0.999)
        self.na_cols = [c for c in self.nums if X[c].isna().any()]
        Z = self._nums(X)
        if self.quantile:
            from sklearn.preprocessing import QuantileTransformer

            self.qt = QuantileTransformer(n_quantiles=min(1000, max(10, len(X))), output_distribution="normal",
                                          subsample=200_000, random_state=0)
            Z = self.qt.fit_transform(Z)
        self.mu = Z.mean(axis=0)
        self.sd = Z.std(axis=0) + 1e-9
        return self

    def _nums(self, X):
        N = X[self.nums].astype(float).clip(self.lo, self.hi, axis=1)
        return N.fillna(self.median).fillna(0.0).to_numpy()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._nums(X)
        if self.quantile:
            Z = self.qt.transform(Z)
        Z = (Z - self.mu) / self.sd
        parts = [Z]
        if self.na_cols:
            parts.append(X[self.na_cols].isna().astype(float).to_numpy())
        for c, lv in self.levels.items():
            s = X[c].astype(str).to_numpy()
            parts.append(np.stack([(s == v).astype(float) for v in lv], axis=1))
        for c, fr in self.freqs.items():
            parts.append(X[c].astype(str).map(fr).fillna(0.0).to_numpy(dtype=float)[:, None])
        return np.hstack(parts).astype(np.float32)


class Model:
    family = FAMILY

    def __init__(self, task_type, n_classes, params=None, seed=0, n_jobs=8, fast=False, objective="l2",
                 deadline=None):
        self.task_type = task_type
        self.n_classes = n_classes
        self.params = {**DEFAULTS, **(params or {})}
        self.seed = seed
        self.n_jobs = n_jobs
        self.best_iteration = None

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        from sklearn.linear_model import LogisticRegression, Ridge

        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        if self.task_type == "regression":
            self.model = Ridge(alpha=float(self.params["alpha"]), random_state=self.seed)
        else:
            self.model = LogisticRegression(C=float(self.params["C"]), max_iter=int(self.params["max_iter"]),
                                            n_jobs=self.n_jobs, random_state=self.seed)
        self.model.fit(Z, y, sample_weight=w)
        return self

    def predict(self, X):
        Z = self.pre.transform(X)
        if self.task_type == "regression":
            return np.asarray(self.model.predict(Z), dtype=float)
        proba = self.model.predict_proba(Z)
        full = task_metrics.expand_proba(proba, [int(c) for c in self.model.classes_], max(self.n_classes, 2))
        return full[:, 1] if self.task_type == "binary" else full

    def importance(self) -> dict:
        return {}
