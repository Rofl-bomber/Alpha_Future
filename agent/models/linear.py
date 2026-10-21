"""Ridge для регрессии, LogisticRegression для классификации.
Числовые: медиана + стандартизация с клиппингом, флаги пропусков; категории: one-hot топ-уровней."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .base import ModelBase


class _LinearPrep:
    def __init__(self, cat_cols, max_levels=30):
        self.cat_cols = list(cat_cols)
        self.max_levels = max_levels

    def fit(self, X: pd.DataFrame):
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        num = X[self.num_cols].astype(float)
        self.median = num.median()
        filled = num.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std().replace(0, 1.0).fillna(1.0)
        self.na_cols = [c for c in self.num_cols if num[c].isna().any()]
        self.levels = {}
        for c in self.cat_cols:
            vc = X[c].astype(object).value_counts()
            self.levels[c] = list(vc.index[: self.max_levels])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = X[self.num_cols].astype(float)
        parts = [((num.fillna(self.median) - self.mean) / self.std).clip(-5, 5).fillna(0.0).values]
        if self.na_cols:
            parts.append(num[self.na_cols].isna().values.astype(float))
        for c in self.cat_cols:
            v = X[c].astype(object).values
            parts.append(np.stack([(v == lv) for lv in self.levels[c]], axis=1).astype(float)
                         if self.levels[c] else np.zeros((len(X), 0)))
        return np.hstack(parts) if parts else np.zeros((len(X), 1))


class LinearModel(ModelBase):
    name = "linear"
    defaults = dict()
    uses_eval = False
    param_schema = {
        "alpha": ("float", 1e-6, 1e4),
        "C": ("float", 1e-4, 1e4),
        "max_iter": ("int", 100, 10000),
        "class_weight": ("choice", ["balanced", None]),
        "max_levels": ("int", 1, 200),
    }
    param_aliases = {"reg_lambda": "alpha", "lambda": "alpha", "l2": "alpha", "c": "C",
                     "max_categories": "max_levels"}
    clf_only = {"C", "max_iter", "class_weight"}
    reg_only = {"alpha"}
    doc_defaults = {"alpha": 1.0, "C": 1.0, "max_iter": 2000, "max_levels": 30}

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        p = dict(self.params)
        max_levels = int(p.pop("max_levels", 30))
        self.prep = _LinearPrep(cat_cols, max_levels).fit(X_tr)
        Z = self.prep.transform(X_tr)
        if self.is_clf:
            p.setdefault("C", 1.0)
            p.setdefault("max_iter", 2000)
            self.model = LogisticRegression(**p)
        else:
            p.setdefault("alpha", 1.0)
            self.model = Ridge(**p)
        self.model.fit(Z, y_tr)
        return self

    def predict(self, X):
        Z = self.prep.transform(X)
        if self.is_clf:
            return self.expand_proba(self.model.predict_proba(Z), self.model.classes_)
        return np.asarray(self.model.predict(Z), dtype=float)
