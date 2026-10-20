from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .base import ModelBase


def _prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Z = X.copy()
    for c in cat_cols:
        Z[c] = Z[c].astype(object).where(Z[c].notna(), "nan").astype(str)
    return Z


class CatBoostModel(ModelBase):
    name = "catboost"
    defaults = dict(iterations=2000, learning_rate=0.08, depth=6, l2_leaf_reg=3.0, early_stopping_rounds=100)

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        p = dict(self.params)
        es = int(p.pop("early_stopping_rounds", 100))
        self.cat_cols = list(cat_cols)
        common = dict(thread_count=self.n_threads, random_seed=self.seed, verbose=False,
                      allow_writing_files=False)
        if self.is_clf:
            p.setdefault("loss_function", "MultiClass" if self.n_classes > 2 else "Logloss")
            self.model = CatBoostClassifier(**p, **common)
        else:
            p.setdefault("loss_function", "RMSE")
            self.model = CatBoostRegressor(**p, **common)
        tr = Pool(_prep(X_tr, self.cat_cols), y_tr, cat_features=self.cat_cols)
        va = Pool(_prep(X_va, self.cat_cols), y_va, cat_features=self.cat_cols)
        self.model.fit(tr, eval_set=va, early_stopping_rounds=es, use_best_model=True)
        self.best_iteration = self.model.get_best_iteration()
        self.importances = dict(zip(X_tr.columns, map(float, self.model.get_feature_importance())))
        return self

    def predict(self, X):
        Z = Pool(_prep(X, self.cat_cols), cat_features=self.cat_cols)
        if self.is_clf:
            return self.expand_proba(self.model.predict_proba(Z), self.model.classes_)
        return np.asarray(self.model.predict(Z), dtype=float)
