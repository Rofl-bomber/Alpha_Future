from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import ModelBase


class XGBoostModel(ModelBase):
    name = "xgboost"
    defaults = dict(n_estimators=3000, learning_rate=0.05, max_depth=7, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, max_bin=256,
                    early_stopping_rounds=100)

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        p = dict(self.params)
        common = dict(tree_method="hist", enable_categorical=True, n_jobs=self.n_threads,
                      random_state=self.seed, verbosity=0)
        if self.is_clf:
            # xgboost требует сплошные метки 0..m-1 — перекодируем присутствующие классы
            self.present = np.unique(y_tr)
            remap = {c: i for i, c in enumerate(self.present)}
            ytr = np.array([remap[v] for v in y_tr])
            mask_va = np.isin(y_va, self.present)
            yva = np.array([remap[v] for v in y_va[mask_va]])
            self.model = xgb.XGBClassifier(**p, **common)
            self.model.fit(X_tr, ytr, eval_set=[(X_va[mask_va], yva)], verbose=False)
        else:
            self.model = xgb.XGBRegressor(**p, **common)
            self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.best_iteration = getattr(self.model, "best_iteration", None)
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        self.importances = {c: float(score.get(c, 0.0)) for c in X_tr.columns}
        return self

    def predict(self, X):
        if self.is_clf:
            return self.expand_proba(self.model.predict_proba(X), self.present)
        return np.asarray(self.model.predict(X), dtype=float)
