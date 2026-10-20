from __future__ import annotations

import lightgbm as lgb
import numpy as np

from .base import ModelBase


class LightGBMModel(ModelBase):
    name = "lightgbm"
    defaults = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    early_stopping_rounds=100)

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        p = dict(self.params)
        es = int(p.pop("early_stopping_rounds", 100))
        common = dict(n_jobs=self.n_threads, random_state=self.seed, verbose=-1)
        if self.is_clf:
            self.model = lgb.LGBMClassifier(**p, **common)
        else:
            self.model = lgb.LGBMRegressor(**p, **common)
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                       callbacks=[lgb.early_stopping(es, verbose=False)])
        self.best_iteration = self.model.best_iteration_
        imp = self.model.booster_.feature_importance(importance_type="gain")
        # LightGBM может переименовать колонки со спецсимволами — сопоставляем по позиции
        self.importances = dict(zip(list(X_tr.columns), map(float, imp)))
        return self

    def predict(self, X):
        if self.is_clf:
            return self.expand_proba(self.model.predict_proba(X), self.model.classes_)
        return np.asarray(self.model.predict(X), dtype=float)
