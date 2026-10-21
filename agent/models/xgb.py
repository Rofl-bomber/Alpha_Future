from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import ModelBase


class XGBoostModel(ModelBase):
    name = "xgboost"
    defaults = dict(n_estimators=3000, learning_rate=0.05, max_depth=7, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, max_bin=256,
                    early_stopping_rounds=100)
    param_schema = {
        "learning_rate": ("float", 0.005, 0.3),
        "n_estimators": ("int", 50, 10000),
        "max_depth": ("int", 1, 16),
        "max_leaves": ("int", 0, 1024),
        "grow_policy": ("choice", ["depthwise", "lossguide"]),
        "min_child_weight": ("float", 0.0, 100.0),
        "subsample": ("float", 0.3, 1.0),
        "colsample_bytree": ("float", 0.2, 1.0),
        "colsample_bylevel": ("float", 0.2, 1.0),
        "colsample_bynode": ("float", 0.2, 1.0),
        "reg_lambda": ("float", 0.0, 100.0),
        "reg_alpha": ("float", 0.0, 100.0),
        "gamma": ("float", 0.0, 20.0),
        "max_bin": ("int", 16, 1024),
        "max_cat_to_onehot": ("int", 1, 64),
        "objective": ("objective",),
        "huber_slope": ("float", 1e-6, 1e6),
        "scale_pos_weight": ("float", 0.0, 1000.0),
        "early_stopping_rounds": ("int", 20, 500),
    }
    objectives = {
        "regression": ["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"],
        "binary": ["binary:logistic"],
        "multiclass": ["multi:softprob"],
    }
    param_aliases = {
        "eta": "learning_rate", "lr": "learning_rate",
        "num_boost_round": "n_estimators", "iterations": "n_estimators", "num_trees": "n_estimators",
        "depth": "max_depth", "num_leaves": "max_leaves",
        "lambda": "reg_lambda", "l2_leaf_reg": "reg_lambda", "reg_l2": "reg_lambda", "lambda_l2": "reg_lambda",
        "alpha": "reg_alpha", "reg_l1": "reg_alpha", "lambda_l1": "reg_alpha",
        "min_split_loss": "gamma", "min_split_gain": "gamma",
        "feature_fraction": "colsample_bytree", "bagging_fraction": "subsample",
        "huber_delta": "huber_slope", "delta": "huber_slope",
        "max_bins": "max_bin", "loss": "objective", "loss_function": "objective",
        "early_stopping": "early_stopping_rounds",
    }
    clf_only = {"scale_pos_weight"}
    reg_only = {"huber_slope"}
    doc_defaults = {"max_leaves": 0, "grow_policy": "depthwise", "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
                    "reg_alpha": 0.0, "gamma": 0.0, "huber_slope": 1.0,
                    "objective": {"regression": "reg:squarederror", "binary": "binary:logistic",
                                  "multiclass": "multi:softprob"}}

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
