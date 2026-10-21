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
    param_schema = {
        "iterations": ("int", 50, 10000),
        "learning_rate": ("float", 0.005, 0.3),
        "depth": ("int", 2, 10),
        "l2_leaf_reg": ("float", 0.0, 100.0),
        "random_strength": ("float", 0.0, 20.0),
        "bagging_temperature": ("float", 0.0, 10.0),
        "bootstrap_type": ("choice", ["Bayesian", "Bernoulli", "MVS", "No"]),
        "subsample": ("float", 0.3, 1.0),
        "rsm": ("float", 0.1, 1.0),
        "border_count": ("int", 16, 1024),
        "grow_policy": ("choice", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "min_data_in_leaf": ("int", 1, 500),
        "max_leaves": ("int", 4, 64),
        "one_hot_max_size": ("int", 1, 255),
        "loss_function": ("objective",),
        "auto_class_weights": ("choice", ["Balanced", "SqrtBalanced", None]),
        "early_stopping_rounds": ("int", 20, 500),
    }
    objectives = {
        "regression": ["RMSE", "MAE"],
        "binary": ["Logloss", "CrossEntropy"],
        "multiclass": ["MultiClass", "MultiClassOneVsAll"],
    }
    param_aliases = {
        "n_estimators": "iterations", "num_boost_round": "iterations", "num_trees": "iterations",
        "eta": "learning_rate", "lr": "learning_rate",
        "max_depth": "depth",
        "reg_lambda": "l2_leaf_reg", "lambda": "l2_leaf_reg", "lambda_l2": "l2_leaf_reg", "reg_l2": "l2_leaf_reg",
        "colsample_bytree": "rsm", "colsample_bylevel": "rsm", "feature_fraction": "rsm",
        "bagging_fraction": "subsample",
        "min_child_samples": "min_data_in_leaf", "num_leaves": "max_leaves",
        "max_bin": "border_count", "objective": "loss_function", "loss": "loss_function",
        "class_weight": "auto_class_weights",
        "early_stopping": "early_stopping_rounds",
    }
    clf_only = {"auto_class_weights"}
    doc_defaults = {"random_strength": 1.0, "border_count": 254, "grow_policy": "SymmetricTree",
                    "one_hot_max_size": 2,
                    "loss_function": {"regression": "RMSE", "binary": "Logloss", "multiclass": "MultiClass"}}

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
