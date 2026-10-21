from __future__ import annotations

import lightgbm as lgb
import numpy as np

from .base import ModelBase


class LightGBMModel(ModelBase):
    name = "lightgbm"
    defaults = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    early_stopping_rounds=100)
    param_schema = {
        "learning_rate": ("float", 0.005, 0.3),
        "n_estimators": ("int", 50, 10000),
        "num_leaves": ("int", 8, 512),
        "max_depth": ("int", -1, 16),
        "min_child_samples": ("int", 1, 500),
        "min_child_weight": ("float", 0.0, 100.0),
        "subsample": ("float", 0.3, 1.0),
        "subsample_freq": ("int", 0, 10),
        "colsample_bytree": ("float", 0.2, 1.0),
        "reg_lambda": ("float", 0.0, 100.0),
        "reg_alpha": ("float", 0.0, 100.0),
        "min_split_gain": ("float", 0.0, 10.0),
        "max_bin": ("int", 15, 1023),
        "cat_smooth": ("float", 0.0, 100.0),
        "cat_l2": ("float", 0.0, 100.0),
        "objective": ("objective",),
        "alpha": ("float", 1e-6, 1e6),
        "class_weight": ("choice", ["balanced", None]),
        "early_stopping_rounds": ("int", 20, 500),
    }
    objectives = {
        "regression": ["regression", "regression_l1", "huber", "fair"],
        "binary": ["binary"],
        "multiclass": ["multiclass", "multiclassova"],
    }
    param_aliases = {
        "eta": "learning_rate", "lr": "learning_rate", "shrinkage_rate": "learning_rate",
        "num_iterations": "n_estimators", "num_boost_round": "n_estimators", "iterations": "n_estimators",
        "n_iter": "n_estimators", "num_trees": "n_estimators",
        "max_leaves": "num_leaves", "num_leaf": "num_leaves",
        "min_data_in_leaf": "min_child_samples", "min_data": "min_child_samples",
        "min_sum_hessian_in_leaf": "min_child_weight",
        "bagging_fraction": "subsample", "bagging_freq": "subsample_freq",
        "feature_fraction": "colsample_bytree", "colsample": "colsample_bytree",
        "lambda_l2": "reg_lambda", "l2_leaf_reg": "reg_lambda", "reg_l2": "reg_lambda", "lambda": "reg_lambda",
        "lambda_l1": "reg_alpha", "reg_l1": "reg_alpha",
        "min_gain_to_split": "min_split_gain", "gamma": "min_split_gain",
        "huber_delta": "alpha", "delta": "alpha",
        "max_bins": "max_bin",
        "loss": "objective", "loss_function": "objective",
        "early_stopping": "early_stopping_rounds",
    }
    clf_only = {"class_weight"}
    reg_only = {"alpha"}
    doc_defaults = {"max_depth": -1, "min_child_weight": 0.001, "reg_alpha": 0.0, "min_split_gain": 0.0,
                    "max_bin": 255, "alpha": 0.9,
                    "objective": {"regression": "regression", "binary": "binary", "multiclass": "multiclass"}}

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
