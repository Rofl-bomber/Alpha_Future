from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .base import ModelBase, to_numeric_frame


class ExtraTreesModel(ModelBase):
    name = "extratrees"
    defaults = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5)
    uses_eval = False
    param_schema = {
        "n_estimators": ("int", 50, 2000),
        "min_samples_leaf": ("int", 1, 200),
        "min_samples_split": ("int", 2, 200),
        "max_features": ("float", 0.05, 1.0),
        "max_depth": ("int", 1, 64),
        "bootstrap": ("bool",),
        "class_weight": ("choice", ["balanced", "balanced_subsample", None]),
    }
    param_aliases = {
        "n_trees": "n_estimators", "iterations": "n_estimators", "num_trees": "n_estimators",
        "min_child_samples": "min_samples_leaf", "min_data_in_leaf": "min_samples_leaf",
        "colsample_bytree": "max_features", "feature_fraction": "max_features",
        "depth": "max_depth",
    }
    clf_only = {"class_weight"}
    doc_defaults = {"min_samples_split": 2, "bootstrap": False}

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        self.cat_cols = list(cat_cols)
        p = dict(self.params)
        common = dict(n_jobs=self.n_threads, random_state=self.seed)
        cls = ExtraTreesClassifier if self.is_clf else ExtraTreesRegressor
        self.model = cls(**p, **common)
        Z = to_numeric_frame(X_tr, self.cat_cols, fill=-999.0)
        self.model.fit(Z.values, y_tr)
        self.importances = dict(zip(X_tr.columns, map(float, self.model.feature_importances_)))
        return self

    def predict(self, X):
        Z = to_numeric_frame(X, self.cat_cols, fill=-999.0).values
        if self.is_clf:
            return self.expand_proba(self.model.predict_proba(Z), self.model.classes_)
        return np.asarray(self.model.predict(Z), dtype=float)
