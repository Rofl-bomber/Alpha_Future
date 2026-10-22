"""ExtraTrees: разнообразие для ансамбля. Категории → коды, пропуски → -999."""

import numpy as np

import feat_builtin
import task_metrics

FAMILY = "extratrees"

DEFAULTS = {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 2, "max_depth": None,
            "max_samples": None}
RANGES = {"n_estimators": (int, 50, 2000), "max_features": (float, 0.05, 1.0), "min_samples_leaf": (int, 1, 100),
          "max_depth": (int, 2, 64), "max_samples": (float, 0.1, 1.0)}


def space(trial) -> dict:
    return {"max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True)}


class Model:
    family = FAMILY

    def __init__(self, task_type, n_classes, params=None, seed=0, n_jobs=8, fast=False, objective="l2",
                 deadline=None):
        self.task_type = task_type
        self.n_classes = n_classes
        self.params = {**DEFAULTS, **(params or {})}
        self.seed = seed
        self.n_jobs = n_jobs
        self.fast = fast
        self.best_iteration = None
        self.truncated = False  # без пошагового обучения: при нехватке времени процесс снимается целиком

    def _X(self, X):
        return feat_builtin.to_numeric(X).fillna(-999.0).to_numpy(dtype=np.float32)

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        p = dict(self.params)
        n_est = int(p["n_estimators"]) // (2 if self.fast else 1)
        kw = dict(n_estimators=max(50, n_est), max_features=p["max_features"], min_samples_leaf=int(p["min_samples_leaf"]),
                  max_depth=p["max_depth"], n_jobs=self.n_jobs, random_state=self.seed)
        if p.get("max_samples"):
            kw.update(bootstrap=True, max_samples=float(p["max_samples"]))
        elif len(X) > 300_000:
            kw.update(bootstrap=True, max_samples=0.5)
        self.cols = list(X.columns)
        if self.task_type == "regression":
            self.model = ExtraTreesRegressor(**kw)
        else:
            self.model = ExtraTreesClassifier(**kw)
        self.model.fit(self._X(X), y, sample_weight=w)
        return self

    def predict(self, X):
        Z = self._X(X)
        if self.task_type == "regression":
            return np.asarray(self.model.predict(Z), dtype=float)
        proba = self.model.predict_proba(Z)
        full = task_metrics.expand_proba(proba, [int(c) for c in self.model.classes_], max(self.n_classes, 2))
        return full[:, 1] if self.task_type == "binary" else full

    def importance(self) -> dict:
        return {c: float(v) for c, v in zip(self.cols, self.model.feature_importances_)}
