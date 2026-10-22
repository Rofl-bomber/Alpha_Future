"""CatBoost: обёртка с единым интерфейсом. Категории передаются нативно (как строки)."""

import time

import numpy as np
import pandas as pd

import task_metrics

FAMILY = "catboost"

DEFAULTS = {
    "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 3.0, "border_count": 254, "random_strength": 1.0,
    "bagging_temperature": 1.0, "one_hot_max_size": 4, "min_data_in_leaf": 1,
    "n_estimators": 5000, "early_stopping_rounds": 150,
}

RANGES = {
    "learning_rate": (float, 0.005, 0.3), "depth": (int, 2, 10), "l2_leaf_reg": (float, 0.1, 100.0),
    "border_count": (int, 32, 1024), "random_strength": (float, 0.0, 10.0), "bagging_temperature": (float, 0.0, 10.0),
    "one_hot_max_size": (int, 1, 255), "min_data_in_leaf": (int, 1, 500),
    "n_estimators": (int, 50, 20000), "early_stopping_rounds": (int, 20, 500),
}


def space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
    }


class _TimeLimit:
    """Останавливает обучение на дедлайне; owner.truncated = True — модель недообучена."""

    def __init__(self, deadline, owner):
        self.deadline = deadline
        self.owner = owner

    def after_iteration(self, info):
        if self.deadline and time.time() > self.deadline:
            self.owner.truncated = True
            return False
        return True


def _prep(X: pd.DataFrame):
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object]
    if not cats:
        return X, []
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str).fillna("nan")
    return X, cats


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
        self.objective = objective or "l2"
        self.deadline = deadline
        self.model = None
        self.best_iteration = None
        self.truncated = False

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        p = dict(self.params)
        n_est = int(p.pop("n_estimators"))
        es = int(p.pop("early_stopping_rounds"))
        if self.fast:
            p["learning_rate"] = min(0.3, p["learning_rate"] * 2)
            n_est = min(n_est, 1500)
        common = dict(iterations=n_est, random_seed=self.seed, thread_count=self.n_jobs, verbose=False,
                      allow_writing_files=False, **p)
        Xp, cats = _prep(X)
        train = Pool(Xp, label=y, weight=w, cat_features=cats)
        eval_set = None
        if X_val is not None:
            Xv, _ = _prep(X_val)
            eval_set = Pool(Xv, label=y_val, cat_features=cats)
            common["early_stopping_rounds"] = es
            common["use_best_model"] = True
        if self.task_type == "regression":
            loss = {"l1": "MAE", "huber": "Huber:delta=1.0"}.get(self.objective, "RMSE")
            self.model = CatBoostRegressor(loss_function=loss, **common)
        elif self.task_type == "binary":
            self.model = CatBoostClassifier(loss_function="Logloss", **common)
        else:
            self.model = CatBoostClassifier(loss_function="MultiClass", **common)
        self.model.fit(train, eval_set=eval_set, callbacks=[_TimeLimit(self.deadline, self)])
        bi = self.model.get_best_iteration()
        self.best_iteration = int(bi) + 1 if bi is not None and X_val is not None else self.model.tree_count_
        return self

    def predict(self, X):
        Xp, _ = _prep(X)
        if self.task_type == "regression":
            return np.asarray(self.model.predict(Xp), dtype=float)
        proba = self.model.predict_proba(Xp)
        classes = [int(c) for c in self.model.classes_]
        full = task_metrics.expand_proba(proba, classes, max(self.n_classes, 2))
        return full[:, 1] if self.task_type == "binary" else full

    def importance(self) -> dict:
        names = self.model.feature_names_
        vals = self.model.get_feature_importance()
        return {n: float(v) for n, v in zip(names, vals)}
