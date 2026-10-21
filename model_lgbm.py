"""LightGBM: обёртка с единым интерфейсом fit / predict / importance.

Предсказания во внутреннем представлении (см. task_metrics): регрессия — значения,
binary — вероятность класса 1, multiclass — (n, K).
"""

import time

import numpy as np

FAMILY = "lgbm"

DEFAULTS = {
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 0.0, "lambda_l2": 1.0, "max_depth": -1,
    "max_bin": 255, "min_split_gain": 0.0, "cat_smooth": 10.0, "cat_l2": 10.0, "max_cat_to_onehot": 4,
    "min_data_per_group": 50, "extra_trees": False, "path_smooth": 0.0,
    "n_estimators": 5000, "early_stopping_rounds": 100,
}

RANGES = {
    "learning_rate": (float, 0.005, 0.3), "num_leaves": (int, 7, 1023), "min_child_samples": (int, 2, 500),
    "feature_fraction": (float, 0.2, 1.0), "bagging_fraction": (float, 0.3, 1.0), "bagging_freq": (int, 0, 10),
    "lambda_l1": (float, 0.0, 100.0), "lambda_l2": (float, 0.0, 100.0), "max_depth": (int, -1, 16),
    "max_bin": (int, 31, 1023), "min_split_gain": (float, 0.0, 5.0), "cat_smooth": (float, 0.0, 100.0),
    "cat_l2": (float, 0.0, 100.0), "max_cat_to_onehot": (int, 1, 64), "min_data_per_group": (int, 1, 500),
    "extra_trees": (bool, 0, 1), "path_smooth": (float, 0.0, 100.0),
    "n_estimators": (int, 50, 20000), "early_stopping_rounds": (int, 20, 500),
}


def space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


class _TimeLimit:
    def __init__(self, deadline):
        self.deadline = deadline
        self.order = 40

    def __call__(self, env):
        import lightgbm as lgb

        if self.deadline and time.time() > self.deadline:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)


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
        self.booster = None
        self.best_iteration = None

    def _lgb_params(self):
        p = {k: v for k, v in self.params.items() if k not in ("n_estimators", "early_stopping_rounds")}
        if self.fast:
            p["learning_rate"] = min(0.3, p["learning_rate"] * 2)
        if self.task_type == "regression":
            p["objective"] = {"l1": "regression_l1", "huber": "huber"}.get(self.objective, "regression")
        elif self.task_type == "binary":
            p["objective"] = "binary"
        else:
            p["objective"] = "multiclass"
            p["num_class"] = self.n_classes
        p.update(seed=self.seed, num_threads=self.n_jobs, verbose=-1, feature_pre_filter=False)
        return p

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        import lightgbm as lgb

        n_est = int(self.params["n_estimators"])
        if self.fast:
            n_est = min(n_est, 1500)
        train = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
        valid_sets, callbacks = [], [_TimeLimit(self.deadline)]
        if X_val is not None:
            valid = lgb.Dataset(X_val, label=y_val, reference=train, free_raw_data=False)
            valid_sets = [valid]
            callbacks.append(lgb.early_stopping(int(self.params["early_stopping_rounds"]), verbose=False))
        self.booster = lgb.train(self._lgb_params(), train, num_boost_round=n_est, valid_sets=valid_sets,
                                 callbacks=callbacks)
        bi = self.booster.best_iteration
        self.best_iteration = int(bi) if bi and bi > 0 else self.booster.current_iteration()
        return self

    def predict(self, X):
        p = self.booster.predict(X, num_iteration=self.best_iteration)
        return np.asarray(p, dtype=float)

    def importance(self) -> dict:
        names = self.booster.feature_name()
        gains = self.booster.feature_importance(importance_type="gain")
        return {n: float(g) for n, g in zip(names, gains)}
