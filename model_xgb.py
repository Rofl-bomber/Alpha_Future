"""XGBoost: обёртка с единым интерфейсом. Категории — нативно (enable_categorical, hist)."""

import time

import numpy as np

FAMILY = "xgb"

DEFAULTS = {
    "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0, "max_bin": 256, "max_cat_to_onehot": 4,
    "n_estimators": 5000, "early_stopping_rounds": 100,
}

RANGES = {
    "learning_rate": (float, 0.005, 0.3), "max_depth": (int, 2, 14), "min_child_weight": (float, 0.0, 100.0),
    "subsample": (float, 0.3, 1.0), "colsample_bytree": (float, 0.2, 1.0), "reg_lambda": (float, 0.0, 100.0),
    "reg_alpha": (float, 0.0, 100.0), "gamma": (float, 0.0, 10.0), "max_bin": (int, 32, 1024),
    "max_cat_to_onehot": (int, 1, 64), "n_estimators": (int, 50, 20000), "early_stopping_rounds": (int, 20, 500),
}


def space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _time_callback(deadline):
    import xgboost as xgb

    class _TimeLimit(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            return bool(deadline and time.time() > deadline)

    return _TimeLimit()


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

    def fit(self, X, y, w=None, X_val=None, y_val=None):
        import xgboost as xgb

        p = {k: v for k, v in self.params.items() if k not in ("n_estimators", "early_stopping_rounds")}
        n_est = int(self.params["n_estimators"])
        if self.fast:
            p["learning_rate"] = min(0.3, p["learning_rate"] * 2)
            n_est = min(n_est, 1500)
        if self.task_type == "regression":
            p["objective"] = {"l1": "reg:absoluteerror", "huber": "reg:pseudohubererror"}.get(
                self.objective, "reg:squarederror")
        elif self.task_type == "binary":
            p["objective"] = "binary:logistic"
        else:
            p["objective"] = "multi:softprob"
            p["num_class"] = self.n_classes
        p.update(tree_method="hist", nthread=self.n_jobs, seed=self.seed, verbosity=0)
        dtrain = xgb.DMatrix(X, label=y, weight=w, enable_categorical=True)
        evals = []
        es = None
        if X_val is not None:
            evals = [(xgb.DMatrix(X_val, label=y_val, enable_categorical=True), "val")]
            es = int(self.params["early_stopping_rounds"])
        self.booster = xgb.train(p, dtrain, num_boost_round=n_est, evals=evals, early_stopping_rounds=es,
                                 verbose_eval=False, callbacks=[_time_callback(self.deadline)])
        if X_val is not None and getattr(self.booster, "best_iteration", None) is not None:
            self.best_iteration = int(self.booster.best_iteration) + 1
        else:
            self.best_iteration = int(self.booster.num_boosted_rounds())
        return self

    def predict(self, X):
        import xgboost as xgb

        d = xgb.DMatrix(X, enable_categorical=True)
        p = self.booster.predict(d, iteration_range=(0, self.best_iteration))
        return np.asarray(p, dtype=float)

    def importance(self) -> dict:
        return {k: float(v) for k, v in self.booster.get_score(importance_type="gain").items()}
