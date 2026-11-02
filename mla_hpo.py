"""Optuna hyper-parameter search for gradient boosting models (time-boxed, fold-pruned)."""
from __future__ import annotations

import time

import numpy as np

import mla_metrics as MT
from mla_models import ModelSpec, run_cv
from mla_postprocess import fit_decision
from mla_utils import LOG


def _space(trial, family: str, has_cats: bool, n_rows: int, task: str):
    if family == "lgbm":
        max_leaves = 256 if n_rows > 20000 else 64
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 8, max_leaves, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 30.0, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 9, 12]),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 10.0, log=True),
        }
        if has_cats:
            p["cat_smooth"] = trial.suggest_float("cat_smooth", 1.0, 100.0, log=True)
            p["min_data_per_group"] = trial.suggest_int("min_data_per_group", 5, 200, log=True)
            p["max_cat_threshold"] = trial.suggest_int("max_cat_threshold", 8, 128, log=True)
        return p
    if family == "xgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 11),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 2.0, log=True),
        }
    if family == "cat":
        return {
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    raise ValueError(family)


DEFAULT_STARTS = {
    "lgbm": [
        {"num_leaves": 63, "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l1": 1e-8,
         "lambda_l2": 1.0, "max_depth": -1, "min_sum_hessian_in_leaf": 1e-3},
        {"num_leaves": 31, "min_child_samples": 50, "feature_fraction": 0.5, "bagging_fraction": 0.8, "lambda_l1": 0.1,
         "lambda_l2": 5.0, "max_depth": -1, "min_sum_hessian_in_leaf": 1e-3},
        {"num_leaves": 127, "min_child_samples": 10, "feature_fraction": 0.7, "bagging_fraction": 0.9, "lambda_l1": 1e-8,
         "lambda_l2": 0.1, "max_depth": -1, "min_sum_hessian_in_leaf": 1e-3},
    ],
    "xgb": [{"max_depth": 7, "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0,
             "reg_alpha": 1e-8, "gamma": 1e-8}],
    "cat": [{"depth": 7, "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5}],
}


def tune(family: str, problem, state, folds, fold_ids: list, n_jobs: int, budget_s: float, base_params: dict,
         seed: int, fold_tag: str, max_trials: int = 200, extra_spec: dict | None = None) -> tuple[dict, list]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_end = time.time() + budget_s
    has_cats = bool(state.cat_cols)
    n_rows = state.n_train
    extra_spec = extra_spec or {}
    history = []
    failures = [0]

    def objective(trial):
        p = dict(base_params)
        p.update(_space(trial, family, has_cats, n_rows, problem.task))
        if family == "lgbm" and p.get("max_depth", -1) > 0:
            p["num_leaves"] = int(min(p["num_leaves"], 2 ** p["max_depth"]))
        spec = ModelSpec(family, p, seed=seed, name=f"{family}_hpo", **extra_spec)
        oof_parts = {}
        res_all = None
        for step, fi in enumerate(fold_ids):
            if time.time() > t_end:
                raise optuna.TrialPruned()
            r = run_cv(spec, problem, state, folds, n_jobs, fold_tag, predict_test=False, deadline=t_end, fold_subset=[fi])
            if r is None:
                failures[0] += 1
                if failures[0] >= 3 and not history:
                    study.stop()  # the model cannot be trained at all (missing library, bad params): stop wasting trials
                raise optuna.TrialPruned()
            oof_parts[fi] = r
            trial.report(MT.utility(problem.metric, r.score), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        # combine OOF of the evaluated folds
        idx = np.concatenate([folds[fi][1] for fi in fold_ids])
        oof = np.concatenate([oof_parts[fi].oof[folds[fi][1]] for fi in fold_ids])
        y = problem.y[idx]
        dec = fit_decision(problem.metric, problem.task, y, oof, problem.n_classes, validate=False)
        from mla_postprocess import metric_with_decision
        val = metric_with_decision(problem.metric, y, oof, dec, problem.n_classes)
        u = MT.utility(problem.metric, val)
        iters = [int(np.mean(oof_parts[fi].best_iters)) for fi in fold_ids]
        trial.set_user_attr("iters", iters)
        history.append({"params": {k: v for k, v in p.items()}, "utility": u, "iters": iters})
        return u

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=6, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    for st in DEFAULT_STARTS.get(family, []):
        try:
            study.enqueue_trial({k: v for k, v in st.items() if k in _space_keys(family, has_cats)})
        except Exception:
            pass
    remaining = t_end - time.time()
    if remaining > 5:
        study.optimize(objective, n_trials=max_trials, timeout=remaining, catch=(Exception,), gc_after_trial=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    LOG.info(f"HPO[{family}]: {len(done)} complete / {len(study.trials)} trials")
    if not done:
        return dict(base_params), history
    best = study.best_trial
    p = dict(base_params)
    p.update(best.params)
    if family == "lgbm" and p.get("max_depth", -1) > 0:
        p["num_leaves"] = int(min(p["num_leaves"], 2 ** p["max_depth"]))
    LOG.info(f"HPO[{family}] best utility {best.value:.6f} params {best.params}")
    return p, history


def _space_keys(family, has_cats):
    """Names of the parameters sampled for this family (used to filter enqueued start points)."""
    keys = []

    class Rec:
        def __getattr__(self, name):
            def f(key, *a, **k):
                keys.append(key)
                if name == "suggest_categorical":
                    return a[0][0]
                if name == "suggest_int":
                    return a[0]
                return a[0]
            return f
    _space(Rec(), family, has_cats, 100000, "regression")
    return set(keys)

