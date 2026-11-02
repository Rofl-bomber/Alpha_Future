"""AutoGluon-Tabular as one (diverse) ensemble member. Uses our engineered features, its own bagged models,
returns out-of-fold and test predictions in the internal format. Fully optional: any failure -> None."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mla_metrics as MT
from mla_features import FeatureState
from mla_models import CVResult, ModelSpec, Problem
from mla_utils import LOG, format_exc

AG_METRIC = {
    "rmse": "root_mean_squared_error", "mse": "mean_squared_error", "mae": "mean_absolute_error", "r2": "r2",
    "medae": "median_absolute_error", "mape": "mean_absolute_percentage_error", "rmsle": "root_mean_squared_error",
    "msle": "root_mean_squared_error", "smape": "mean_absolute_error",
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1", "f1_macro": "f1_macro",
    "f1_micro": "f1_micro", "f1_weighted": "f1_weighted", "precision": "precision", "recall": "recall",
    "precision_macro": "precision_macro", "recall_macro": "recall_macro", "mcc": "mcc", "kappa": "log_loss",
    "qwk": "quadratic_kappa", "roc_auc": "roc_auc", "roc_auc_ovo": "roc_auc_ovo_macro", "gini": "roc_auc",
    "pr_auc": "average_precision", "logloss": "log_loss", "brier": "log_loss",
}


def ag_available() -> bool:
    try:
        import autogluon.tabular  # noqa: F401
        return True
    except Exception:
        return False


def run_autogluon(problem: Problem, state: FeatureState, n_jobs: int, time_limit: float, work_dir: Path, seed: int = 0) -> CVResult | None:
    t0 = time.time()
    path = Path(work_dir) / "autogluon"
    try:
        from autogluon.tabular import TabularPredictor
        shutil.rmtree(path, ignore_errors=True)
        X = state.X
        n_tr = state.n_train
        label = "__target__"
        tr = X.iloc[:n_tr].copy()
        te = X.iloc[n_tr:].reset_index(drop=True)
        task = problem.task
        # target-encoded keys are fold-dependent in our CV; give AutoGluon their out-of-fold version
        # (train rows: inner-fold OOF encodings, test rows: statistics of the whole training set)
        if state.te_specs:
            from mla_features import key_codes, te_for_fold, te_names
            all_tr = np.arange(n_tr)
            test_idx = np.arange(n_tr, len(X))
            for spec_cols in state.te_specs:
                codes = key_codes(X, spec_cols)
                (etr, ctr), outs = te_for_fold(codes, problem.y, all_tr, [test_idx], task, problem.n_classes, seed=seed)
                names = te_names(spec_cols, task, problem.n_classes)
                for j, nm in enumerate(names[:-1]):
                    tr[nm] = etr[:, j]
                    te[nm] = outs[0][0][:, j]
                tr[names[-1]] = ctr
                te[names[-1]] = outs[0][1]
        if task == "regression":
            y = problem.y.astype(float)
            if problem.metric in ("rmsle", "msle"):
                y = np.log1p(np.clip(y, 0, None))
            tr[label] = y
        else:
            tr[label] = problem.y.astype(int)
        metric = AG_METRIC.get(problem.metric, "log_loss" if task != "regression" else "root_mean_squared_error")
        if task == "multiclass" and metric in ("f1", "precision", "recall", "roc_auc", "average_precision"):
            metric = "log_loss"
        # label metrics: optimise log loss (our decision layer tunes thresholds/weights afterwards)
        if task != "regression" and MT.get(problem.metric).needs == "label":
            metric = "log_loss"
        pred = TabularPredictor(label=label, problem_type=task, eval_metric=metric, path=str(path), verbosity=0)
        fit_limit = max(30.0, time_limit * 0.85)
        pred.fit(tr, time_limit=fit_limit, presets="good_quality", num_bag_folds=5, num_bag_sets=1, num_stack_levels=0,
                 dynamic_stacking=False, num_cpus=n_jobs, num_gpus=0, fit_weighted_ensemble=True,
                 excluded_model_types=["KNN", "FASTAI"], ag_args_fit={"num_cpus": n_jobs}, refit_full=False,
                 save_bag_folds=True, set_best_to_refit_full=False, calibrate=False)
        best = pred.model_best
        if task == "regression":
            oof = pred.predict_oof(model=best).values.astype(float)
            test = pred.predict(te, model=best).values.astype(float)
            if problem.metric in ("rmsle", "msle"):
                oof, test = np.expm1(oof), np.expm1(test)
        else:
            po = pred.predict_proba_oof(model=best, as_multiclass=True)
            pt = pred.predict_proba(te, model=best, as_multiclass=True)
            cols = list(range(problem.n_classes))
            oof = po.reindex(columns=cols, fill_value=0.0).values.astype(float)
            test = pt.reindex(columns=cols, fill_value=0.0).values.astype(float)
            oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-12, None)
            test = test / np.clip(test.sum(1, keepdims=True), 1e-12, None)
        if not np.all(np.isfinite(oof)) or len(oof) != n_tr:
            LOG.warning("autogluon produced invalid OOF predictions")
            return None
        score = MT.safe_score(problem.metric, problem.y, oof)
        lb = pred.leaderboard(silent=True)
        info = {"best": best, "models": lb["model"].tolist()[:12]}
        LOG.info(f"[models] autogluon ({best}): {problem.metric}={score:.6g} in {time.time() - t0:.0f}s; models={info['models'][:6]}")
        spec = ModelSpec("autogluon", {}, name="autogluon")
        return CVResult("autogluon", spec, oof, test, [], [], score, time.time() - t0, [], None, True, np.ones(n_tr, bool), extra={"ag": info})
    except Exception as e:
        LOG.warning(f"autogluon failed: {type(e).__name__}: {e}")
        LOG.debug(format_exc())
        return None
    finally:
        # bagged models can be large on disk; keep the workspace light
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def fit_ag(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    """AutoGluon as a per-fold fitter inside our own CV (honest OOF: target encodings are computed on the
    fold's training part only). Bagging inside the fold's training rows, so the validation fold is never seen."""
    import tempfile
    import uuid
    from autogluon.tabular import TabularPredictor
    task = problem.task
    label = "__target__"
    tr = Xtr.copy()
    tr[label] = ytr
    metric = AG_METRIC.get(problem.metric, "log_loss" if task != "regression" else "root_mean_squared_error")
    if task != "regression" and (MT.get(problem.metric).needs == "label" or (task == "multiclass" and metric in ("f1", "roc_auc", "average_precision"))):
        metric = "log_loss"
    if task == "regression" and spec.target_transform == "log1p":
        metric = "root_mean_squared_error"
    limit = max(20.0, (deadline - time.time()) * 0.6) if deadline else 300.0  # rest: bagged predictions + overhead
    path = Path(spec.params.get("work_dir", tempfile.gettempdir())) / f"ag_{uuid.uuid4().hex[:8]}"
    try:
        pred = TabularPredictor(label=label, problem_type=task, eval_metric=metric, path=str(path), verbosity=0)
        pred.fit(tr, time_limit=limit, presets="good_quality", num_bag_folds=int(spec.params.get("bag_folds", 3)), num_bag_sets=1,
                 num_stack_levels=0, dynamic_stacking=False, num_cpus=n_jobs, num_gpus=0, fit_weighted_ensemble=True,
                 excluded_model_types=["KNN", "FASTAI"], refit_full=False, save_bag_folds=True,
                 set_best_to_refit_full=False, calibrate=False)
        if task == "regression":
            pva = pred.predict(Xva).values.astype(float)
            pte = pred.predict(Xte).values.astype(float) if Xte is not None else None
        else:
            cols = list(range(problem.n_classes))
            pva = pred.predict_proba(Xva, as_multiclass=True).reindex(columns=cols, fill_value=0.0).values.astype(float)
            pte = pred.predict_proba(Xte, as_multiclass=True).reindex(columns=cols, fill_value=0.0).values.astype(float) if Xte is not None else None
            if task == "binary":
                pva = pva[:, 1]
                pte = pte[:, 1] if pte is not None else None
        return pva, pte, 1, None
    finally:
        shutil.rmtree(path, ignore_errors=True)
