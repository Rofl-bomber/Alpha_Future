"""Model wrappers (LightGBM / XGBoost / CatBoost / linear / ExtraTrees / MLP) and the CV runner."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import mla_metrics as MT
from mla_features import FeatureState, build_te_block
from mla_postprocess import metric_with_decision
from mla_utils import LOG, format_exc, short_hash

warnings.filterwarnings("ignore")

HIGH_CARD = 200
XGB_NATIVE_CAT = 16
CAT_NATIVE_MAX = 2000
PARALLEL_FAMILIES = {"lgbm", "cat"}  # XGBoost: OpenMP contention across Python threads (70x slower)
_NO_PARALLEL = [False]  # set by tests / env to force sequential folds  # CatBoost CTRs on key-like columns with tens of thousands of levels are extremely slow  # XGBoost: one-hot small categoricals natively, frequency-rank codes for the rest (much faster)


@dataclass
class Problem:
    name: str
    task: str  # regression | binary | multiclass
    n_classes: int
    y: np.ndarray
    metric: str
    y_range: tuple | None = None
    nonneg: bool = False

    @property
    def kind(self):
        return "regression" if self.task == "regression" else "classification"


@dataclass
class ModelSpec:
    family: str
    params: dict = field(default_factory=dict)
    target_transform: str = "none"
    class_weight: str | None = None
    name: str = ""
    seed: int = 0

    def label(self) -> str:
        return self.name or f"{self.family}"


@dataclass
class CVResult:
    name: str
    spec: ModelSpec
    oof: np.ndarray
    test: np.ndarray | None
    fold_ids: list
    fold_scores: list
    score: float
    elapsed: float
    best_iters: list
    importance: pd.Series | None
    complete: bool
    oof_mask: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# views of the feature frame for each library
# ---------------------------------------------------------------------------------------------
_VIEW_CACHE: dict = {}


def model_view(state: FeatureState, family: str) -> pd.DataFrame:
    key = (state.key(), id(state.X), state.X.shape, family)
    if key in _VIEW_CACHE:
        return _VIEW_CACHE[key]
    X = state.X
    if family in ("lgbm", "xgb"):
        limit = HIGH_CARD if family == "lgbm" else XGB_NATIVE_CAT
        high = [c for c in state.cat_cols if X[c].cat.categories.size > limit]
        if not high:
            V = X
        else:
            V = X.copy()
            for c in high:
                vc = V[c].value_counts()
                rank = {k: i for i, k in enumerate(vc.index)}
                V[c] = V[c].map(rank).astype(float)
    elif family == "cat":
        V = X.copy()
        for c in state.cat_cols:
            if X[c].cat.categories.size > CAT_NATIVE_MAX:
                vc = V[c].value_counts()
                rank = {k: i for i, k in enumerate(vc.index)}
                V[c] = V[c].map(rank).astype(float)
            else:
                V[c] = V[c].astype("object").where(V[c].notna(), "__NA__").astype(str)
    else:
        V = X
    while len(_VIEW_CACHE) >= 4:
        _VIEW_CACHE.pop(next(iter(_VIEW_CACHE)))
    _VIEW_CACHE[key] = V
    return V


def cat_columns_in_view(state: FeatureState, family: str) -> list:
    if family == "cat":
        return [c for c in state.cat_cols if state.X[c].cat.categories.size <= CAT_NATIVE_MAX]
    if family == "lgbm":
        return [c for c in state.cat_cols if state.X[c].cat.categories.size <= HIGH_CARD]
    if family == "xgb":
        return [c for c in state.cat_cols if state.X[c].cat.categories.size <= XGB_NATIVE_CAT]
    return list(state.cat_cols)


# ---------------------------------------------------------------------------------------------
def transform_target(y, how):
    if how == "log1p":
        return np.log1p(np.clip(y, -0.999999, None))
    if how == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inverse_target(p, how):
    if how == "log1p":
        return np.expm1(p)
    if how == "sqrt":
        return np.clip(p, 0, None) ** 2
    return p


def class_weights(y, K, how):
    if not how:
        return None
    cnt = np.bincount(y.astype(int), minlength=K).astype(float)
    cnt[cnt == 0] = 1
    alpha = 1.0 if how == "balanced" else 0.5
    w_class = (len(y) / (K * cnt)) ** alpha
    w = w_class[y.astype(int)]
    return w / w.mean()


def as_proba(pred, task):
    if task == "binary":
        p = np.asarray(pred, float).reshape(-1)
        return np.stack([1 - p, p], 1)
    return np.asarray(pred, float)


# ---------------------------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------------------------
LGB_BASE = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l1=0.0, lambda_l2=1.0, max_bin=255, min_data_per_group=50, cat_smooth=10.0,
                cat_l2=10.0, max_cat_threshold=64, max_cat_to_onehot=8, verbose=-1)


def lgb_objective(problem: Problem, transform: str, params: dict) -> dict:
    p = {}
    if problem.task == "binary":
        p.update(objective="binary", metric="binary_logloss")
    elif problem.task == "multiclass":
        p.update(objective="multiclass", num_class=problem.n_classes, metric="multi_logloss")
    else:
        m = problem.metric
        obj = params.get("objective")
        if obj is None:
            if m in ("mae", "medae") and transform == "none":
                obj = "l1"
            elif m == "mape" and transform == "none":
                obj = "mape"
            else:
                obj = "l2"
        p.update(objective=obj, metric={"l1": "l1", "mape": "mape", "huber": "l2", "fair": "l2"}.get(obj, "l2"))
    return p


def fit_lgbm(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    import lightgbm as lgb
    params = dict(LGB_BASE)
    params.update({k: v for k, v in spec.params.items() if k not in ("n_estimators", "early_stopping", "objective")})
    params.update(lgb_objective(problem, spec.target_transform, spec.params))
    if spec.params.get("objective") and problem.task == "regression":
        params["objective"] = spec.params["objective"]
    params.update(n_jobs=n_jobs, seed=spec.seed, verbose=-1)
    if params.get("deterministic"):
        params["force_row_wise"] = True
    n_rounds = int(spec.params.get("n_estimators", 20000))
    es = int(spec.params.get("early_stopping", max(50, int(5 / max(params["learning_rate"], 1e-3)))))
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cats or "auto", free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cats or "auto", free_raw_data=False)
    hist = {}

    def time_cb(env):
        if deadline is not None and time.time() > deadline:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    time_cb.order = 40
    model = lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(es, verbose=False), lgb.record_evaluation(hist), time_cb])
    vals = list(hist["va"].values())[0]
    best = int(np.argmin(vals)) + 1 if vals else model.current_iteration()
    if model.best_iteration and model.best_iteration > 0:
        best = model.best_iteration
    pva = model.predict(Xva, num_iteration=best)
    pte = model.predict(Xte, num_iteration=best) if Xte is not None else None
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    return pva, pte, best, imp


# ---------------------------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------------------------
XGB_BASE = dict(eta=0.05, max_depth=7, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                reg_alpha=0.0, max_bin=256, tree_method="hist", max_cat_to_onehot=16, max_cat_threshold=32)


def fit_xgb(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    import xgboost as xgb
    params = dict(XGB_BASE)
    params.update({k: v for k, v in spec.params.items() if k not in ("n_estimators", "early_stopping")})
    if problem.task == "binary":
        params.update(objective="binary:logistic", eval_metric="logloss")
    elif problem.task == "multiclass":
        params.update(objective="multi:softprob", num_class=problem.n_classes, eval_metric="mlogloss")
    else:
        obj = spec.params.get("objective") or ("reg:absoluteerror" if problem.metric in ("mae", "medae") and spec.target_transform == "none" else "reg:squarederror")
        params.update(objective=obj, eval_metric="mae" if "absolute" in obj else "rmse")
    params.update(nthread=n_jobs, seed=spec.seed, verbosity=0)
    n_rounds = int(spec.params.get("n_estimators", 20000))
    es = int(spec.params.get("early_stopping", max(50, int(5 / max(params["eta"], 1e-3)))))
    dtr = xgb.DMatrix(Xtr, ytr, weight=wtr, enable_categorical=True, missing=np.nan)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True, missing=np.nan)

    class TimeCB(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            return deadline is not None and time.time() > deadline

    model = xgb.train(params, dtr, num_boost_round=n_rounds, evals=[(dva, "va")],
                      callbacks=[xgb.callback.EarlyStopping(rounds=es, save_best=False), TimeCB()], verbose_eval=False)
    best = int(getattr(model, "best_iteration", model.num_boosted_rounds() - 1)) + 1
    pva = model.predict(dva, iteration_range=(0, best))
    pte = model.predict(xgb.DMatrix(Xte, enable_categorical=True, missing=np.nan), iteration_range=(0, best)) if Xte is not None else None
    sc = model.get_score(importance_type="gain")
    imp = pd.Series({c: sc.get(c, 0.0) for c in Xtr.columns})
    return pva, pte, best, imp


# ---------------------------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------------------------
CAT_BASE = dict(learning_rate=0.06, depth=7, l2_leaf_reg=3.0, border_count=254, one_hot_max_size=4, random_strength=1.0,
                bootstrap_type="Bayesian", bagging_temperature=0.5)


def fit_cat(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    params = dict(CAT_BASE)
    params.update({k: v for k, v in spec.params.items() if k not in ("n_estimators", "early_stopping")})
    n_rounds = int(spec.params.get("n_estimators", 20000))
    es = int(spec.params.get("early_stopping", 200))
    params.update(iterations=n_rounds, thread_count=n_jobs, random_seed=spec.seed, verbose=0, allow_writing_files=False,
                  use_best_model=True, od_type="Iter", od_wait=es)
    if problem.task == "binary":
        params.update(loss_function="Logloss")
        M = CatBoostClassifier
    elif problem.task == "multiclass":
        params.update(loss_function="MultiClass", classes_count=problem.n_classes)
        M = CatBoostClassifier
    else:
        lf = spec.params.get("loss_function") or ("MAE" if problem.metric in ("mae", "medae") and spec.target_transform == "none" else "RMSE")
        params.update(loss_function=lf)
        M = CatBoostRegressor
    params.pop("objective", None)

    class TimeCB:
        def after_iteration(self, info):
            return not (deadline is not None and time.time() > deadline)

    model = M(**params)
    ptr = Pool(Xtr, ytr, cat_features=cats, weight=wtr)
    pva_pool = Pool(Xva, yva, cat_features=cats)
    model.fit(ptr, eval_set=pva_pool, callbacks=[TimeCB()])
    best = int(model.get_best_iteration() or model.tree_count_) + 1
    if problem.task == "regression":
        pva = model.predict(pva_pool)
        pte = model.predict(Pool(Xte, cat_features=cats)) if Xte is not None else None
    else:
        pva = model.predict_proba(pva_pool)
        pte = model.predict_proba(Pool(Xte, cat_features=cats)) if Xte is not None else None
        if problem.task == "binary":
            pva, pte = pva[:, 1], (pte[:, 1] if pte is not None else None)
    imp = pd.Series(model.get_feature_importance(), index=Xtr.columns)
    return pva, pte, best, imp


# ---------------------------------------------------------------------------------------------
# sklearn models (ExtraTrees / linear): numeric matrix with simple encodings
# ---------------------------------------------------------------------------------------------
def _numeric_matrix(Xtr, Xva, Xte, cats, scale: bool):
    from sklearn.preprocessing import QuantileTransformer
    frames = []
    for D in (Xtr, Xva, Xte):
        if D is None:
            frames.append(None)
            continue
        D = D.copy()
        for c in cats:
            if c in D.columns:
                D[c] = D[c].cat.codes.astype(float) if isinstance(D[c].dtype, pd.CategoricalDtype) else pd.to_numeric(D[c], errors="coerce")
        frames.append(D.astype(float))
    tr, va, te = frames
    med = tr.median()
    na_cols = [c for c in tr.columns if tr[c].isna().any()]
    out = []
    for D in (tr, va, te):
        if D is None:
            out.append(None)
            continue
        ind = D[na_cols].isna().astype(float).add_suffix("_isna") if na_cols else None
        D = D.fillna(med).fillna(0.0)
        if ind is not None and scale:
            D = pd.concat([D, ind], axis=1)
        out.append(D)
    if scale:
        qt = QuantileTransformer(n_quantiles=min(256, len(out[0])), output_distribution="normal", subsample=200000, random_state=0)
        cols = out[0].columns
        out = [pd.DataFrame(qt.fit_transform(out[0]), columns=cols)] + [None if D is None else pd.DataFrame(qt.transform(D[cols]), columns=cols) for D in out[1:]]
    return out


def fit_sklearn(spec, problem, Xtr, ytr, Xva, yva, Xte, wtr, n_jobs, deadline, cats):
    fam = spec.family
    tr, va, te = _numeric_matrix(Xtr, Xva, Xte, cats, scale=(fam == "linear"))
    if fam == "et":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        total = int(spec.params.get("n_estimators", 300))
        kw = dict(n_estimators=0, max_features=spec.params.get("max_features", 0.5), warm_start=True,
                  min_samples_leaf=int(spec.params.get("min_samples_leaf", 2)), n_jobs=n_jobs, random_state=spec.seed)
        model = (ExtraTreesRegressor if problem.task == "regression" else ExtraTreesClassifier)(**kw)
        # grow the forest in chunks so that the time budget is respected (sklearn has no callbacks)
        t0 = time.time()
        while model.n_estimators < total:
            model.n_estimators = min(total, model.n_estimators + 50)
            model.fit(tr, ytr, sample_weight=wtr)
            per_chunk = (time.time() - t0) / max(1, model.n_estimators / 50)
            if deadline is not None and time.time() + per_chunk > deadline:
                break
    else:
        from sklearn.linear_model import LogisticRegression, Ridge
        if problem.task == "regression":
            model = Ridge(alpha=float(spec.params.get("alpha", 3.0)))
        else:
            model = LogisticRegression(C=float(spec.params.get("C", 0.5)), max_iter=400, n_jobs=n_jobs)
    if fam != "et":
        model.fit(tr, ytr, sample_weight=wtr)
    if problem.task == "regression":
        pva = model.predict(va)
        pte = model.predict(te) if te is not None else None
    else:
        pva = model.predict_proba(va)
        pte = model.predict_proba(te) if te is not None else None
        if problem.task == "binary":
            pva, pte = pva[:, 1], (pte[:, 1] if pte is not None else None)
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=tr.columns).reindex(Xtr.columns).fillna(0)
    return pva, pte, 1, imp


FITTERS = {"lgbm": fit_lgbm, "xgb": fit_xgb, "cat": fit_cat, "et": fit_sklearn, "linear": fit_sklearn}


_AVAILABLE: dict = {}


def family_available(family: str) -> bool:
    mod = {"lgbm": "lightgbm", "xgb": "xgboost", "cat": "catboost", "mlp": "torch", "et": "sklearn", "linear": "sklearn",
           "ag": "autogluon.tabular"}.get(family)
    if mod is None:
        return True
    if mod not in _AVAILABLE:
        try:
            __import__(mod)
            _AVAILABLE[mod] = True
        except Exception:
            _AVAILABLE[mod] = False
            LOG.warning(f"library {mod} is not available; {family} models disabled")
    return _AVAILABLE[mod]


def _get_fitter(family):
    if family == "mlp":
        from mla_models_nn import fit_mlp
        return fit_mlp
    if family == "ag":
        from mla_models_ag import fit_ag
        return fit_ag
    return FITTERS[family]


# ---------------------------------------------------------------------------------------------
def run_cv(spec: ModelSpec, problem: Problem, state: FeatureState, folds: list, n_jobs: int, fold_tag: str,
           predict_test: bool = True, deadline: float | None = None, fold_subset: list | None = None,
           per_fold_budget: float | None = None) -> CVResult | None:
    """Train one model spec over folds. Returns OOF (NaN-free only on completed folds)."""
    t0 = time.time()
    fam = "lgbm" if spec.family == "lgbm" else spec.family
    view_fam = fam if fam in ("lgbm", "xgb", "cat") else ("lgbm" if fam in ("et", "linear") else "raw")
    V = model_view(state, view_fam if view_fam != "raw" else "nn")
    cats = cat_columns_in_view(state, view_fam) if view_fam in ("lgbm", "xgb", "cat") else list(state.cat_cols)
    if fam in ("et", "linear"):
        cats = [c for c in state.cat_cols if c in V.columns and isinstance(V[c].dtype, pd.CategoricalDtype)]
    n_tr = state.n_train
    y = problem.y
    K = problem.n_classes
    task = problem.task
    test_idx = np.arange(n_tr, len(V)) if predict_test else None
    Xte_base = V.iloc[n_tr:] if predict_test else None
    oof = np.full((n_tr, K), np.nan) if task == "multiclass" else np.full(n_tr, np.nan)
    test_pred = None
    fold_list = list(range(len(folds))) if fold_subset is None else list(fold_subset)
    best_iters, imps, done = [], [], []
    fit_secs, fit_iters = [], []
    fitter = _get_fitter(fam)
    yt = transform_target(y, spec.target_transform) if task == "regression" else y
    # fold-level parallelism: GBDT libraries release the GIL, and several folds with a few threads each use
    # 8 cores far better than one fold with 8 threads (LightGBM/XGBoost scale sub-linearly on ~1e5 rows)
    n_par = 1
    if fam in PARALLEL_FAMILIES and len(fold_list) >= 2 and n_jobs >= 4 and not _NO_PARALLEL[0]:
        n_par = int(min(len(fold_list), max(1, n_jobs // 3)))
    threads = max(1, n_jobs // n_par)

    def fit_fold(fi):
        tr_idx, va_idx = folds[fi]
        te_blk = build_te_block(state, y, task, K, tr_idx, va_idx, test_idx, f"{fold_tag}:{fi}:{problem.name}")
        Xtr = V.iloc[tr_idx].reset_index(drop=True)
        Xva = V.iloc[va_idx].reset_index(drop=True)
        Xte = Xte_base.reset_index(drop=True) if Xte_base is not None else None
        if te_blk is not None:
            Xtr = pd.concat([Xtr, te_blk[0]], axis=1)
            Xva = pd.concat([Xva, te_blk[1]], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte, te_blk[2]], axis=1)
        w = class_weights(y[tr_idx], K, spec.class_weight) if task != "regression" else None
        fold_deadline = deadline
        if per_fold_budget is not None:
            fold_deadline = min(deadline or 1e18, time.time() + per_fold_budget * (1.0 if n_par == 1 else 1.5))
        t_fit = time.time()
        out = fitter(spec, problem, Xtr, yt[tr_idx], Xva, yt[va_idx], Xte, w, threads, fold_deadline, cats)
        return out, time.time() - t_fit

    results = {}
    if n_par == 1:
        for fi in fold_list:
            if deadline is not None and time.time() > deadline - 2:
                break
            try:
                results[fi] = fit_fold(fi)
            except Exception as e:
                LOG.warning(f"[{spec.label()}] fold {fi} failed: {type(e).__name__}: {e}")
                LOG.debug(format_exc())
                break
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=n_par) as ex:
            futs = {}
            for fi in fold_list:
                futs[ex.submit(lambda f=fi: None if (deadline is not None and time.time() > deadline - 2) else fit_fold(f))] = fi
            for fut in cf.as_completed(futs):
                fi = futs[fut]
                try:
                    r_ = fut.result()
                    if r_ is not None:
                        results[fi] = r_
                except Exception as e:
                    LOG.warning(f"[{spec.label()}] fold {fi} failed: {type(e).__name__}: {e}")
                    LOG.debug(format_exc())
    for fi in fold_list:
        if fi not in results:
            continue
        (pva, pte, best, imp), secs = results[fi]
        tr_idx, va_idx = folds[fi]
        fit_secs.append(secs)
        fit_iters.append(max(1, int(best)))
        if task == "regression":
            pva = inverse_target(pva, spec.target_transform)
            pte = inverse_target(pte, spec.target_transform) if pte is not None else None
            oof[va_idx] = pva
        elif task == "binary":
            oof[va_idx] = np.asarray(pva).reshape(-1)
        else:
            oof[va_idx] = pva
        if pte is not None:
            pte = np.asarray(pte, float)
            test_pred = pte if test_pred is None else test_pred + pte
        best_iters.append(best)
        if imp is not None:
            imps.append(imp)
        done.append(fi)
    if not done:
        return None
    if test_pred is not None:
        test_pred = test_pred / len(done)
    mask = np.zeros(n_tr, bool)
    for fi in done:
        mask[folds[fi][1]] = True
    pred_oof = as_proba(oof, task) if task != "regression" else oof
    score = MT.safe_score(problem.metric, y[mask], pred_oof[mask])
    fold_scores = []
    for fi in done:
        va = folds[fi][1]
        fold_scores.append(MT.safe_score(problem.metric, y[va], pred_oof[va]))
    importance = None
    if imps:
        importance = pd.concat(imps, axis=1).fillna(0).mean(1)
    res = CVResult(spec.label(), spec, pred_oof, (as_proba(test_pred, task) if (test_pred is not None and task != "regression") else test_pred),
                   done, fold_scores, score, time.time() - t0, best_iters, importance, len(done) == len(fold_list), mask,
                   extra={"fit_secs": fit_secs, "fit_iters": fit_iters})
    return res


def oof_fold_utilities(problem: Problem, oof, folds, fold_ids, decision) -> np.ndarray:
    """Per-fold utility (higher=better) of OOF predictions under a decision rule."""
    out = []
    for fi in fold_ids:
        va = folds[fi][1]
        v = metric_with_decision(problem.metric, problem.y[va], oof[va], decision, problem.n_classes)
        out.append(MT.utility(problem.metric, v))
    return np.asarray(out, float)
