"""Validation scheme selection, fold construction, adversarial validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from mla_utils import LOG


@dataclass
class CVPlan:
    kind: str = "kfold"  # stratified | kfold | group | time_blocks
    n_splits: int = 5
    group_col: str | None = None
    time_col: str | None = None
    reason: str = ""
    groups: np.ndarray | None = field(default=None, repr=False)
    order: np.ndarray | None = field(default=None, repr=False)

    def describe(self) -> str:
        s = f"{self.kind} ({self.n_splits} folds)"
        if self.group_col:
            s += f" grouped by {self.group_col}"
        if self.time_col:
            s += f" ordered by {self.time_col}"
        if self.reason:
            s += f" — {self.reason}"
        return s


def default_n_splits(n: int) -> int:
    if n < 1500:
        return 10
    if n > 1_500_000:
        return 3
    if n > 400_000:
        return 4
    return 5


def choose_cv(task_type: str, train: pd.DataFrame, test: pd.DataFrame, profile: dict, llm_cv: dict | None,
              exclude: set) -> CVPlan:
    n = len(train)
    k = default_n_splits(n)
    base = "stratified" if task_type in ("binary", "multiclass") else "kfold"
    llm_cv = llm_cv or {}

    def group_ok(col: str) -> tuple[bool, float, float]:
        if col not in train.columns or col not in test.columns or col in exclude:
            return False, 1.0, 1.0
        tr = train[col]
        nun = tr.nunique(dropna=True)
        if nun < 20:
            return False, 1.0, 1.0
        mean_size = tr.notna().sum() / max(1, nun)
        frac = test[col].dropna().isin(set(tr.dropna().unique())).mean() if test[col].notna().any() else 1.0
        return (mean_size >= 2.0 and frac < 0.1), float(frac), float(mean_size)

    # 1) LLM-suggested time split
    tcol = llm_cv.get("time_column") if isinstance(llm_cv, dict) else None
    ttype = str(llm_cv.get("type", "")).lower() if isinstance(llm_cv, dict) else ""
    cands_time = []
    if tcol and tcol in train.columns and tcol in test.columns:
        cands_time.append(tcol)
    for c, p in profile.items():
        if p.get("kind") == "datetime" and c in test.columns and c not in cands_time:
            cands_time.append(c)
    for c in cands_time:
        try:
            tr = train[c]
            te = test[c]
            if not pd.api.types.is_numeric_dtype(tr):
                tr = pd.to_datetime(tr, errors="coerce")
                te = pd.to_datetime(te, errors="coerce")
            if tr.notna().mean() < 0.9 or te.notna().mean() < 0.9:
                continue
            q = tr.quantile(0.97)
            frac_after = (te >= q).mean()
            if frac_after > 0.9 and (ttype in ("time", "temporal", "timeseries", "time_series") or c != tcol):
                vals = tr.values
                order = np.argsort(pd.Series(vals).rank(method="first").values)
                nun = pd.Series(vals).nunique()
                if nun <= 20:
                    groups = pd.Series(vals).astype(str).values
                    return CVPlan("group", min(k, nun), c, c, f"test is later in time than train ({c}); hold out whole periods", groups=groups)
                return CVPlan("time_blocks", k, None, c, f"test is later in time than train ({c})", order=order)
        except Exception as e:
            LOG.debug(f"time check failed for {c}: {e}")

    # 2) group split (test entities unseen in train)
    gcands = []
    gcol = llm_cv.get("group_column") if isinstance(llm_cv, dict) else None
    if gcol:
        ok, frac, ms = group_ok(gcol)
        if ok or (frac < 0.3 and ms >= 2.0 and ttype == "group"):
            gcands.append((frac, gcol))
    for c, p in profile.items():
        if c in exclude or p.get("kind") not in ("categorical", "numeric"):
            continue
        if p.get("kind") == "numeric" and not p.get("is_integer", False):
            continue
        ok, frac, ms = group_ok(c)
        if ok:
            gcands.append((frac, c))
    if gcands:
        gcands.sort()
        frac, c = gcands[0]
        groups = train[c].astype(str).fillna("__nan__").values
        nun = len(np.unique(groups))
        return CVPlan("group", min(k, nun), c, None, f"test rows belong to unseen '{c}' values ({100 * frac:.1f}% overlap)", groups=groups)
    return CVPlan(base, k, None, None, "random split (train/test look i.i.d.)")


def make_folds(plan: CVPlan, y: np.ndarray, task_type: str, seed: int, n_splits: int | None = None) -> list:
    n = len(y)
    k = n_splits or plan.n_splits
    idx = np.arange(n)
    if plan.kind == "group" and plan.groups is not None:
        k = max(2, min(k, len(np.unique(plan.groups))))
        if task_type in ("binary", "multiclass"):
            try:
                sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                return [(tr, va) for tr, va in sgk.split(idx, y, plan.groups)]
            except Exception:
                pass
        # shuffle group order deterministically via seed
        ug = np.unique(plan.groups)
        rng = np.random.RandomState(seed)
        perm = {g: i for i, g in enumerate(rng.permutation(ug))}
        g2 = np.array([perm[g] for g in plan.groups])
        return [(tr, va) for tr, va in GroupKFold(n_splits=k).split(idx, y, g2)]
    if plan.kind == "time_blocks" and plan.order is not None:
        blocks = np.array_split(plan.order, k)
        folds = []
        for b in blocks:
            mask = np.zeros(n, bool)
            mask[b] = True
            folds.append((idx[~mask], idx[mask]))
        return folds
    if task_type in ("binary", "multiclass"):
        counts = np.bincount(y.astype(int))
        counts = counts[counts > 0]
        if counts.min() >= 2:
            skf = StratifiedKFold(n_splits=min(k, max(2, int(counts.max()))), shuffle=True, random_state=seed)
            return [(tr, va) for tr, va in skf.split(idx, y)]
    if task_type == "regression":
        try:
            bins = pd.qcut(pd.Series(y).rank(method="first"), q=min(10, max(2, n // (k * 5))), labels=False)
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            return [(tr, va) for tr, va in skf.split(idx, bins)]
        except Exception:
            pass
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(idx)]


def adversarial_validation(X: pd.DataFrame, n_train: int, n_jobs: int, max_rows: int = 120_000, seed: int = 0) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    n = len(X)
    y = np.r_[np.zeros(n_train), np.ones(n - n_train)]
    idx = np.arange(n)
    if n > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_rows, replace=False)
    Xs, ys = X.iloc[idx], y[idx]
    oof = np.zeros(len(idx))
    imp = np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=n_jobs, seed=seed, max_cat_to_onehot=8)
    for tr, va in skf.split(Xs, ys):
        m = lgb.train(params, lgb.Dataset(Xs.iloc[tr], ys[tr]), num_boost_round=150)
        oof[va] = m.predict(Xs.iloc[va])
        imp += m.feature_importance("gain")
    auc = float(roc_auc_score(ys, oof))
    order = np.argsort(-imp)
    tot = imp.sum() + 1e-12
    top = [(X.columns[i], float(imp[i] / tot)) for i in order[:10] if imp[i] > 0]
    return {"auc": auc, "top_features": top}
