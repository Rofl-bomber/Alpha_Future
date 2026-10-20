"""Builtin-операции над признаками.

Всё, что не использует таргет, считается один раз на train+test (build_feature_frame).
OOF target encoding использует таргет и поэтому считается только внутри фолдов
харнессом (target_encode), строго out-of-fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

RARE = "__rare__"


@dataclass
class FeatureFrame:
    X: pd.DataFrame                 # train+test, индекс 0..n_all-1
    cat_cols: list[str]             # категориальные колонки (dtype category), которые видит модель
    te_cols: list[str] = field(default_factory=list)  # категории для OOF target encoding
    te_keep_original: bool = True   # False -> исходная категория заменяется TE
    custom_cols: dict[str, list[str]] = field(default_factory=dict)


def _top_by_importance(cols: list[str], importances: dict[str, float] | None, k: int) -> list[str]:
    if not importances:
        return cols[:k]
    return sorted(cols, key=lambda c: -importances.get(c, 0.0))[:k]


def build_feature_frame(config: dict, spec, data, registry=None, importances: dict | None = None) -> FeatureFrame:
    pre, feats = config["preprocess"], config["features"]
    X = data.X.copy()
    n_tr = data.n_train
    cat_cols = [c for c in spec.cat_cols if c in X.columns]
    num_cols = [c for c in spec.num_cols if c in X.columns]

    # --- пользовательские признаки fe_XXX
    custom_cols: dict[str, list[str]] = {}
    for fe in feats["custom"]:
        if registry is None or fe not in registry:
            raise KeyError(f"признак {fe} не зарегистрирован")
        block = registry.frame(fe)
        custom_cols[fe] = list(block.columns)
        for c in block.columns:
            X[c] = block[c].values
            if block[c].dtype == object or isinstance(block[c].dtype, pd.CategoricalDtype):
                cat_cols.append(c)
            else:
                num_cols.append(c)

    # --- drop
    drop = [c for c in feats["drop"] if c in X.columns]
    for fe in feats["drop"]:
        if fe in custom_cols:
            drop += custom_cols[fe]
    if drop:
        X = X.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]
        num_cols = [c for c in num_cols if c not in drop]

    # --- группировка редких категорий (частоты по train)
    k = int(pre["rare_min_count"])
    if k > 0:
        for c in cat_cols:
            vc = X[c].iloc[:n_tr].value_counts()
            rare = vc.index[vc < k]
            if len(rare):
                X[c] = X[c].where(~X[c].isin(rare) | X[c].isna(), RARE)

    # --- флаги пропусков
    if pre["missing_flags"]:
        for c in list(cat_cols) + list(num_cols):
            m = X[c].isna()
            if m.any() and not m.all():
                X[f"isna__{c}"] = m.astype("int8")

    # --- частотное кодирование (train+test)
    if "freq_enc" in feats["builtin"]:
        for c in cat_cols:
            X[f"freq__{c}"] = X[c].map(X[c].value_counts(dropna=False) / len(X)).astype(float)

    # --- агрегаты по группам
    if "group_stats" in feats["builtin"]:
        by = [c for c in feats["group_by"] if c in cat_cols]
        if not by:
            cand = [c for c in cat_cols if 2 <= X[c].nunique() <= 1000]
            by = _top_by_importance(cand, importances, 3)
        cols = [c for c in feats["group_cols"] if c in num_cols]
        if not cols:
            cols = _top_by_importance([c for c in num_cols if c in data.X.columns], importances, 5)
        for g in by:
            key = X[g].astype(object).where(X[g].notna(), "__nan__")
            for c in cols:
                grp = X[c].groupby(key)
                mean = key.map(grp.mean())
                std = key.map(grp.std())
                X[f"gs_mean__{c}__by__{g}"] = mean.astype(float)
                X[f"gs_std__{c}__by__{g}"] = std.astype(float)
                X[f"gs_diff__{c}__by__{g}"] = (X[c] - mean).astype(float)

    # --- кодирование категорий
    te_cols: list[str] = []
    te_keep = True
    enc = pre["cat_encoding"]
    if "target_enc" in feats["builtin"]:
        te_cols = list(cat_cols)
    if enc == "target":
        te_cols = list(cat_cols)
        te_keep = False
    if enc in ("ordinal", "freq"):
        for c in cat_cols:
            if enc == "ordinal":
                codes = pd.Categorical(X[c]).codes.astype(float)
                codes[codes < 0] = np.nan
                X[c] = codes
            else:
                X[c] = X[c].map(X[c].value_counts(dropna=False) / len(X)).astype(float)
        model_cats: list[str] = []
    else:
        model_cats = list(cat_cols)
    for c in model_cats:
        X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None))

    return FeatureFrame(X=X, cat_cols=model_cats, te_cols=te_cols, te_keep_original=te_keep,
                        custom_cols=custom_cols)


# ---------------------------------------------------------------- OOF target encoding

def _te_targets(y: np.ndarray, spec) -> np.ndarray:
    """Матрица целей для кодирования: регрессия — y; классификация — one-hot (binary — один столбец)."""
    if not spec.is_clf:
        return np.asarray(y, dtype=float).reshape(-1, 1)
    oh = np.eye(spec.n_classes)[np.asarray(y, dtype=int)]
    return oh[:, 1:2] if spec.n_classes == 2 else oh


def _te_fit_apply(keys_fit: pd.Series, T_fit: np.ndarray, keys_apply: pd.Series, smoothing: float) -> np.ndarray:
    prior = T_fit.mean(axis=0)
    df = pd.DataFrame(T_fit, index=keys_fit.index)
    df["__k"] = keys_fit.values
    g = df.groupby("__k", observed=True)
    cnt = g.size()
    means = g.mean()
    w = (cnt / (cnt + smoothing)).values[:, None]
    enc = means.values * w + prior[None, :] * (1 - w)
    enc_df = pd.DataFrame(enc, index=means.index)
    out = enc_df.reindex(keys_apply.values).values
    out = np.where(np.isnan(out), prior[None, :], out)
    return out


def target_encode(X_fit: pd.DataFrame, y_fit: np.ndarray, X_apply: list[pd.DataFrame], cols: list[str],
                  spec, seed: int = 42, smoothing: float = 20.0, inner_folds: int = 5):
    """Возвращает (enc_fit, [enc_apply...]) — DataFrame с колонками te__*.
    Для строк X_fit кодировка вычисляется out-of-fold внутри обучающей части фолда."""
    T = _te_targets(y_fit, spec)
    names = []
    for c in cols:
        names += [f"te__{c}"] if T.shape[1] == 1 else [f"te__{c}__{j}" for j in range(T.shape[1])]

    def keys(df, c):
        return df[c].astype(object).where(df[c].notna(), "__nan__")

    fit_out = np.zeros((len(X_fit), len(names)))
    apply_out = [np.zeros((len(a), len(names))) for a in X_apply]
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    col0 = 0
    for c in cols:
        kfit = keys(X_fit, c)
        w = T.shape[1]
        for itr, iva in kf.split(np.arange(len(X_fit))):
            fit_out[iva, col0:col0 + w] = _te_fit_apply(kfit.iloc[itr], T[itr], kfit.iloc[iva], smoothing)
        for a, out in zip(X_apply, apply_out):
            out[:, col0:col0 + w] = _te_fit_apply(kfit, T, keys(a, c), smoothing)
        col0 += w
    enc_fit = pd.DataFrame(fit_out, columns=names, index=X_fit.index)
    enc_apply = [pd.DataFrame(o, columns=names, index=a.index) for o, a in zip(apply_out, X_apply)]
    return enc_fit, enc_apply
