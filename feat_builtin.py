"""Библиотека готовых операций: признаки (без таргета) и кодирования (с таргетом — только OOF).

Операции признаков работают на объединённых признаках train+test без таргета,
поэтому утечка таргета через них невозможна. Целевое кодирование делает только
encode_fold и строго out-of-fold.
"""

import numpy as np
import pandas as pd

OPS_DOC = {
    "freq_enc": "частота каждого значения категориальных колонок (по train+test)",
    "cat_combos": "сочетания пар низкокардинальных категорий как новые категории",
    "group_stats": "среднее числового признака по группе категории и отклонение от него",
    "num_ratios": "попарные отношения самых важных числовых признаков",
    "row_stats": "статистики по строке: среднее, std, min, max стандартизованных числовых",
    "missing_count": "число пропусков в строке",
    "dup_count": "сколько раз такая же строка признаков встречается в train+test",
    "date_parts": "год, месяц, день, день недели, час, эпоха из колонок-дат",
    "text_stats": "длина, число слов, доля цифр и заглавных в текстовых колонках",
    "text_tfidf_svd": "TF-IDF текста → 16 компонент SVD",
    "segment_flags": "флаги сегментов с крупнейшими ошибками (из анализа ошибок)",
}

ENCODINGS = ("native", "ordinal", "freq", "target_oof", "onehot")


def applicable_ops(spec, hints: dict | None = None) -> list:
    ops = []
    if hints and hints.get("segments"):
        ops.append("segment_flags")
    if spec.cat_cols:
        ops += ["freq_enc", "group_stats"]
        if len(spec.cat_cols) >= 2:
            ops.append("cat_combos")
    if len(spec.num_cols) >= 2:
        ops += ["num_ratios", "row_stats"]
    ops += ["missing_count", "dup_count"]
    if spec.datetime_cols:
        ops.append("date_parts")
    if spec.text_cols:
        ops += ["text_stats", "text_tfidf_svd"]
    return ops


def default_ops(spec) -> list:
    """Операции, без которых сырые колонки не дойдут до модели (даты и тексты)."""
    ops = []
    if spec.datetime_cols:
        ops.append("date_parts")
    if spec.text_cols:
        ops.append("text_stats")
    return ops


def _card(X, c):
    return X[c].nunique(dropna=True)


def _top_num(ctx, X, k):
    top = [c for c in ctx.get("top_num", []) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(top) < k:
        rest = [c for c in ctx["num_cols"] if c in X.columns and c not in top]
        rest = sorted(rest, key=lambda c: -float(X[c].std() or 0) / (abs(float(X[c].mean() or 0)) + 1e-9))
        top += rest
    return top[:k]


def apply_ops(X: pd.DataFrame, ops: list, ctx: dict) -> pd.DataFrame:
    new = {}
    cats = [c for c in ctx["cat_cols"] if c in X.columns]
    nums = [c for c in ctx["num_cols"] if c in X.columns]
    for op in ops:
        if op == "freq_enc":
            for c in cats:
                vc = X[c].astype(str).map(X[c].astype(str).value_counts(normalize=True))
                new[f"{c}__freq"] = vc.astype(float).to_numpy()
        elif op == "cat_combos":
            low = sorted([c for c in cats if 2 <= _card(X, c) <= 100], key=lambda c: _card(X, c))[:4]
            for i in range(len(low)):
                for j in range(i + 1, len(low)):
                    a, b = low[i], low[j]
                    new[f"{a}__x__{b}"] = pd.Categorical(X[a].astype(str) + "|" + X[b].astype(str))
        elif op == "group_stats":
            gcols = sorted([c for c in cats if 2 <= _card(X, c) <= 1000], key=lambda c: _card(X, c))[:3]
            for g in gcols:
                key = X[g].astype(str)
                for c in _top_num(ctx, X, 5):
                    m = X[c].groupby(key).transform("mean")
                    new[f"{c}__mean_by_{g}"] = m.to_numpy()
                    new[f"{c}__dev_by_{g}"] = (X[c] - m).to_numpy()
        elif op == "num_ratios":
            top = _top_num(ctx, X, 5)
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a, b = X[top[i]], X[top[j]]
                    new[f"{top[i]}__div__{top[j]}"] = (a / (b.abs() + 1e-9) * np.sign(b + 1e-12)).to_numpy()
        elif op == "row_stats" and nums:
            Z = X[nums].astype(float)
            Z = (Z - Z.mean()) / (Z.std() + 1e-9)
            new["row__mean"] = Z.mean(axis=1).to_numpy()
            new["row__std"] = Z.std(axis=1).to_numpy()
            new["row__min"] = Z.min(axis=1).to_numpy()
            new["row__max"] = Z.max(axis=1).to_numpy()
        elif op == "segment_flags":
            for col, value in ctx.get("segments", []):
                if col in X.columns:
                    new[f"seg__{col}__{str(value)[:20]}"] = (X[col].astype(str) == str(value)).astype(float).to_numpy()
        elif op == "missing_count":
            new["row__n_missing"] = X.isna().sum(axis=1).astype(float).to_numpy()
        elif op == "dup_count":
            cols = [c for c in X.columns if X[c].dtype != object]
            h = pd.util.hash_pandas_object(X[cols], index=False)
            new["row__dup_count"] = h.map(h.value_counts()).astype(float).to_numpy()
        elif op == "date_parts":
            for c in ctx.get("datetime_cols", []):
                if c not in X.columns:
                    continue
                t = pd.to_datetime(X[c], errors="coerce", format="mixed")
                new[f"{c}__year"] = t.dt.year.astype(float).to_numpy()
                new[f"{c}__month"] = t.dt.month.astype(float).to_numpy()
                new[f"{c}__day"] = t.dt.day.astype(float).to_numpy()
                new[f"{c}__dow"] = t.dt.dayofweek.astype(float).to_numpy()
                new[f"{c}__doy"] = t.dt.dayofyear.astype(float).to_numpy()
                if (t.dt.hour.fillna(0) != 0).any():
                    new[f"{c}__hour"] = t.dt.hour.astype(float).to_numpy()
                epoch = (t - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
                new[f"{c}__epoch_days"] = epoch.astype(float).to_numpy()
        elif op == "text_stats":
            for c in ctx.get("text_cols", []):
                if c not in X.columns:
                    continue
                s = X[c].fillna("").astype(str)
                length = s.str.len().astype(float)
                new[f"{c}__len"] = length.to_numpy()
                new[f"{c}__words"] = s.str.split().str.len().astype(float).to_numpy()
                new[f"{c}__digits"] = (s.str.count(r"\d") / (length + 1)).to_numpy()
                new[f"{c}__upper"] = (s.str.count(r"[A-ZА-Я]") / (length + 1)).to_numpy()
        elif op == "text_tfidf_svd":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            for c in ctx.get("text_cols", []):
                if c not in X.columns:
                    continue
                s = X[c].fillna("").astype(str)
                tf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
                try:
                    M = tf.fit_transform(s)
                    k = min(16, M.shape[1] - 1)
                    if k >= 2:
                        comp = TruncatedSVD(n_components=k, random_state=0).fit_transform(M)
                        for j in range(k):
                            new[f"{c}__svd{j}"] = comp[:, j]
                except ValueError:
                    continue
    if not new:
        return X
    add = pd.DataFrame(new, index=X.index)
    add = add.loc[:, [c for c in add.columns if c not in X.columns]]
    return pd.concat([X, add], axis=1)


def drop_raw_objects(X: pd.DataFrame) -> pd.DataFrame:
    """Сырые текстовые и датовые колонки (object) моделям не передаются."""
    obj = [c for c in X.columns if X[c].dtype == object]
    return X.drop(columns=obj) if obj else X


# ---------------------------------------------------------------- предобработка без таргета


def rare_merge(X: pd.DataFrame, cat_cols, min_count: int) -> pd.DataFrame:
    if min_count <= 1:
        return X
    X = X.copy()
    for c in cat_cols:
        if c not in X.columns:
            continue
        s = X[c].astype(object)
        vc = s.value_counts()
        rare = set(vc[vc < min_count].index)
        if rare:
            s = s.where(~s.isin(rare), "__rare__")
            X[c] = s.astype("category")
    return X


def missing_flags(X: pd.DataFrame, cols) -> pd.DataFrame:
    flags = {}
    for c in cols:
        if c in X.columns:
            r = X[c].isna().mean()
            if 0.005 < r < 0.99:
                flags[f"{c}__isna"] = X[c].isna().astype(float).to_numpy()
    if not flags:
        return X
    return pd.concat([X, pd.DataFrame(flags, index=X.index)], axis=1)


def category_columns(X: pd.DataFrame) -> list:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Категории → коды (NaN для пропуска). Для моделей без нативных категорий."""
    out = {}
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.cat.codes.astype(float)
            out[c] = codes.where(codes >= 0, np.nan).to_numpy()
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float).to_numpy()
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float).to_numpy()
    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------- кодирования внутри фолда


def _te_targets(y, task_type, n_classes):
    """Матрица «целей» для target encoding: регрессия — y; классы — one-hot (до 10 классов)."""
    y = np.asarray(y)
    if task_type == "multiclass":
        counts = np.bincount(y.astype(int), minlength=n_classes)
        keep = np.argsort(-counts)[: min(10, n_classes)]
        return np.stack([(y == k).astype(float) for k in keep], axis=1), [f"te{k}" for k in keep]
    if task_type == "multilabel":
        return y[:, :10].astype(float), [f"te{j}" for j in range(min(10, y.shape[1]))]
    if y.ndim == 2:
        return y[:, :1].astype(float), ["te"]
    return y.astype(float)[:, None], ["te"]


def _te_map(keys: pd.Series, T: np.ndarray, prior: np.ndarray, m: float):
    df = pd.DataFrame(T, index=keys.index)
    df["__k"] = keys.to_numpy()
    g = df.groupby("__k")
    s = g.sum()
    n = g.size()
    return (s.add(prior * m, axis=1)).div(n + m, axis=0)


def encode_fold(Xtr: pd.DataFrame, others: list, ytr, cat_cols, method: str, task_type: str, n_classes: int,
                seed: int = 0, m: float = 20.0):
    """Применить кодирование категорий. Таргет используется только для target_oof и только OOF."""
    cats = [c for c in cat_cols if c in Xtr.columns]
    if method == "native" or not cats:
        return Xtr, others
    if method == "ordinal":
        return to_numeric(Xtr), [to_numeric(o) for o in others]
    if method == "freq":
        def fq(X, ref):
            X = X.copy()
            for c in cats:
                vc = ref[c].astype(str).value_counts(normalize=True)
                X[c] = X[c].astype(str).map(vc).astype(float).fillna(0.0).to_numpy()
            return X
        allX = pd.concat([Xtr] + others, axis=0)
        return fq(Xtr, allX), [fq(o, allX) for o in others]
    if method == "onehot":
        def oh(X, levels):
            X = X.copy()
            for c in cats:
                if c in levels:
                    for v in levels[c]:
                        X[f"{c}=={v}"] = (X[c].astype(str) == v).astype(float).to_numpy()
                    X = X.drop(columns=[c])
            return to_numeric(X)
        levels = {}
        for c in cats:
            vc = Xtr[c].astype(str).value_counts()
            if len(vc) <= 30:
                levels[c] = list(vc.index)
        return oh(Xtr, levels), [oh(o, levels) for o in others]
    if method == "target_oof":
        T, names = _te_targets(ytr, task_type, n_classes)
        prior = T.mean(axis=0)
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 5, len(Xtr))
        Xtr2 = Xtr.copy()
        others2 = [o.copy() for o in others]
        for c in cats:
            ktr = Xtr[c].astype(str).reset_index(drop=True)
            enc_tr = np.zeros((len(Xtr), T.shape[1]))
            for f in range(5):
                mask = fold == f
                mp = _te_map(ktr[~mask], T[~mask], prior, m)
                enc_tr[mask] = mp.reindex(ktr[mask].to_numpy()).fillna(pd.Series(prior, index=mp.columns)).to_numpy()
            mp_full = _te_map(ktr, T, prior, m)
            for j, nm in enumerate(names):
                Xtr2[f"{c}__{nm}"] = enc_tr[:, j]
            for o, o2 in zip(others, others2):
                ko = o[c].astype(str).to_numpy()
                enc = mp_full.reindex(ko).fillna(pd.Series(prior, index=mp_full.columns)).to_numpy()
                for j, nm in enumerate(names):
                    o2[f"{c}__{nm}"] = enc[:, j]
            # исходную категорию оставляем как частоту, чтобы модели не теряли информацию
        allX = pd.concat([Xtr] + others, axis=0)
        for c in cats:
            vc = allX[c].astype(str).value_counts(normalize=True)
            Xtr2[c] = Xtr[c].astype(str).map(vc).astype(float).to_numpy()
            for o, o2 in zip(others, others2):
                o2[c] = o[c].astype(str).map(vc).astype(float).to_numpy()
        return Xtr2, others2
    raise ValueError(f"неизвестное кодирование {method}")
