"""Аварийный оператор: полный пайплайн от Инженера со строгим контрактом.

    def fit_predict(train: pd.DataFrame, y, predict_on: list[pd.DataFrame], spec: dict,
                    time_budget_s: float) -> list[np.ndarray]

Харнесс вызывает функцию на каждом поисковом фолде (обучающая часть с метками,
валидационная — без), затем отдельно для selection-holdout и test, тоже без меток.
Метрику по OOF считает харнесс. Выход приводится к внутреннему представлению
(task_metrics) и проверяется: длина, форма, NaN.
"""

import time

import numpy as np

import task_metrics


def load_fn(code: str):
    ns = {}
    exec(compile(code, "<escape_fit_predict>", "exec"), ns)
    fn = ns.get("fit_predict")
    if not callable(fn):
        raise ValueError("в коде нет функции fit_predict")
    return fn


def coerce(out, n: int, spec) -> np.ndarray:
    """Привести выход fit_predict к внутреннему представлению."""
    a = np.asarray(out)
    tt = spec.task_type
    if a.shape[0] != n:
        raise ValueError(f"fit_predict вернула {a.shape[0]} строк вместо {n}")
    if tt == "regression":
        a = a.reshape(n, -1)[:, 0].astype(float)
    elif tt == "multi_regression":
        a = a.reshape(n, -1).astype(float)
    elif tt == "binary":
        if a.ndim == 2 and a.shape[1] == 2:
            a = a[:, 1]
        a = a.reshape(-1)
        if a.dtype.kind in "OUS":  # метки классов строками → индексы
            index = {str(c): i for i, c in enumerate(spec.classes)}
            a = np.array([index[str(v)] for v in a], dtype=float)
        a = a.astype(float)
    elif tt == "multiclass":
        K = spec.n_classes
        if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
            a = a.reshape(-1)
            index = {str(c): i for i, c in enumerate(spec.classes)}
            if a.dtype.kind in "OUS":
                idx = np.array([index[str(v)] for v in a])
            else:
                idx = a.astype(int)
            a = np.eye(K)[idx]
        a = a.astype(float)
        if a.shape[1] != K:
            raise ValueError(f"ожидалось {K} столбцов вероятностей, получено {a.shape[1]}")
        a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
    elif tt == "multilabel":
        a = a.reshape(n, -1).astype(float)
    if not np.isfinite(a).all():
        raise ValueError("NaN или inf в выходе fit_predict")
    return a


def run_escape(code, spec, Xs, ys, Xh, Xte, folds, deadline, predict_ho_te=True, ks=None):
    """OOF по фолдам ks (по умолчанию — все) + отдельный вызов для holdout и test. Возвращает (oof, ho, te)."""
    fn = load_fn(code)
    spec_d = {"task_type": spec.task_type, "metric": spec.metric, "classes": list(spec.classes),
              "targets": list(spec.targets), "cat_cols": list(spec.cat_cols), "text_cols": list(spec.text_cols),
              "datetime_cols": list(spec.datetime_cols)}
    ks = list(range(int(folds.max()) + 1)) if ks is None else list(ks)
    oof = None
    for i, k in enumerate(ks):
        tr, va = folds != k, folds == k
        budget = max(5.0, (deadline - time.time()) / (len(ks) - i + (1 if predict_ho_te else 0)))
        out = fn(Xs[tr].reset_index(drop=True), ys[tr], [Xs[va].reset_index(drop=True)], dict(spec_d), budget)
        p = coerce(out[0], int(va.sum()), spec)
        if oof is None:
            oof = np.zeros((len(ys),) + p.shape[1:])
        oof[va] = p
    ho = te = None
    if predict_ho_te:
        budget = max(5.0, deadline - time.time())
        out = fn(Xs.reset_index(drop=True), ys, [Xh.reset_index(drop=True), Xte.reset_index(drop=True)],
                 dict(spec_d), budget)
        ho = coerce(out[0], len(Xh), spec)
        te = coerce(out[1], len(Xte), spec)
    return oof, ho, te


def check_on_sample(code, spec, X, y, n_rows=2000, seed=0, budget=60.0):
    """Быстрая проверка контракта на подвыборке: обучение на 75%, предсказание на 25%."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[: min(n_rows, len(X))]
    cut = int(len(idx) * 0.75)
    tr, te = idx[:cut], idx[cut:]
    fn = load_fn(code)
    spec_d = {"task_type": spec.task_type, "metric": spec.metric, "classes": list(spec.classes),
              "targets": list(spec.targets), "cat_cols": list(spec.cat_cols), "text_cols": list(spec.text_cols),
              "datetime_cols": list(spec.datetime_cols)}
    t0 = time.time()
    out = fn(X.iloc[tr].reset_index(drop=True), y[tr], [X.iloc[te].reset_index(drop=True)], spec_d, budget)
    p = coerce(out[0], len(te), spec)
    metric = task_metrics.metric_for(spec)
    s = task_metrics.score(metric, y[te], p, spec.task_type)
    return {"ok": True, "score_on_sample": s, "time_s": round(time.time() - t0, 2), "shape": list(p.shape)}
