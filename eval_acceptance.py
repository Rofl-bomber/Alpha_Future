"""Критерий принятия гипотез и детерминированные проверки на взлом метрики.

Принятие: средний парный выигрыш по фолдам больше max(порог шума, SE парных разниц),
и улучшение есть минимум на доле фолдов cfg.accept_min_fold_frac (3 из 5).
Проверки: утечка таргета через новый признак, зависимость от порядка строк и id,
подозрительный скачок, подозрительно быстрое обучение, некорректный выход.
"""

import math

import numpy as np
import pandas as pd


def paired_compare(cand_scores, ref_scores, greater_is_better: bool, min_delta: float, min_fold_frac: float,
                   se_mult: float = 1.0) -> dict:
    a = np.asarray(cand_scores, dtype=float)
    b = np.asarray(ref_scores, dtype=float)
    k = min(len(a), len(b))
    if k == 0:
        return {"accepted": False, "mean_gain": float("nan"), "se": float("nan"), "wins": 0, "k": 0}
    d = (a[:k] - b[:k]) * (1 if greater_is_better else -1)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(k)) if k > 1 else abs(mean)
    wins = int((d > 0).sum())
    need = math.ceil(min_fold_frac * k - 1e-9)
    accepted = bool(mean > max(min_delta, se_mult * se) and wins >= need)
    return {"accepted": accepted, "mean_gain": mean, "se": se, "wins": wins, "k": k, "need_wins": need}


def noise_floor(fold_scores, cfg) -> float:
    """Порог шума: доля от разброса фолдов L1 и минимальная относительная дельта."""
    s = np.asarray(fold_scores, dtype=float)
    if len(s) < 2:
        return 0.0
    rel = cfg.min_delta_rel * max(abs(float(s.mean())), 1e-9)
    return float(max(rel, 0.05 * s.std(ddof=1) / math.sqrt(len(s))))


def relative_jump(cand_mean: float, ref_mean: float, greater_is_better: bool) -> float:
    """Относительное улучшение (положительное — кандидат лучше)."""
    base = max(abs(ref_mean), 1e-12)
    return (cand_mean - ref_mean) / base * (1 if greater_is_better else -1)


def leak_check(new_cols: pd.DataFrame, y_search, row_ids, task_type: str, threshold: float) -> list:
    """Флаги по новым признакам (только строки поисковой части train).

    - сильная связь с таргетом (|Спирмен| > порога; для классов — по one-hot);
    - монотонность по порядку строк или по id.
    """
    flags = []
    if new_cols is None or new_cols.shape[1] == 0:
        return flags
    y = np.asarray(y_search)
    pos = np.arange(len(new_cols))
    for c in new_cols.columns:
        s = new_cols[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype("category").cat.codes.replace(-1, np.nan)
        s = pd.Series(np.asarray(s, dtype=float))
        if s.notna().sum() < 50 or s.nunique() < 2:
            continue
        r = s.rank()
        if task_type in ("regression", "multi_regression"):
            yy = y if y.ndim == 1 else y[:, 0]
            corr = abs(r.corr(pd.Series(yy).rank()))
        elif task_type in ("binary", "multiclass"):
            corr = 0.0
            for k in np.unique(y):
                corr = max(corr, abs(r.corr(pd.Series((y == k).astype(float)))))
        else:
            corr = 0.0
        if corr == corr and corr > threshold:
            flags.append(f"leak_target:{c}:{corr:.3f}")
        c_pos = abs(r.corr(pd.Series(pos).rank()))
        if c_pos == c_pos and c_pos > threshold:
            flags.append(f"row_order:{c}:{c_pos:.3f}")
        if row_ids is not None:
            ids = pd.to_numeric(pd.Series(np.asarray(row_ids)), errors="coerce")
            if ids.notna().all():
                c_id = abs(r.corr(ids.rank()))
                if c_id == c_id and c_id > threshold:
                    flags.append(f"id_like:{c}:{c_id:.3f}")
    return flags


def output_check(pred, n_rows: int, task_type: str, n_classes: int) -> str | None:
    p = np.asarray(pred)
    if p.shape[0] != n_rows:
        return f"длина {p.shape[0]} вместо {n_rows}"
    if not np.isfinite(p).all():
        return "NaN или inf в предсказаниях"
    if task_type in ("binary", "multiclass", "multilabel"):
        if p.min() < -1e-6 or p.max() > 1 + 1e-6:
            return "вероятности вне [0, 1]"
        if task_type == "multiclass" and (p.ndim != 2 or p.shape[1] != n_classes):
            return f"ожидалось {n_classes} столбцов вероятностей"
    return None
