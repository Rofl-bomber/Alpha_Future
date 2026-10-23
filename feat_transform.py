"""Цепочка transform — код Инженера для подэтапов Preprocessing и Features.

Контракт слота:
    def transform(train_df, y_train, others) -> (train_df, others)
train_df — обучающая часть фолда (текущие данные решения), y_train — её таргет, others — список
кадров без таргета (валидация, holdout, test). Харнесс вызывает функцию внутри каждого фолда,
поэтому статистики таргета, посчитанные по y_train, не утекают в валидацию.

Харнесс проверяет только форму выхода (normalize): то же число строк в том же порядке, одинаковые
колонки во всех кадрах, типы — числа, bool или категории (строки → категории, даты → дни, inf → NaN).
Принятые transform образуют цепочку config.features.chain: id кода по порядку, код — codes/<id>.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class TransformError(Exception):
    pass


def load_fn(code: str):
    ns = {"np": np, "pd": pd}
    exec(compile(code, "<transform>", "exec"), ns)
    fn = ns.get("transform")
    if not callable(fn):
        raise TransformError("в коде нет функции transform(train_df, y_train, others)")
    return fn


def _to_frame(obj, name: str, n: int) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if not isinstance(obj, pd.DataFrame):
        raise TransformError(f"{name}: ожидался DataFrame, получено {type(obj).__name__}")
    if len(obj) != n:
        raise TransformError(f"{name}: {len(obj)} строк вместо {n} (строки нельзя добавлять и удалять)")
    obj = obj.reset_index(drop=True)
    obj.columns = [str(c) for c in obj.columns]
    dup = obj.columns[obj.columns.duplicated()].tolist()
    if dup:
        raise TransformError(f"{name}: повторяющиеся имена колонок {dup[:5]}")
    return obj


def normalize(train_out, others_out, n_train: int, n_others: list, keep_raw=()) -> tuple:
    """Проверить и привести выход transform к единому виду. keep_raw — сырые текст/даты, их не трогаем."""
    tr = _to_frame(train_out, "train_df", n_train)
    if not isinstance(others_out, (list, tuple)) or len(others_out) != len(n_others):
        raise TransformError(f"others: ожидался список из {len(n_others)} кадров")
    others = [_to_frame(o, f"others[{i}]", n) for i, (o, n) in enumerate(zip(others_out, n_others))]
    cols = list(tr.columns)
    for i, o in enumerate(others):
        miss = [c for c in cols if c not in o.columns]
        if miss:
            raise TransformError(f"others[{i}]: нет колонок {miss[:8]}, которые есть в train_df")
        others[i] = o[cols]
    frames = [tr] + others
    for c in cols:
        s0 = tr[c]
        if c in keep_raw and s0.dtype == object:
            continue
        if isinstance(s0.dtype, pd.CategoricalDtype) or s0.dtype == object or pd.api.types.is_string_dtype(s0):
            if all(isinstance(f[c].dtype, pd.CategoricalDtype) and f[c].cat.categories.equals(s0.cat.categories)
                   and s0.cat.categories.dtype == object for f in frames):
                continue  # уже согласованные строковые категории
            cats = pd.Index([], dtype=object)
            for f in frames:
                s = f[c]
                vals = s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else pd.Index(s.dropna().unique())
                cats = cats.union(pd.Index(vals.astype(str), dtype=object))
            for f in frames:
                s = f[c]
                f[c] = pd.Categorical(s.astype(str).where(s.notna(), None), categories=cats)
            continue
        for f in frames:
            s = f[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                v = (s - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
            elif pd.api.types.is_timedelta64_dtype(s):
                v = s.dt.total_seconds()
            elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                v = s.astype(float)
            else:
                v = pd.to_numeric(s, errors="coerce")
            f[c] = v.replace([np.inf, -np.inf], np.nan).astype("float64")
    return frames[0], frames[1:]


def apply(code: str, train_df: pd.DataFrame, y_train, others: list, keep_raw=()) -> tuple:
    fn = load_fn(code)
    n_tr, n_o = len(train_df), [len(o) for o in others]
    out = fn(train_df.copy(), np.asarray(y_train).copy(), [o.copy() for o in others])
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TransformError("transform должна вернуть (train_df, others)")
    return normalize(out[0], out[1], n_tr, n_o, keep_raw)


def read_code(workdir, tid: str) -> str:
    return (Path(workdir) / "codes" / f"{tid}.py").read_text(encoding="utf-8")


def run_chain(ids: list, workdir, train_df, y_train, others: list, keep_raw=()) -> tuple:
    for tid in ids:
        try:
            train_df, others = apply(read_code(workdir, tid), train_df, y_train, others, keep_raw)
        except TransformError as e:
            raise TransformError(f"{tid}: {e}") from e
    return train_df, others


def cache_key(parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def describe_diff(before: pd.DataFrame, after: pd.DataFrame, limit: int = 40) -> str:
    """Что transform сделал с колонками: добавлено / удалено / изменено (для Инженера после запуска)."""
    b, a = set(before.columns), list(after.columns)
    added = [c for c in a if c not in b]
    removed = [c for c in before.columns if c not in set(a)]
    changed = []
    for c in a:
        if c in b and c not in added:
            x, y = before[c], after[c]
            same_type = str(x.dtype) == str(y.dtype)
            try:
                same = same_type and x.reset_index(drop=True).equals(y.reset_index(drop=True))
            except Exception:
                same = False
            if not same:
                changed.append(c)
    lines = [f"колонок было {before.shape[1]}, стало {after.shape[1]}; добавлено {len(added)}, удалено {len(removed)}, "
             f"изменено {len(changed)}"]
    for title, names in (("добавлены", added), ("изменены", changed)):
        for c in names[:limit]:
            s = after[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                lines.append(f"  {title}: {c} [категория] уник.={s.nunique()}, NaN={s.isna().mean():.1%}")
            else:
                lines.append(f"  {title}: {c} min={s.min():.4g}, mean={s.mean():.4g}, max={s.max():.4g}, "
                             f"NaN={s.isna().mean():.1%}, уник.={s.nunique()}")
    if removed:
        lines.append("  удалены: " + ", ".join(removed[:limit]))
    return "\n".join(lines)
