"""Общий интерфейс моделей раннера.

fit(X_tr, y_tr, X_va, y_va, cat_cols) -> self
predict(X) -> (n,) для регрессии, (n, K) вероятности для классификации

X_va, y_va — выборка для early stopping (внутренний сплит обучающей части фолда, см. runner);
скор считается на внешнем валидационном фолде, которого модель не видит.

Схема параметров (param_schema) — источник правды для проверки гипотез и для меню Исследователя:
    ("int", lo, hi) | ("float", lo, hi) | ("choice", [значения]) | ("bool",) | ("objective",)
lo/hi включительно, None — без границы; ("objective",) — список из objectives[task_type].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Параметры, которые выставляет харнесс и которые нельзя переопределять из гипотез.
RESERVED = {
    "n_jobs", "nthread", "num_threads", "thread_count", "random_state", "random_seed", "seed",
    "verbose", "verbosity", "silent", "logging_level", "num_class", "allow_writing_files",
    "train_dir", "enable_categorical", "device", "task_type", "gpu_id", "tree_method",
}


def _task_key(spec) -> str:
    return spec.task_type if spec.task_type in ("regression", "binary", "multiclass") else "multiclass"


def _to_float(v) -> float:
    if isinstance(v, bool):
        raise ValueError("ожидается число, получено bool")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    raise ValueError(f"ожидается число, получено {type(v).__name__}")


def _fmt_num(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


class ModelBase:
    name = "base"
    defaults: dict = {}
    uses_eval: bool = True             # нужна ли модели выборка для early stopping
    param_schema: dict = {}
    param_aliases: dict = {}
    objectives: dict = {}              # task_type -> допустимые значения для ("objective",)
    clf_only: set = set()              # параметры только для классификации
    reg_only: set = set()              # параметры только для регрессии
    doc_defaults: dict = {}            # значения по умолчанию для меню, которых нет в defaults

    def __init__(self, params: dict, spec, n_threads: int = 1, seed: int = 42):
        self.spec = spec
        self.is_clf = spec.is_clf
        self.n_classes = spec.n_classes
        self.n_threads = n_threads
        self.seed = seed
        self.params = {**self.defaults, **{k: v for k, v in (params or {}).items() if k not in RESERVED}}
        self.importances: dict[str, float] | None = None
        self.best_iteration: int | None = None

    def fit(self, X_tr, y_tr, X_va, y_va, cat_cols):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    # --- проверка параметров
    @classmethod
    def _check_value(cls, key: str, value, spec):
        rule = cls.param_schema[key]
        kind = rule[0]
        if kind == "objective":
            choices = cls.objectives.get(_task_key(spec), [])
            kind, rule = "choice", ("choice", choices)
        if kind in ("int", "float"):
            x = _to_float(value)
            if not np.isfinite(x):
                raise ValueError(f"{key} = {value!r}: нужно конечное число")
            if kind == "int":
                if not float(x).is_integer():
                    raise ValueError(f"{key} = {value!r}: нужно целое число")
                x = int(x)
            lo, hi = rule[1], rule[2]
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                rng = f"{_fmt_num(lo) if lo is not None else '-inf'}–{_fmt_num(hi) if hi is not None else 'inf'}"
                raise ValueError(f"{key} = {_fmt_num(x)} вне диапазона {rng}")
            return x
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
                return value.strip().lower() in ("true", "1")
            raise ValueError(f"{key} = {value!r}: нужно true/false")
        if kind == "choice":
            choices = rule[1]
            for c in choices:
                if c is None and (value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null"))):
                    return None
                if isinstance(c, str) and isinstance(value, str) and c.lower() == value.strip().lower():
                    return c
            shown = ", ".join("null" if c is None else str(c) for c in choices)
            raise ValueError(f"{key} = {value!r}: допустимо {shown}")
        raise ValueError(f"{key}: неизвестный тип правила {kind}")

    @classmethod
    def validate(cls, params: dict | None, spec) -> tuple[dict, list[str], list[str]]:
        """Возвращает (канонические параметры, ошибки, исправления).
        Алиасы заменяются на канонические имена; неизвестные имена и значения вне диапазона — ошибки."""
        out: dict = {}
        errors: list[str] = []
        fixes: list[str] = []
        task_clf = spec.is_clf
        for k, v in (params or {}).items():
            key = str(k)
            if key in RESERVED:
                fixes.append(f"{key}: задаётся харнессом, убран")
                continue
            if key not in cls.param_schema and key in cls.param_aliases:
                canon = cls.param_aliases[key]
                fixes.append(f"{key} -> {canon}")
                key = canon
            if key not in cls.param_schema:
                allowed = ", ".join(cls.allowed_names(spec))
                errors.append(f"неизвестный параметр {k} для {cls.name}; допустимые: {allowed}")
                continue
            if key in cls.clf_only and not task_clf:
                errors.append(f"параметр {key} модели {cls.name} применим только к классификации")
                continue
            if key in cls.reg_only and task_clf:
                errors.append(f"параметр {key} модели {cls.name} применим только к регрессии")
                continue
            try:
                out[key] = cls._check_value(key, v, spec)
            except ValueError as e:
                errors.append(f"{cls.name}: {e}")
        return out, errors, fixes

    @classmethod
    def allowed_names(cls, spec) -> list[str]:
        names = []
        for k in cls.param_schema:
            if k in cls.clf_only and not spec.is_clf:
                continue
            if k in cls.reg_only and spec.is_clf:
                continue
            names.append(k)
        return names

    @classmethod
    def menu(cls, spec) -> str:
        """Строка меню для Исследователя: имя, тип/диапазон, значение по умолчанию."""
        dflt = {**cls.defaults, **cls.doc_defaults}
        parts = []
        for k in cls.allowed_names(spec):
            rule = cls.param_schema[k]
            kind = rule[0]
            if kind in ("int", "float"):
                lo = _fmt_num(rule[1]) if rule[1] is not None else "-inf"
                hi = _fmt_num(rule[2]) if rule[2] is not None else "inf"
                desc = f"{kind} {lo}–{hi}"
            elif kind == "bool":
                desc = "true|false"
            else:
                choices = cls.objectives.get(_task_key(spec), []) if kind == "objective" else rule[1]
                desc = "|".join("null" if c is None else str(c) for c in choices)
            d = dflt.get(k, None)
            if isinstance(d, dict):
                d = d.get(_task_key(spec))
            parts.append(f"{k} {desc}" + (f" ({_fmt_num(d)})" if d is not None else ""))
        return "; ".join(parts)

    # --- утилиты
    def expand_proba(self, p: np.ndarray, classes) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = np.stack([1 - p, p], axis=1)
        classes = list(np.asarray(classes).astype(int))
        if len(classes) == self.n_classes and classes == list(range(self.n_classes)):
            return p
        out = np.zeros((p.shape[0], self.n_classes))
        for j, c in enumerate(classes):
            out[:, c] = p[:, j]
        return out


def to_numeric_frame(X: pd.DataFrame, cat_cols: list[str], fill: float | None = None) -> pd.DataFrame:
    """Категории -> коды (NaN = -1 или NaN), остальное как есть."""
    Z = X.copy()
    for c in cat_cols:
        codes = Z[c].cat.codes.astype(float) if isinstance(Z[c].dtype, pd.CategoricalDtype) else \
            pd.Categorical(Z[c]).codes.astype(float)
        codes[codes < 0] = np.nan
        Z[c] = codes
    Z = Z.astype(float)
    if fill is not None:
        Z = Z.fillna(fill)
    return Z
