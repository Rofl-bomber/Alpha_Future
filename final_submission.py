"""Сабмит: сборка таблицы по шаблону, валидатор и атомарная запись.

Валидатор: колонки и их порядок как в sample_submission; все id из test в том же
порядке, без лишних и дублей; нет NaN и inf; метки только из классов train;
вероятности в [0, 1]. Запись: временный файл рядом → os.replace.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import task_metrics

log = logging.getLogger("agent")


def constant_pred(spec, y, n_test: int):
    """Уровень 0: константа в формате внутреннего представления."""
    y = np.asarray(y)
    if spec.task_type == "regression":
        v = float(np.median(y)) if task_metrics.REGISTRY.get(spec.metric, None) and \
            task_metrics.REGISTRY[spec.metric].objective == "l1" else float(np.mean(y))
        return np.full(n_test, v)
    if spec.task_type == "multi_regression":
        return np.tile(y.mean(axis=0), (n_test, 1))
    if spec.task_type == "binary":
        return np.full(n_test, float(y.mean()))
    if spec.task_type == "multiclass":
        prior = np.bincount(y, minlength=spec.n_classes) / len(y)
        return np.tile(prior, (n_test, 1))
    if spec.task_type == "multilabel":
        return np.tile(y.mean(axis=0), (n_test, 1))
    raise ValueError(spec.task_type)


def constant_post(spec, y):
    """Постобработка для константы: порог так, чтобы предсказывался самый частый класс."""
    if spec.task_type == "binary":
        return {"thr": 0.5 if np.mean(y) >= 0.5 else 1.01}
    if spec.task_type == "multilabel":
        return {"thr": [0.5] * spec.n_classes}
    return {}


def _columns(spec):
    cols = list(spec.submission_columns)
    need = ([spec.id_col] if spec.id_col else []) + (
        [str(c) for c in spec.classes] if spec.prediction_kind == "proba_columns" else list(spec.targets))
    if not cols or not set(map(str, need)) <= set(map(str, cols)):
        cols = need
    return cols


def build_frame(pred, spec, test_ids, post=None) -> pd.DataFrame:
    p = task_metrics.apply_post(pred, spec.task_type, post, split="test")
    n = p.shape[0]
    cols = _columns(spec)
    out = {}
    if spec.id_col:
        out[spec.id_col] = test_ids.to_numpy() if test_ids is not None else np.arange(n)
    tt = spec.task_type
    if tt == "regression":
        out[spec.targets[0]] = np.asarray(p, dtype=float)
    elif tt == "multi_regression":
        for j, t in enumerate(spec.targets):
            out[t] = p[:, j]
    elif tt == "binary":
        if spec.prediction_kind == "proba":
            out[spec.targets[0]] = np.clip(p, 0, 1)
        else:
            lab = task_metrics.to_labels(p, tt, post)
            out[spec.targets[0]] = np.array(spec.classes, dtype=object)[lab]
    elif tt == "multiclass":
        if spec.prediction_kind == "proba_columns":
            P = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
            by_name = {str(c): j for j, c in enumerate(spec.classes)}
            for c in cols:
                if c != spec.id_col:
                    out[c] = P[:, by_name[str(c)]]
        else:
            lab = task_metrics.to_labels(p, tt, post)
            out[spec.targets[0]] = np.array(spec.classes, dtype=object)[lab]
    elif tt == "multilabel":
        if spec.metric_need == "proba":
            for j, t in enumerate(spec.targets):
                out[t] = np.clip(p[:, j], 0, 1)
        else:
            lab = task_metrics.to_labels(p, tt, post)
            for j, t in enumerate(spec.targets):
                out[t] = lab[:, j]
    df = pd.DataFrame(out)
    return df[[c for c in cols if c in df.columns]]


def validate(df: pd.DataFrame, spec, test_ids, n_test: int) -> list:
    errors = []
    cols = _columns(spec)
    if [str(c) for c in df.columns] != [str(c) for c in cols]:
        errors.append(f"колонки {list(df.columns)} вместо {cols}")
    if len(df) != n_test:
        errors.append(f"строк {len(df)} вместо {n_test}")
    if spec.id_col and spec.id_col in df.columns:
        if df[spec.id_col].duplicated().any():
            errors.append("повторяющиеся id")
        if test_ids is not None and len(df) == len(test_ids) and \
                not (df[spec.id_col].astype(str).to_numpy() == test_ids.astype(str).to_numpy()).all():
            errors.append("id не совпадают с test или порядок другой")
    for c in df.columns:
        if c == spec.id_col:
            continue
        s = df[c]
        if s.isna().any():
            errors.append(f"пропуски в {c}")
        if pd.api.types.is_numeric_dtype(s) and not np.isfinite(s.to_numpy(dtype=float)).all():
            errors.append(f"inf в {c}")
    if spec.task_type in ("binary", "multiclass") and spec.prediction_kind == "label":
        allowed = {str(c) for c in spec.classes}
        bad = ~df[spec.targets[0]].astype(str).isin(allowed)
        if bad.any():
            errors.append(f"{int(bad.sum())} меток вне списка классов")
    if spec.prediction_kind in ("proba", "proba_columns"):
        vals = df.drop(columns=[spec.id_col] if spec.id_col else []).to_numpy(dtype=float)
        if (vals < 0).any() or (vals > 1).any():
            errors.append("вероятности вне [0, 1]")
    return errors


class SubmissionWriter:
    """Пишет сабмит атомарно и помнит, какая версия на диске."""

    def __init__(self, spec, test_ids, n_test: int):
        self.spec = spec
        self.test_ids = test_ids
        self.n_test = n_test
        self.path = Path(spec.output_path)
        self.current = None

    def write(self, pred, post, tag: str, score=None) -> bool:
        try:
            df = build_frame(pred, self.spec, self.test_ids, post)
        except Exception as e:  # noqa: BLE001
            log.error("сабмит %s не собран: %s", tag, e)
            return False
        errors = validate(df, self.spec, self.test_ids, self.n_test)
        if errors:
            log.error("сабмит %s не прошёл валидацию: %s", tag, "; ".join(errors))
            return False
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)
        self.current = {"tag": tag, "score": score}
        log.info("сабмит записан: %s%s", tag, f" (CV {score:.6g})" if score is not None else "")
        return True
