"""Метрики прототипа: rmse (меньше — лучше) и f1_macro (больше — лучше).
Для классификации предсказания — матрица вероятностей (n, K)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def rmse(y_true, pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def f1_macro(y_true, proba, n_classes: int | None = None) -> float:
    proba = np.asarray(proba)
    labels_pred = proba.argmax(axis=1) if proba.ndim == 2 else proba.astype(int)
    k = n_classes or (proba.shape[1] if proba.ndim == 2 else int(max(labels_pred.max(), np.max(y_true)) + 1))
    return float(f1_score(y_true, labels_pred, average="macro", labels=list(range(k)), zero_division=0))


GREATER_IS_BETTER = {"rmse": False, "f1_macro": True}


def score(metric: str, y_true, pred, n_classes: int | None = None) -> float:
    if metric == "rmse":
        return rmse(y_true, pred)
    if metric == "f1_macro":
        return f1_macro(y_true, pred, n_classes)
    raise ValueError(f"Неизвестная метрика: {metric}")


def improvement(metric: str, new: float, base: float) -> float:
    """Положительное значение = new лучше base."""
    return (new - base) if GREATER_IS_BETTER[metric] else (base - new)


def relative_improvement(metric: str, new: float, base: float) -> float:
    return improvement(metric, new, base) / max(abs(base), 1e-12)
