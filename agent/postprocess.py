"""Постобработка: множители вероятностей классов под macro F1 (подбор на OOF)
и клиппинг регрессии в диапазон train."""

from __future__ import annotations

import numpy as np

GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 25))


def _fast_f1_macro(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean())


def fit_class_multipliers(y: np.ndarray, proba: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Покоординатный подъём по множителям классов, максимизируем macro F1 по argmax(p * w)."""
    y = np.asarray(y, dtype=int)
    k = proba.shape[1]
    w = np.ones(k)
    best = _fast_f1_macro(y, proba.argmax(1), k)
    for _ in range(n_rounds):
        improved = False
        for j in range(k):
            cur = w[j]
            for g in GRID:
                w[j] = g
                s = _fast_f1_macro(y, (proba * w).argmax(1), k)
                if s > best + 1e-9:
                    best, cur, improved = s, g, True
            w[j] = cur
        if not improved:
            break
    return w / w.mean()


def apply_class_multipliers(proba: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        return proba
    p = proba * w[None, :]
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def postprocess(spec, pp_cfg: dict, y: np.ndarray, oof_raw: np.ndarray, test_raw: np.ndarray,
                y_range: tuple[float, float]):
    """Возвращает (oof, test, params). Параметры подбираются только на OOF."""
    params: dict = {}
    oof, test = oof_raw, test_raw
    if spec.is_clf:
        if pp_cfg.get("class_weights_opt"):
            w = fit_class_multipliers(y, oof_raw)
            params["class_multipliers"] = [round(float(v), 4) for v in w]
            oof, test = apply_class_multipliers(oof_raw, w), apply_class_multipliers(test_raw, w)
    else:
        if pp_cfg.get("clip_to_train_range"):
            lo, hi = y_range
            oof, test = np.clip(oof_raw, lo, hi), np.clip(test_raw, lo, hi)
            params["clip"] = [float(lo), float(hi)]
    return oof, test, params
