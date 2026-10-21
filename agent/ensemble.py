"""Бленд топ-экспериментов по OOF: неотрицательные веса (scipy), сумма = 1.
Для классификации смешиваются вероятности; постобработка — поверх бленда.

Cross-fit (ТЗ 4.3): веса для фолда k подбираются по OOF остальных фолдов, скор бленда — по cross-fit OOF;
постобработка вложена так же. Для test веса и постобработка подбираются на всём OOF."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .metrics import score as metric_score
from .postprocess import postprocess_cv


@dataclass
class BlendResult:
    members: list[str]
    weights: list[float]                  # веса на всём OOF (ими смешан test)
    oof_raw: np.ndarray                   # cross-fit бленд до постобработки
    test_raw: np.ndarray
    oof: np.ndarray                       # cross-fit бленд после cross-fit постобработки
    test: np.ndarray
    fold_scores: list[float]
    score: float
    score_insample: float | None = None   # веса и постобработка на всём OOF — только для диагностики
    fold_weights: list = field(default_factory=list)
    pp_params: dict = field(default_factory=dict)


def _loss(spec, y, pred):
    if spec.is_clf:
        p = np.clip(pred, 1e-9, 1.0)
        return float(-np.mean(np.log(p[np.arange(len(y)), y])))
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _mix(w, arrs):
    return sum(wi * a for wi, a in zip(w, arrs))


def _fit_weights(spec, y, oofs: list[np.ndarray]) -> np.ndarray:
    m = len(oofs)
    x0 = np.full(m, 1.0 / m)
    if m == 1:
        return np.array([1.0])
    opt = minimize(lambda w: _loss(spec, y, _mix(w, oofs)), x0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * m, constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   options={"maxiter": 200})
    w = np.clip(opt.x if opt.success else x0, 0, None)
    return w / w.sum() if w.sum() > 0 else x0


def blend(results: list, spec, data, folds, pp_cfg: dict) -> BlendResult:
    """results — ExpResult с сохранёнными oof_raw/test_raw (одни и те же фолды)."""
    y = data.y
    oofs = [r.oof_raw for r in results]
    tests = [r.test_raw for r in results]
    n = len(y)

    covered = np.zeros(n, dtype=bool)
    for _, va in folds:
        covered[va] = True
    w_full = _fit_weights(spec, y[covered], [o[covered] for o in oofs])
    oof_full = _mix(w_full, oofs)
    test_raw = _mix(w_full, tests)

    oof_cf = np.array(oof_full, dtype=float, copy=True)
    fold_w = []
    for _, va in folds:
        other = covered.copy()
        other[va] = False
        w_k = _fit_weights(spec, y[other], [o[other] for o in oofs]) if other.sum() else w_full
        oof_cf[va] = _mix(w_k, [o[va] for o in oofs])
        fold_w.append([round(float(v), 4) for v in w_k])

    y_range = (float(np.min(y)), float(np.max(y))) if not spec.is_clf else (0.0, 0.0)
    oof, test, params, oof_in = postprocess_cv(spec, pp_cfg, y, oof_cf, test_raw, y_range, folds,
                                               oof_for_test=oof_full)
    fs = [round(float(metric_score(spec.metric, y[va], oof[va], spec.n_classes or None)), 6) for _, va in folds]
    fs_in = [float(metric_score(spec.metric, y[va], oof_in[va], spec.n_classes or None)) for _, va in folds]
    return BlendResult(members=[r.exp_id for r in results], weights=[round(float(v), 4) for v in w_full],
                       oof_raw=oof_cf, test_raw=test_raw, oof=oof, test=test,
                       fold_scores=fs, score=float(np.mean(fs)), score_insample=float(np.mean(fs_in)),
                       fold_weights=fold_w, pp_params=params)
