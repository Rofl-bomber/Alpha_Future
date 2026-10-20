"""Бленд топ-экспериментов по OOF: неотрицательные веса (scipy), сумма = 1.
Для классификации смешиваются вероятности; постобработка — поверх бленда."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .metrics import score as metric_score
from .postprocess import postprocess


@dataclass
class BlendResult:
    members: list[str]
    weights: list[float]
    oof_raw: np.ndarray
    test_raw: np.ndarray
    oof: np.ndarray
    test: np.ndarray
    fold_scores: list[float]
    score: float
    pp_params: dict = field(default_factory=dict)


def _loss(spec, y, pred):
    if spec.is_clf:
        p = np.clip(pred, 1e-9, 1.0)
        return float(-np.mean(np.log(p[np.arange(len(y)), y])))
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def blend(results: list, spec, data, folds, pp_cfg: dict) -> BlendResult:
    """results — ExpResult с сохранёнными oof_raw/test_raw (одни и те же фолды)."""
    y = data.y
    oofs = [r.oof_raw for r in results]
    tests = [r.test_raw for r in results]
    m = len(results)

    def mix(w, arrs):
        return sum(wi * a for wi, a in zip(w, arrs))

    if m == 1:
        w = np.array([1.0])
    else:
        x0 = np.full(m, 1.0 / m)
        opt = minimize(lambda w: _loss(spec, y, mix(w, oofs)), x0, method="SLSQP",
                       bounds=[(0.0, 1.0)] * m, constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                       options={"maxiter": 200})
        w = np.clip(opt.x if opt.success else x0, 0, None)
        w = w / w.sum() if w.sum() > 0 else x0
    oof_raw, test_raw = mix(w, oofs), mix(w, tests)
    y_range = (float(np.min(y)), float(np.max(y))) if not spec.is_clf else (0.0, 0.0)
    oof, test, params = postprocess(spec, pp_cfg, y, oof_raw, test_raw, y_range)
    fs = [round(float(metric_score(spec.metric, y[va], oof[va], spec.n_classes or None)), 6) for _, va in folds]
    return BlendResult(members=[r.exp_id for r in results], weights=[round(float(v), 4) for v in w],
                       oof_raw=oof_raw, test_raw=test_raw, oof=oof, test=test,
                       fold_scores=fs, score=float(np.mean(fs)), pp_params=params)
