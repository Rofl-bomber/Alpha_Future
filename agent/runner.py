"""run_experiment: конфигурация -> фолды -> скоры, OOF, предсказания test, время, ошибка.

Честная оценка (ТЗ, блок 4):
- early stopping идёт по внутреннему сплиту обучающей части фолда (cv.inner_split),
  внешний валидационный фолд модель не видит;
- постобработка с подбором параметров — cross-fit (postprocess.postprocess_cv).
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cv import inner_split
from .features.builtin import build_feature_frame, target_encode
from .metrics import score as metric_score
from .models import get_model
from .postprocess import postprocess_cv


@dataclass
class ExpResult:
    exp_id: str
    config: dict
    fold_scores: list[float] = field(default_factory=list)
    score: float | None = None            # среднее по фолдам после cross-fit постобработки
    oof_score: float | None = None        # по всему OOF (cross-fit)
    raw_score: float | None = None        # среднее по фолдам до постобработки
    score_insample: float | None = None   # постобработка подобрана на всём OOF — только для диагностики
    oof_raw: np.ndarray | None = None     # в исходной шкале таргета, до постобработки
    test_raw: np.ndarray | None = None
    oof: np.ndarray | None = None         # после cross-fit постобработки
    test: np.ndarray | None = None
    pp_params: dict = field(default_factory=dict)
    time_s: float = 0.0
    error: str | None = None
    feature_names: list[str] = field(default_factory=list)
    importances: dict | None = None
    best_iters: list = field(default_factory=list)
    reused_from: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None


def _safe_names(cols) -> list[str]:
    out, seen = [], set()
    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) or "f"
        base, i = s, 1
        while s in seen:
            s = f"{base}_{i}"
            i += 1
        seen.add(s)
        out.append(s)
    return out


class _TargetTransform:
    def __init__(self, kind: str, y: np.ndarray):
        self.kind = kind
        self.shift = float(min(0.0, np.min(y))) if kind == "log1p" else 0.0

    def fwd(self, y):
        return np.log1p(np.asarray(y, dtype=float) - self.shift) if self.kind == "log1p" else y

    def inv(self, p):
        return np.expm1(p) + self.shift if self.kind == "log1p" else p


def _fold_scores(spec, y, pred, folds) -> list[float]:
    return [metric_score(spec.metric, y[va], pred[va], spec.n_classes or None) for _, va in folds]


def run_experiment(config: dict, spec, folds, data, registry=None, importances: dict | None = None,
                   n_threads: int = 1, seed: int = 42, exp_id: str = "", es_frac: float = 0.1) -> ExpResult:
    t0 = time.time()
    res = ExpResult(exp_id=exp_id, config=config)
    try:
        if spec.is_clf and config["target"]["transform"] != "none":
            raise ValueError("target.transform применим только к регрессии")
        ff = build_feature_frame(config, spec, data, registry, importances)
        X = ff.X
        n_tr = data.n_train
        y = data.y
        tt = _TargetTransform(config["target"]["transform"], y) if not spec.is_clf else None
        y_fit = tt.fwd(y) if tt else y
        X_te_full = X.iloc[n_tr:]
        k = spec.n_classes if spec.is_clf else 0
        oof = np.zeros((n_tr, k)) if spec.is_clf else np.zeros(n_tr)
        test = np.zeros((len(X_te_full), k)) if spec.is_clf else np.zeros(len(X_te_full))
        imps: dict[str, float] = {}
        mcfg = config["model"]
        for fold_k, (tr, va) in enumerate(folds):
            X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_te_full
            cats = list(ff.cat_cols)
            if ff.te_cols:
                enc_tr, (enc_va, enc_te) = target_encode(X_tr[ff.te_cols], y[tr], [X_va[ff.te_cols], X_te[ff.te_cols]],
                                                         ff.te_cols, spec, seed=seed)
                X_tr = pd.concat([X_tr, enc_tr], axis=1)
                X_va = pd.concat([X_va, enc_va], axis=1)
                X_te = pd.concat([X_te, enc_te], axis=1)
                if not ff.te_keep_original:
                    X_tr, X_va, X_te = (d.drop(columns=ff.te_cols) for d in (X_tr, X_va, X_te))
                    cats = [c for c in cats if c not in ff.te_cols]
            names = list(X_tr.columns)
            safe = _safe_names(names)
            ren = dict(zip(names, safe))
            X_tr, X_va, X_te = (d.rename(columns=ren) for d in (X_tr, X_va, X_te))
            cats_safe = [ren[c] for c in cats]
            model = get_model(mcfg["name"], mcfg["params"], spec, n_threads=n_threads, seed=seed)
            y_tr = y_fit[tr]
            if getattr(model, "uses_eval", True):
                # early stopping по внутреннему сплиту обучающей части: внешний фолд va модель не видит
                p_fit, p_es = inner_split(tr, y, folds, es_frac, seed + fold_k, spec.is_clf)
                model.fit(X_tr.iloc[p_fit], y_tr[p_fit], X_tr.iloc[p_es], y_tr[p_es], cats_safe)
            else:
                model.fit(X_tr, y_tr, X_tr.iloc[:0], y_tr[:0], cats_safe)
            p_va = model.predict(X_va)
            p_te = model.predict(X_te)
            if tt:
                p_va, p_te = tt.inv(p_va), tt.inv(p_te)
            oof[va] = p_va
            test += p_te / len(folds)
            res.best_iters.append(model.best_iteration)
            if model.importances:
                back = dict(zip(safe, names))
                tot = sum(model.importances.values()) or 1.0
                for c, v in model.importances.items():
                    imps[back.get(c, c)] = imps.get(back.get(c, c), 0.0) + v / tot / len(folds)
            res.feature_names = names
        if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test)):
            raise ValueError("модель вернула NaN/inf в предсказаниях")
        res.oof_raw, res.test_raw = oof, test
        res.importances = imps or None
        res.raw_score = float(np.mean(_fold_scores(spec, y, oof, folds)))
        _finish(res, spec, folds, data)
    except Exception as e:  # прототип: только try/except, без отдельного процесса
        res.error = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=6)[-2500:]
    res.time_s = round(time.time() - t0, 2)
    return res


def _covered(folds, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for _, va in folds:
        m[va] = True
    return m


def _finish(res: ExpResult, spec, folds, data):
    y = data.y
    y_range = (float(np.min(y)), float(np.max(y))) if not spec.is_clf else (0.0, 0.0)
    oof, test, params, oof_in = postprocess_cv(spec, res.config["postprocess"], y, res.oof_raw, res.test_raw,
                                               y_range, folds)
    res.oof, res.test, res.pp_params = oof, test, params
    res.fold_scores = [round(float(s), 6) for s in _fold_scores(spec, y, oof, folds)]
    res.score = float(np.mean(res.fold_scores))
    res.score_insample = float(np.mean(_fold_scores(spec, y, oof_in, folds)))
    cov = _covered(folds, len(y))
    res.oof_score = float(metric_score(spec.metric, y[cov], oof[cov], spec.n_classes or None))


def rerun_postprocess(base: ExpResult, config: dict, spec, folds, data, exp_id: str = "") -> ExpResult:
    """Гипотезы, меняющие только postprocess, не требуют переобучения: берём сырые OOF/test базы."""
    t0 = time.time()
    res = ExpResult(exp_id=exp_id, config=config, oof_raw=base.oof_raw, test_raw=base.test_raw,
                    raw_score=base.raw_score, feature_names=base.feature_names, importances=base.importances,
                    best_iters=base.best_iters, reused_from=base.exp_id)
    try:
        _finish(res, spec, folds, data)
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=6)[-2500:]
    res.time_s = round(time.time() - t0, 2)
    return res
