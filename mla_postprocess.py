"""Metric-aware decision rules fitted on OOF predictions (thresholds, class weights, temperature, clipping)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

import mla_metrics as MT
from mla_utils import LOG

FAST_LABEL = {"accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted", "precision", "recall",
              "precision_macro", "recall_macro", "mcc", "kappa", "qwk"}


def confusion(y, p, K, sw=None):
    return np.bincount(y.astype(np.int64) * K + p.astype(np.int64), weights=sw, minlength=K * K).reshape(K, K).astype(float)


def fast_label_score(metric: str, y: np.ndarray, pred: np.ndarray, K: int, sw=None) -> float:
    """Label metrics from a (optionally sample-weighted) confusion matrix."""
    if metric not in FAST_LABEL:
        return MT.get(metric).fn(y, pred)
    C = confusion(y, pred, K, sw)
    tp = np.diag(C)
    support = C.sum(1)
    predc = C.sum(0)
    n = C.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predc > 0, tp / predc, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    if metric == "accuracy" or metric == "f1_micro":
        return float(tp.sum() / n)
    if metric == "balanced_accuracy":
        present = support > 0
        return float(rec[present].mean())
    if metric == "f1_macro":
        # sklearn macro: average over labels present in y_true or y_pred
        present = (support > 0) | (predc > 0)
        return float(f1[present].mean())
    if metric == "f1_weighted":
        return float((f1 * support).sum() / max(1.0, support.sum()))
    if metric == "precision_macro":
        present = (support > 0) | (predc > 0)
        return float(prec[present].mean())
    if metric == "recall_macro":
        present = (support > 0) | (predc > 0)
        return float(rec[present].mean())
    if metric == "f1":
        return float(f1[1]) if K > 1 else 0.0
    if metric == "precision":
        return float(prec[1])
    if metric == "recall":
        return float(rec[1])
    if metric == "mcc":
        t = support
        p = predc
        c = tp.sum()
        s = n
        num = c * s - (t * p).sum()
        den = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
        return float(num / den) if den > 0 else 0.0
    if metric in ("kappa", "qwk"):
        if metric == "kappa":
            W = 1 - np.eye(K)
        else:
            idx = np.arange(K)
            W = (idx[:, None] - idx[None, :]) ** 2 / max(1, (K - 1) ** 2)
        E = np.outer(support, predc) / max(1.0, n)
        den = (W * E).sum()
        return float(1 - (W * C).sum() / den) if den > 0 else 0.0
    return MT.get(metric).fn(y, pred)


@dataclass
class Decision:
    kind: str = "none"  # none | threshold | weights | temperature | regression
    params: dict = field(default_factory=dict)
    gain: float = 0.0

    def transform_proba(self, P: np.ndarray) -> np.ndarray:
        if self.kind == "temperature":
            T = self.params["T"]
            L = np.log(np.clip(P, 1e-12, 1.0)) / T
            L -= L.max(1, keepdims=True)
            E = np.exp(L)
            return E / E.sum(1, keepdims=True)
        return P

    def decision_arg(self):
        if self.kind == "threshold":
            return self.params["t"]
        if self.kind == "weights":
            return np.asarray(self.params["w"])
        return None

    def labels(self, P: np.ndarray) -> np.ndarray:
        return MT.to_labels(self.transform_proba(P), self.decision_arg())

    def apply_regression(self, p: np.ndarray) -> np.ndarray:
        if self.kind != "regression":
            return p
        q = p * self.params.get("a", 1.0) + self.params.get("b", 0.0)
        lo, hi = self.params.get("lo"), self.params.get("hi")
        if lo is not None:
            q = np.maximum(q, lo)
        if hi is not None:
            q = np.minimum(q, hi)
        return q

    def to_dict(self):
        return {"kind": self.kind, "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.params.items()}, "gain": self.gain}


def metric_with_decision(metric: str, y, pred, decision: Decision | None, K: int, sw=None) -> float:
    m = MT.get(metric)
    if m.kind == "regression":
        p = decision.apply_regression(pred) if decision else pred
        if sw is not None and metric in ("rmse", "mse", "mae"):
            e = (np.asarray(y, float) - np.asarray(p, float))
            v = np.average(e ** 2 if metric != "mae" else np.abs(e), weights=sw)
            return float(np.sqrt(v)) if metric == "rmse" else float(v)
        return m.fn(y, p)
    P = decision.transform_proba(pred) if decision else pred
    if m.needs == "label":
        lab = MT.to_labels(P, decision.decision_arg() if decision else None)
        return fast_label_score(metric, y, lab, K, sw)
    return MT.score(metric, y, P)


# ---------------------------------------------------------------------------------------------
def _fit_threshold(metric, y, P, sw=None):
    p1 = P[:, 1]
    qs = np.unique(np.quantile(p1, np.linspace(0.005, 0.995, 300)))
    cand = np.unique(np.r_[qs, 0.5])
    best_t, best = 0.5, -np.inf
    for t in cand:
        s = MT.utility(metric, fast_label_score(metric, y, (p1 >= t).astype(int), 2, sw))
        if s > best:
            best, best_t = s, float(t)
    return {"t": best_t}


def _fit_weights(metric, y, P, n_pass=3, sw=None):
    """Coordinate ascent on per-class multiplicative weights for argmax(P * w).

    For class k only rows where k competes with the best other class can change, so each grid point costs O(N)."""
    K = P.shape[1]
    w = np.ones(K)
    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 15))
    cur_lab = (P * w).argmax(1)
    base = MT.utility(metric, fast_label_score(metric, y, cur_lab, K, sw))
    for _ in range(n_pass):
        improved = False
        for k in range(K):
            S = P * w
            Sk = S[:, k].copy()
            S[:, k] = -np.inf
            other_lab = S.argmax(1)
            other_max = S[np.arange(len(S)), other_lab]
            best_f, best_s, best_lab = 1.0, base, cur_lab
            for f in grid:
                if f == 1.0:
                    continue
                lab = np.where(Sk * f > other_max, k, other_lab)
                sc = MT.utility(metric, fast_label_score(metric, y, lab, K, sw))
                if sc > best_s + 1e-12:
                    best_s, best_f, best_lab = sc, f, lab
            if best_f != 1.0:
                w[k] *= best_f
                base, cur_lab = best_s, best_lab
                improved = True
        if not improved:
            break
    return {"w": (w / w.max()).tolist()}


def _fit_temperature(metric, y, P):
    def f(T):
        L = np.log(np.clip(P, 1e-12, 1)) / T
        L -= L.max(1, keepdims=True)
        E = np.exp(L)
        Q = E / E.sum(1, keepdims=True)
        return -MT.utility(metric, MT.score(metric, y, Q))
    r = minimize_scalar(f, bounds=(0.4, 3.0), method="bounded", options={"xatol": 1e-3})
    return {"T": float(r.x)}


def _fit_regression(metric, y, p, lo, hi):
    best = {"a": 1.0, "b": 0.0, "lo": lo, "hi": hi}
    if metric in ("rmse", "mse", "r2"):
        A = np.c_[p, np.ones_like(p)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if 0.7 < a < 1.3:
            best.update(a=a, b=b)
    elif metric in ("mae", "medae"):
        b = float(np.median(y - p))
        best.update(b=b)
    return best


def _dec_score(metric, y, pred, kind, params, K, sw=None):
    d = Decision(kind, params)
    return MT.utility(metric, metric_with_decision(metric, y, pred, d, K, sw))


def fit_decision(metric: str, task_type: str, y: np.ndarray, pred: np.ndarray, n_classes: int = 0,
                 y_range: tuple | None = None, nonneg: bool = False, seed: int = 0, validate: bool = True,
                 sw: np.ndarray | None = None) -> Decision:
    """Fit a decision rule on OOF predictions; keep it only if it helps on held-out halves."""
    m = MT.get(metric)
    K = n_classes
    try:
        if m.kind == "regression":
            lo, hi = (y_range if y_range else (None, None))
            if nonneg:
                lo = 0.0 if lo is None else max(lo, 0.0)
            fitter = lambda yy, pp, ww=None: _fit_regression(metric, yy, pp, lo, hi)  # noqa
            kind = "regression"
            base_params = {"a": 1.0, "b": 0.0, "lo": lo, "hi": hi}
        elif m.needs == "label":
            if task_type == "binary":
                fitter, kind, base_params = (lambda yy, pp, ww=None: _fit_threshold(metric, yy, pp, ww)), "threshold", {"t": 0.5}
            else:
                fitter, kind, base_params = (lambda yy, pp, ww=None: _fit_weights(metric, yy, pp, sw=ww)), "weights", {"w": [1.0] * K}
        elif metric in ("logloss", "brier"):
            fitter, kind, base_params = (lambda yy, pp, ww=None: _fit_temperature(metric, yy, pp)), "temperature", {"T": 1.0}
        else:
            return Decision("none")
        full = fitter(y, pred, sw)
        if not validate:
            return Decision(kind, full)
        # held-out check: 3 random 50/50 splits, fit on one half, evaluate on the other
        rng = np.random.RandomState(seed)
        gains = []
        n = len(y)
        for _ in range(2):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            for f_idx, e_idx in ((a, b), (b, a)):
                wf = sw[f_idx] if sw is not None else None
                we = sw[e_idx] if sw is not None else None
                prm = fitter(y[f_idx], pred[f_idx], wf)
                g = _dec_score(metric, y[e_idx], pred[e_idx], kind, prm, K, we) - _dec_score(metric, y[e_idx], pred[e_idx], kind, base_params, K, we)
                gains.append(g)
        gain = float(np.mean(gains))
        if kind == "regression":
            # clipping is always safe; affine part only if it helps
            if gain <= 0:
                full = dict(base_params)
            return Decision("regression", full, gain)
        if gain > 0:
            return Decision(kind, full, gain)
        return Decision(kind, base_params, gain) if kind in ("threshold", "weights") else Decision("none", {}, gain)
    except Exception as e:
        LOG.warning(f"fit_decision failed: {e}")
        return Decision("none")
