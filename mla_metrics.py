"""Metric registry.

Internal prediction representation
  * regression     : 1-D float array
  * classification : N x K probability matrix, classes encoded 0..K-1.
                     For binary tasks index 1 is ALWAYS the positive class.
Label metrics convert probabilities to labels with a `decision` (threshold for binary,
per-class weight vector for multiclass) or argmax when no decision is given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm

EPS = 1e-15


@dataclass(frozen=True)
class MetricDef:
    name: str
    kind: str  # regression | classification
    greater_is_better: bool
    needs: str  # value | label | proba
    fn: Callable
    binary_only: bool = False
    title: str = ""


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _mse(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _medae(y, p):
    return float(np.median(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _rmsle(y, p):
    y = np.clip(np.asarray(y, float), 0, None)
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def _msle(y, p):
    return _rmsle(y, p) ** 2


def _mape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), np.finfo(float).eps)))


def _smape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    r = np.where(den == 0, 0.0, 2.0 * np.abs(y - p) / np.where(den == 0, 1, den))
    return float(np.mean(r))


def _r2(y, p):
    return float(skm.r2_score(y, p))


def _norm(P):
    P = np.clip(np.asarray(P, float), EPS, None)
    return P / P.sum(1, keepdims=True)


def _auc(y, P):
    P = np.asarray(P, float)
    K = P.shape[1]
    if K == 2:
        return float(skm.roc_auc_score(y, P[:, 1]))
    present = np.unique(y)
    if len(present) < K:  # some class absent in y: macro over present classes (ovr)
        scores = [skm.roc_auc_score((y == k).astype(int), P[:, k]) for k in present if 0 < (y == k).sum() < len(y)]
        return float(np.mean(scores)) if scores else 0.5
    return float(skm.roc_auc_score(y, _norm(P), multi_class="ovr", average="macro", labels=list(range(K))))


def _auc_ovo(y, P):
    P = np.asarray(P, float)
    if P.shape[1] == 2:
        return float(skm.roc_auc_score(y, P[:, 1]))
    return float(skm.roc_auc_score(y, _norm(P), multi_class="ovo", average="macro", labels=list(range(P.shape[1]))))


def _gini(y, P):
    return 2.0 * _auc(y, P) - 1.0


def _ap(y, P):
    P = np.asarray(P, float)
    if P.shape[1] == 2:
        return float(skm.average_precision_score(y, P[:, 1]))
    Y = np.eye(P.shape[1])[y]
    return float(skm.average_precision_score(Y, P, average="macro"))


def _logloss(y, P):
    P = _norm(P)
    return float(skm.log_loss(y, P, labels=list(range(P.shape[1]))))


def _brier(y, P):
    P = np.asarray(P, float)
    if P.shape[1] == 2:
        return float(np.mean((P[:, 1] - y) ** 2))
    Y = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - Y) ** 2, 1)))


def _lab(fn):
    return fn


REGISTRY: dict[str, MetricDef] = {}


def _reg(m: MetricDef):
    REGISTRY[m.name] = m


# regression
_reg(MetricDef("rmse", "regression", False, "value", _rmse, title="Root mean squared error"))
_reg(MetricDef("mse", "regression", False, "value", _mse, title="Mean squared error"))
_reg(MetricDef("mae", "regression", False, "value", _mae, title="Mean absolute error"))
_reg(MetricDef("medae", "regression", False, "value", _medae, title="Median absolute error"))
_reg(MetricDef("rmsle", "regression", False, "value", _rmsle, title="Root mean squared log error"))
_reg(MetricDef("msle", "regression", False, "value", _msle, title="Mean squared log error"))
_reg(MetricDef("mape", "regression", False, "value", _mape, title="Mean absolute percentage error"))
_reg(MetricDef("smape", "regression", False, "value", _smape, title="Symmetric MAPE"))
_reg(MetricDef("r2", "regression", True, "value", _r2, title="R squared"))
# classification - labels
_reg(MetricDef("accuracy", "classification", True, "label", lambda y, l: float(skm.accuracy_score(y, l)), title="Accuracy"))
_reg(MetricDef("balanced_accuracy", "classification", True, "label", lambda y, l: float(skm.balanced_accuracy_score(y, l)), title="Balanced accuracy"))
_reg(MetricDef("f1", "classification", True, "label", lambda y, l: float(skm.f1_score(y, l, pos_label=1, average="binary", zero_division=0)), binary_only=True, title="F1 (binary, positive class)"))
_reg(MetricDef("f1_macro", "classification", True, "label", lambda y, l: float(skm.f1_score(y, l, average="macro", zero_division=0)), title="Macro F1"))
_reg(MetricDef("f1_micro", "classification", True, "label", lambda y, l: float(skm.f1_score(y, l, average="micro", zero_division=0)), title="Micro F1"))
_reg(MetricDef("f1_weighted", "classification", True, "label", lambda y, l: float(skm.f1_score(y, l, average="weighted", zero_division=0)), title="Weighted F1"))
_reg(MetricDef("precision", "classification", True, "label", lambda y, l: float(skm.precision_score(y, l, pos_label=1, zero_division=0)), binary_only=True, title="Precision (binary)"))
_reg(MetricDef("recall", "classification", True, "label", lambda y, l: float(skm.recall_score(y, l, pos_label=1, zero_division=0)), binary_only=True, title="Recall (binary)"))
_reg(MetricDef("precision_macro", "classification", True, "label", lambda y, l: float(skm.precision_score(y, l, average="macro", zero_division=0)), title="Macro precision"))
_reg(MetricDef("recall_macro", "classification", True, "label", lambda y, l: float(skm.recall_score(y, l, average="macro", zero_division=0)), title="Macro recall"))
_reg(MetricDef("mcc", "classification", True, "label", lambda y, l: float(skm.matthews_corrcoef(y, l)), title="Matthews correlation"))
_reg(MetricDef("kappa", "classification", True, "label", lambda y, l: float(skm.cohen_kappa_score(y, l)), title="Cohen kappa"))
_reg(MetricDef("qwk", "classification", True, "label", lambda y, l: float(skm.cohen_kappa_score(y, l, weights="quadratic")), title="Quadratic weighted kappa"))
# classification - probabilities
_reg(MetricDef("roc_auc", "classification", True, "proba", _auc, title="ROC AUC (binary / macro OVR)"))
_reg(MetricDef("roc_auc_ovo", "classification", True, "proba", _auc_ovo, title="ROC AUC (macro OVO)"))
_reg(MetricDef("gini", "classification", True, "proba", _gini, title="Gini = 2*AUC-1"))
_reg(MetricDef("pr_auc", "classification", True, "proba", _ap, title="Average precision / PR AUC"))
_reg(MetricDef("logloss", "classification", False, "proba", _logloss, title="Log loss / cross-entropy"))
_reg(MetricDef("brier", "classification", False, "proba", _brier, title="Brier score"))


def get(name: str) -> MetricDef:
    return REGISTRY[name]


def to_labels(P: np.ndarray, decision=None) -> np.ndarray:
    P = np.asarray(P, float)
    if P.ndim == 1:
        P = np.stack([1 - P, P], 1)
    if decision is None:
        return P.argmax(1)
    if np.isscalar(decision) or (isinstance(decision, np.ndarray) and decision.ndim == 0):
        return (P[:, 1] >= float(decision)).astype(int)
    w = np.asarray(decision, float)
    return (P * w[None, :]).argmax(1)


def score(name: str, y: np.ndarray, pred: np.ndarray, decision=None) -> float:
    m = REGISTRY[name]
    if m.kind == "regression":
        return m.fn(y, pred)
    if m.needs == "label":
        return m.fn(y, to_labels(pred, decision))
    P = np.asarray(pred, float)
    if P.ndim == 1:
        P = np.stack([1 - P, P], 1)
    return m.fn(y, P)


def utility(name: str, value: float) -> float:
    """Direction-adjusted value: higher is always better."""
    return value if REGISTRY[name].greater_is_better else -value


def safe_score(name, y, pred, decision=None) -> float:
    try:
        v = score(name, y, pred, decision)
        if v is None or not np.isfinite(v):
            raise ValueError
        return v
    except Exception:
        return float("-inf") if REGISTRY[name].greater_is_better else float("inf")


# ---------------------------------------------------------------------------------------------
# Text parsing of metric names
# ---------------------------------------------------------------------------------------------
METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"macro[\s\-_]*(?:averaged\s*)?f[\s\-_]?1|f[\s\-_]?1[\s\-_]*(?:score)?[\s\-_]*\(?\s*macro|f1[\s\-_]*macro|макро[\s\-]*f1|f1[\s\-]*макро", "f1_macro"),
    (r"weighted[\s\-_]*f[\s\-_]?1|f[\s\-_]?1[\s\-_]*(?:score)?[\s\-_]*\(?\s*weighted|взвешенн\w*\s*f1", "f1_weighted"),
    (r"micro[\s\-_]*f[\s\-_]?1|f[\s\-_]?1[\s\-_]*(?:score)?[\s\-_]*\(?\s*micro|микро[\s\-]*f1", "f1_micro"),
    (r"quadratic[\s\-_]*weighted[\s\-_]*kappa|\bqwk\b|квадратичн\w*\s*взвешенн\w*\s*капп", "qwk"),
    (r"cohen'?s?[\s\-_]*kappa|\bkappa\b|\bкапп[аы]\b", "kappa"),
    (r"\bmcc\b|matthews", "mcc"),
    (r"balanced[\s\-_]*accuracy|сбалансированн\w*\s*(?:точност|accuracy)", "balanced_accuracy"),
    (r"\brmsle\b|root[\s\-_]*mean[\s\-_]*squared[\s\-_]*log(?:arithmic)?[\s\-_]*error", "rmsle"),
    (r"\bmsle\b|mean[\s\-_]*squared[\s\-_]*log(?:arithmic)?[\s\-_]*error", "msle"),
    (r"\brmse\b|root[\s\-_]*mean[\s\-_]*squared[\s\-_]*error|корень\s+из\s+средн\w*\s*квадрат", "rmse"),
    (r"\bsmape\b|symmetric[\s\-_]*mean[\s\-_]*absolute[\s\-_]*percentage", "smape"),
    (r"\bmape\b|mean[\s\-_]*absolute[\s\-_]*percentage[\s\-_]*error|средн\w*\s*абсолютн\w*\s*процентн", "mape"),
    (r"\bmedae\b|median[\s\-_]*absolute[\s\-_]*error|медианн\w*\s*абсолютн", "medae"),
    (r"\bmae\b|mean[\s\-_]*absolute[\s\-_]*error|средн\w*\s*абсолютн\w*\s*ошибк", "mae"),
    (r"\bmse\b|mean[\s\-_]*squared[\s\-_]*error|средн\w*\s*квадратичн\w*\s*ошибк|среднеквадратичн\w*\s*ошибк", "mse"),
    (r"\br\s*\^?\s*2\b|\br²|r[\s\-_]*squared|коэффициент\w*\s*детерминац", "r2"),
    (r"pr[\s\-_]*auc|average[\s\-_]*precision|precision[\s\-_]*recall[\s\-_]*auc", "pr_auc"),
    (r"\bgini\b|джини", "gini"),
    (r"roc[\s\-_]*auc|\bauc\b|area\s+under\s+(?:the\s+)?(?:roc\s+)?curve|площад\w*\s*под\s*(?:roc[\s\-]*)?крив", "roc_auc"),
    (r"log[\s\-_]*loss|logloss|cross[\s\-_]*entropy|кросс[\s\-]*энтропи|логарифмическ\w*\s*(?:функци\w*\s*)?потер|log[\s\-_]*likelihood", "logloss"),
    (r"\bbrier\b", "brier"),
    (r"\baccuracy\b|доля\s+(?:верн|правильн)\w*\s*(?:ответ|предсказ|классиф)|точность\s+классификац", "accuracy"),
    (r"\bf[\s\-_]?1(?:[\s\-_]*score)?\b|f[\s\-]?measure|f[\s\-]?мера", "f1"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b|\bполнот[аы]\b", "recall"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), k) for p, k in METRIC_PATTERNS]
_MAIN_LINE = re.compile(r"(основн\w*\s*метрик|main\s+metric|primary\s+metric|evaluation\s+metric|метрик\w*\s*(?:качества|оценки)?\s*[:\-—]|metric\s*[:\-—]|оценива\w*|evaluated|scored|scoring|leaderboard|лидерборд)", re.IGNORECASE)
_SECONDARY = re.compile(r"(справочн|reference|informational|for\s+information|дополнительн|не\s+учитыва)", re.IGNORECASE)


def find_metrics_in_text(text: str) -> list[tuple[str, int, int]]:
    """Return [(metric, line_priority, position)] sorted by priority (lower = stronger)."""
    out = []
    if not text:
        return out
    pos = 0
    for line in text.splitlines():
        prio = 2
        if _MAIN_LINE.search(line):
            prio = 0
        if re.search(r"основн|main|primary", line, re.IGNORECASE):
            prio = -1
        for rx, key in _COMPILED:
            for m in rx.finditer(line):
                # mentions in a 'reference only' clause after the main metric get demoted
                tail = line[: m.start()]
                p = prio
                if _SECONDARY.search(line[m.start(): m.start() + 80]) or (_SECONDARY.search(tail) and prio <= 0):
                    p = 3
                out.append((key, p, pos + m.start()))
        pos += len(line) + 1
    # remove generic matches overlapped by specific ones at same position (e.g. 'f1' inside 'macro f1')
    out.sort(key=lambda t: (t[1], t[2]))
    cleaned = []
    for k, p, s in out:
        if k in ("f1", "precision", "recall", "roc_auc") and any(abs(s - s2) <= 12 and k2 != k for k2, p2, s2 in out if k2 in ("f1_macro", "f1_micro", "f1_weighted", "pr_auc", "precision_macro", "recall_macro", "gini")):
            continue
        cleaned.append((k, p, s))
    return cleaned


def guess_metric_from_text(text: str, kind: str | None = None, n_classes: int | None = None) -> str | None:
    cands = find_metrics_in_text(text)
    for k, p, s in cands:
        m = REGISTRY.get(k)
        if m is None:
            continue
        if kind and m.kind != kind:
            continue
        if k == "f1" and n_classes and n_classes > 2:
            return "f1_macro"
        if k in ("precision", "recall") and n_classes and n_classes > 2:
            return k + "_macro"
        return k
    return None


def default_metric(kind: str, n_classes: int | None = None, proba_expected: bool = False) -> str:
    if kind == "regression":
        return "rmse"
    if n_classes == 2:
        return "roc_auc" if proba_expected else "accuracy"
    return "logloss" if proba_expected else "accuracy"


def register_custom_metric(code: str, kind: str) -> str | None:
    """Validate LLM-written `def metric(y_true, y_pred)` on toy data and register it as 'custom'.
    Regression: y_pred is 1-D. Classification: y_pred is an (n, K) probability matrix."""
    import math
    ns: dict = {"np": np, "math": math}
    try:
        exec(compile(code, "<custom_metric>", "exec"), ns)
        fn = ns["metric"]
        gib = bool(ns.get("GREATER_IS_BETTER", True))
        rng = np.random.default_rng(0)
        if kind == "regression":
            y = np.abs(rng.normal(10, 3, 400)) + 1
            good = float(fn(y, y * (1 + rng.normal(0, 0.01, 400))))
            bad = float(fn(y, rng.permutation(y)))
            needs = "value"
        else:
            K = 3
            y = rng.integers(0, K, 400)
            Pg = np.full((400, K), 0.05)
            Pg[np.arange(400), y] = 0.9
            Pb = rng.dirichlet(np.ones(K), 400)
            good, bad = float(fn(y, Pg)), float(fn(y, Pb))
            needs = "proba"
        if not (np.isfinite(good) and np.isfinite(bad)) or good == bad:
            return None
        if (good > bad) != gib:
            gib = good > bad  # trust the behaviour over the flag
        REGISTRY["custom"] = MetricDef("custom", kind, gib, needs, lambda yt, yp: float(fn(np.asarray(yt), np.asarray(yp))),
                                       title="custom metric from task description")
        return "custom"
    except Exception:
        return None
