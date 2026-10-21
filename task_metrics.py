"""Реестр метрик и преобразования предсказаний.

Внутреннее представление предсказаний (одинаковое во всех модулях):
  regression        — (n,) значения;   multi_regression — (n, T)
  binary            — (n,) вероятность положительного класса (индекс 1)
  multiclass        — (n, K) вероятности классов в порядке spec.classes
  multilabel        — (n, L) вероятности каждой метки
Истинные значения: регрессия — float, классификация — индексы классов,
multilabel — матрица 0/1.

Постобработка (post) — словарь параметров, подобранных на OOF:
  clip=[lo, hi]        — регрессия: обрезка в диапазон train
  class_w=[...]        — multiclass: множители вероятностей перед argmax
  thr=t / thr=[...]    — binary / multilabel: порог(и)
  prior=[...], train_prior=[...] — коррекция сдвига априорных вероятностей (только test)
"""

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

REG_TYPES = ("regression", "multi_regression")
CLS_TYPES = ("binary", "multiclass", "multilabel")


@dataclass
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    need: str  # value | label | proba
    kinds: tuple
    objective: str = "l2"  # подсказка для функции потерь модели: l2 | l1 | None
    target_transform: str = "none"


def _w(w):
    return None if w is None else np.asarray(w, dtype=float)


def _flat(y, p, w):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.ndim == 2:
        if w is not None:
            w = np.repeat(np.asarray(w, dtype=float)[:, None], y.shape[1], axis=1).ravel()
        return y.ravel(), p.ravel(), w
    return y, p.reshape(y.shape), w


def _rmse(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def _mse(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    return float(np.average((y - p) ** 2, weights=w))


def _mae(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    return float(np.average(np.abs(y - p), weights=w))


def _medae(y, p, w=None):
    y, p, _ = _flat(y, p, None)
    return float(np.median(np.abs(y - p)))


def _rmsle(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    return float(np.sqrt(np.average((np.log1p(np.maximum(y, 0)) - np.log1p(np.maximum(p, 0))) ** 2, weights=w)))


def _mape(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    return float(np.average(np.abs(y - p) / np.maximum(np.abs(y), 1e-9), weights=w))


def _smape(y, p, w=None):
    y, p, w = _flat(y, p, _w(w))
    denom = np.maximum((np.abs(y) + np.abs(p)) / 2, 1e-9)
    return float(np.average(np.abs(y - p) / denom, weights=w))


def _r2(y, p, w=None):
    from sklearn.metrics import r2_score

    return float(r2_score(y, p, sample_weight=_w(w)))


def _f1(average):
    def fn(y, lab, w=None):
        from sklearn.metrics import f1_score

        return float(f1_score(y, lab, average=average, sample_weight=_w(w), zero_division=0))

    return fn


def _accuracy(y, lab, w=None):
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y, lab, sample_weight=_w(w)))


def _balanced_accuracy(y, lab, w=None):
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, lab, sample_weight=_w(w)))


def _mcc(y, lab, w=None):
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y, lab, sample_weight=_w(w)))


def _qwk(y, lab, w=None):
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y, lab, weights="quadratic", sample_weight=_w(w)))


def _precision(y, lab, w=None):
    from sklearn.metrics import precision_score

    return float(precision_score(y, lab, sample_weight=_w(w), zero_division=0))


def _recall(y, lab, w=None):
    from sklearn.metrics import recall_score

    return float(recall_score(y, lab, sample_weight=_w(w), zero_division=0))


def _roc_auc(y, proba, w=None):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        if len(np.unique(y)) < 2:
            return 0.5
        return float(roc_auc_score(y, proba, sample_weight=_w(w)))
    if y.ndim == 2:  # multilabel
        scores = [roc_auc_score(y[:, j], proba[:, j], sample_weight=_w(w))
                  for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]
        return float(np.mean(scores)) if scores else 0.5
    scores = []
    for k in range(proba.shape[1]):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            scores.append(roc_auc_score(yk, proba[:, k], sample_weight=_w(w)))
    return float(np.mean(scores)) if scores else 0.5


def _gini(y, proba, w=None):
    return 2 * _roc_auc(y, proba, w) - 1


def _pr_auc(y, proba, w=None):
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return float(average_precision_score(y, proba, sample_weight=_w(w)))
    if y.ndim == 1:
        y = np.eye(proba.shape[1])[y]
    return float(average_precision_score(y, proba, average="macro", sample_weight=_w(w)))


def _logloss(y, proba, w=None):
    from sklearn.metrics import log_loss

    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    if proba.ndim == 1:
        return float(log_loss(y, np.c_[1 - proba, proba], labels=[0, 1], sample_weight=_w(w)))
    if y.ndim == 2:  # multilabel: средний бинарный logloss по меткам
        return float(np.mean([log_loss(y[:, j], np.c_[1 - proba[:, j], proba[:, j]], labels=[0, 1],
                                       sample_weight=_w(w)) for j in range(y.shape[1])]))
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(log_loss(y, proba, labels=list(range(proba.shape[1])), sample_weight=_w(w)))


_ALL = ("regression", "multi_regression", "binary", "multiclass", "multilabel")
_REG = ("regression", "multi_regression")

REGISTRY = {
    "rmse": Metric("rmse", _rmse, False, "value", _REG, "l2"),
    "mse": Metric("mse", _mse, False, "value", _REG, "l2"),
    "mae": Metric("mae", _mae, False, "value", _REG, "l1"),
    "medae": Metric("medae", _medae, False, "value", _REG, "l1"),
    "rmsle": Metric("rmsle", _rmsle, False, "value", _REG, "l2", "log1p"),
    "mape": Metric("mape", _mape, False, "value", _REG, "l1"),
    "smape": Metric("smape", _smape, False, "value", _REG, "l1"),
    "r2": Metric("r2", _r2, True, "value", _REG, "l2"),
    "f1_macro": Metric("f1_macro", _f1("macro"), True, "label", ("multiclass", "multilabel", "binary"), None),
    "f1_weighted": Metric("f1_weighted", _f1("weighted"), True, "label", ("multiclass", "multilabel", "binary"), None),
    "f1_micro": Metric("f1_micro", _f1("micro"), True, "label", ("multiclass", "multilabel", "binary"), None),
    "f1": Metric("f1", _f1("binary"), True, "label", ("binary",), None),
    "precision": Metric("precision", _precision, True, "label", ("binary",), None),
    "recall": Metric("recall", _recall, True, "label", ("binary",), None),
    "accuracy": Metric("accuracy", _accuracy, True, "label", ("binary", "multiclass", "multilabel"), None),
    "balanced_accuracy": Metric("balanced_accuracy", _balanced_accuracy, True, "label", ("binary", "multiclass"), None),
    "mcc": Metric("mcc", _mcc, True, "label", ("binary", "multiclass"), None),
    "qwk": Metric("qwk", _qwk, True, "label", ("binary", "multiclass"), None),
    "roc_auc": Metric("roc_auc", _roc_auc, True, "proba", ("binary", "multiclass", "multilabel"), None),
    "gini": Metric("gini", _gini, True, "proba", ("binary", "multiclass", "multilabel"), None),
    "pr_auc": Metric("pr_auc", _pr_auc, True, "proba", ("binary", "multiclass", "multilabel"), None),
    "logloss": Metric("logloss", _logloss, False, "proba", ("binary", "multiclass", "multilabel"), None),
}

# Порядок важен: сначала более специфичные названия.
ALIASES = [
    (r"macro[\s\-_]*(?:averaged\s+)?f1|f1[\s\-_]*macro|macro[\s\-_]*f[\s\-_]*score", "f1_macro"),
    (r"weighted[\s\-_]*f1|f1[\s\-_]*weighted", "f1_weighted"),
    (r"micro[\s\-_]*f1|f1[\s\-_]*micro", "f1_micro"),
    (r"\brmsle\b|root mean squared log|среднеквадратичн\w* логарифм", "rmsle"),
    (r"\brmse\b|root[\s\-]mean[\s\-]squared[\s\-]error|корень из среднеквадратичн", "rmse"),
    (r"\bsmape\b", "smape"),
    (r"\bmape\b|mean absolute percentage|средняя абсолютная процентная", "mape"),
    (r"\bmedae\b|median absolute error", "medae"),
    (r"\bmae\b|mean absolute error|средняя абсолютная ошибка", "mae"),
    (r"\bmse\b|mean squared error|среднеквадратичн\w* ошибк", "mse"),
    (r"\br2\b|\br²|r\^2|r-squared|коэффициент детерминации", "r2"),
    (r"pr[\s\-_]*auc|average[\s\-_]*precision|auprc", "pr_auc"),
    (r"roc[\s\-_]*auc|\bauc\b|area under", "roc_auc"),
    (r"\bgini\b|джини", "gini"),
    (r"log[\s\-_]*loss|cross[\s\-]*entropy|логлосс|логарифмическ\w* потер", "logloss"),
    (r"balanced[\s\-_]*accuracy", "balanced_accuracy"),
    (r"quadratic[\s\-_]*weighted[\s\-_]*kappa|\bqwk\b|kappa", "qwk"),
    (r"\bmcc\b|matthews", "mcc"),
    (r"\baccuracy\b|доля правильных ответов", "accuracy"),
    (r"\bf1\b|f1[\s\-_]*score|f-мер", "f1"),
]

_PRIMARY_MARKERS = [
    r"основн\w* метрик", r"метрик\w* оценки", r"метрик\w* качества", r"целев\w* метрик",
    r"primary metric", r"main metric", r"evaluation metric", r"evaluated (?:on|by|using)",
    r"оценивается по", r"метрика\s*[—:\-]", r"\*\*метрика", r"metric\s*[:—\-]",
]


def detect_metric(text: str):
    """Найти основную метрику в тексте условия. Возвращает (имя или None, уверенность)."""
    low = text.lower()
    best = None
    for marker in _PRIMARY_MARKERS:
        for m in re.finditer(marker, low):
            window = low[m.start(): m.start() + 300]
            found = _first_alias(window)
            if found and (best is None or m.start() < best[0]):
                best = (m.start(), found)
    if best:
        return best[1], "high"
    found = _first_alias(low)
    return (found, "low") if found else (None, "none")


def _first_alias(text: str):
    hits = []
    for pattern, name in ALIASES:
        m = re.search(pattern, text)
        if m:
            hits.append((m.start(), -len(m.group(0)), name))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def resolve(name: str, task_type: str) -> str:
    """Уточнить общее название под тип задачи (например, «F1» для мультикласса — macro)."""
    if name == "f1" and task_type != "binary":
        return "f1_macro"
    if name in ("precision", "recall") and task_type != "binary":
        return "f1_macro"
    return name


def default_metric(task_type: str) -> str:
    return {"regression": "rmse", "multi_regression": "rmse", "binary": "roc_auc",
            "multiclass": "f1_macro", "multilabel": "f1_macro"}.get(task_type, "rmse")


def get(name: str) -> Metric:
    return REGISTRY[name]


# ---------------------------------------------------------------- пользовательская метрика

_CUSTOM_CACHE = {}


def custom_metric(code: str, need: str, greater_is_better: bool) -> Metric:
    """Метрика, которую написала LLM: def metric(y_true, y_pred) -> float."""
    if code not in _CUSTOM_CACHE:
        ns = {"np": np}
        exec(compile(code, "<custom_metric>", "exec"), ns)
        fn = ns["metric"]

        def wrapped(y, p, w=None):
            return float(fn(np.asarray(y), np.asarray(p)))

        _CUSTOM_CACHE[code] = wrapped
    return Metric("custom", _CUSTOM_CACHE[code], greater_is_better, need, _ALL, "l2")


def metric_for(spec) -> Metric:
    if getattr(spec, "custom_metric_code", None):
        return custom_metric(spec.custom_metric_code, spec.metric_need, spec.greater_is_better)
    return REGISTRY[spec.metric]


# ---------------------------------------------------------------- представления предсказаний

def expand_proba(p, classes_present, n_classes):
    """Привести вероятности модели, обученной на части классов, к n_classes столбцам."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.c_[1 - p, p]
    out = np.zeros((p.shape[0], n_classes))
    for j, c in enumerate(classes_present):
        out[:, int(c)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def prior_adjust(p, train_prior, test_prior):
    """Коррекция сдвига априорных вероятностей (Saerens et al., 2002)."""
    p = np.asarray(p, dtype=float)
    binary = p.ndim == 1
    P = np.c_[1 - p, p] if binary else p
    ratio = np.asarray(test_prior, dtype=float) / np.maximum(np.asarray(train_prior, dtype=float), 1e-12)
    Q = P * ratio
    Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
    return Q[:, 1] if binary else Q


def em_priors(p, train_prior, iters=100, tol=1e-6):
    """Оценка долей классов в выборке без меток EM-алгоритмом по предсказанным вероятностям."""
    p = np.asarray(p, dtype=float)
    P = np.c_[1 - p, p] if p.ndim == 1 else p
    train_prior = np.maximum(np.asarray(train_prior, dtype=float), 1e-12)
    prior = train_prior.copy()
    for _ in range(iters):
        Q = P * (prior / train_prior)
        Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
        new = Q.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    return prior


def apply_post(pred, task_type, post, split="oof"):
    """Применить непрерывную часть постобработки (обрезку, коррекцию приоров для test)."""
    if not post:
        return pred
    p = np.asarray(pred, dtype=float)
    if task_type in REG_TYPES and post.get("clip") is not None:
        lo, hi = post["clip"]
        p = np.clip(p, lo, hi)
    if task_type in ("binary", "multiclass") and split == "test" and post.get("prior") is not None:
        p = prior_adjust(p, post["train_prior"], post["prior"])
    return p


def to_labels(pred, task_type, post=None):
    """Метки (индексы классов) из вероятностей с учётом порогов и множителей классов."""
    post = post or {}
    p = np.asarray(pred, dtype=float)
    if task_type == "binary":
        return (p >= float(post.get("thr", 0.5))).astype(int)
    if task_type == "multiclass":
        w = post.get("class_w")
        if w is not None:
            p = p * np.asarray(w, dtype=float)
        return p.argmax(axis=1)
    if task_type == "multilabel":
        thr = post.get("thr", 0.5)
        return (p >= np.asarray(thr, dtype=float)).astype(int)
    raise ValueError(task_type)


def to_needed(pred, task_type, need, post=None, split="oof"):
    p = apply_post(pred, task_type, post, split)
    if need == "label":
        return to_labels(p, task_type, post)
    if need == "proba" and task_type == "multiclass":
        p = np.asarray(p, dtype=float)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p


def score(metric: Metric, y_true, pred, task_type, post=None, sample_weight=None, split="oof") -> float:
    out = to_needed(pred, task_type, metric.need, post, split)
    return metric.fn(y_true, out, sample_weight)


def gain(value: float, greater_is_better: bool) -> float:
    """Значение в шкале «больше — лучше» для сравнений."""
    return value if greater_is_better else -value
