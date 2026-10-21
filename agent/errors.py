"""Анализ ошибок текущего лучшего (ТЗ, блок 5).

Разбирает честный OOF (cross-fit, блок 4) лучшего эксперимента и показывает Исследователю,
где сосредоточена потеря. Главная величина — избыточный вклад сегмента: доля в потере минус доля строк.
У каждой строки отчёта стабильный id; гипотеза ссылается на него в target_segment, а харнесс после
эксперимента считает изменение метрики на этом сегменте (segment_score).

Виды id: class:<класс>, pair:<истинный>-><предсказанный>, cat:<колонка>=<значение>,
num:<колонка>:q1..q5|na, target:q1..q5, testlike:top25, group:<колонка>=<значения>.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .metrics import score as metric_score
from .shift import fmt_val

MIN_ROWS_SHARE = 0.01    # сегменты меньше 1% строк не показываем
TOP_SEGMENTS = 7
TOP_COLS = 3
MAX_LEVELS = 30


# ------------------------------------------------------------------ общие функции

def _covered(folds, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for _, va in folds:
        m[va] = True
    return m


def _keys(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "nan" if pd.isna(v) else fmt_val(v))


def _quintile_bins(v: np.ndarray) -> np.ndarray:
    """0 — пропуск, 1..5 — квинтили по значениям train."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(len(v), dtype=int)
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return out
    edges = np.nanquantile(v[ok], [0.2, 0.4, 0.6, 0.8])
    out[ok] = np.searchsorted(edges, v[ok], side="right") + 1
    return out


def _class_index(spec, name: str) -> int | None:
    for k, c in enumerate(spec.classes):
        if str(c) == name:
            return k
    return None


def _row_loss(spec, y: np.ndarray, oof: np.ndarray) -> np.ndarray:
    """Вклад строки в потерю. Регрессия — квадрат ошибки. Классификация — ошибка строки с весом
    1 / (K * размер её класса): сумма весов ошибок равна 1 − macro recall, редкие классы весят больше."""
    if spec.is_clf:
        pred = oof.argmax(axis=1)
        k = max(spec.n_classes, 1)
        cnt = np.bincount(y, minlength=k).astype(float)
        w = 1.0 / (k * np.maximum(cnt[y], 1.0))
        return w * (pred != y)
    return (np.asarray(oof, dtype=float) - y) ** 2


def _seg_metric(spec, y: np.ndarray, oof: np.ndarray, m: np.ndarray) -> float | None:
    if m.sum() == 0:
        return None
    if spec.is_clf:
        yt, pr = y[m], oof[m].argmax(axis=1)
        labels = np.unique(yt)
        return float(f1_score(yt, pr, labels=labels, average="macro", zero_division=0))
    return float(metric_score(spec.metric, y[m], oof[m], None))


# ------------------------------------------------------------------ маски сегментов

def segment_mask(seg_id: str, spec, data, shift=None) -> np.ndarray | None:
    """Маска строк train для id сегмента (None — id не распознан)."""
    n = data.n_train
    X = data.X.iloc[:n]
    kind, _, rest = str(seg_id).strip().partition(":")
    try:
        if kind in ("cat", "group"):
            col, _, val = rest.partition("=")
            if col not in X.columns:
                return None
            vals = set(val.rstrip("…").split(",")) if kind == "group" else {val}
            return _keys(X[col]).isin(vals).values
        if kind == "num":
            col, _, q = rest.rpartition(":")
            if col not in X.columns:
                return None
            bins = _quintile_bins(pd.to_numeric(X[col], errors="coerce").values)
            if q == "na":
                return bins == 0
            return bins == int(q.lstrip("q"))
        if kind == "target":
            if spec.is_clf:
                return None
            bins = _quintile_bins(np.asarray(data.y, dtype=float))
            return bins == int(rest.lstrip("q"))
        if kind == "testlike":
            if shift is None or shift.adv_p is None:
                return None
            p = np.asarray(shift.adv_p, dtype=float)
            thr = np.nanpercentile(p, 75)
            return np.nan_to_num(p, nan=-1.0) >= thr
        if kind == "class":
            k = _class_index(spec, rest)
            return None if k is None else (np.asarray(data.y) == k)
        if kind == "pair":
            a, _, _ = rest.partition("->")
            k = _class_index(spec, a)
            return None if k is None else (np.asarray(data.y) == k)
    except (ValueError, KeyError):
        return None
    return None


def segment_score(seg_id: str, spec, data, folds, shift, oof: np.ndarray) -> tuple[float, bool] | None:
    """(значение метрики сегмента, больше — лучше) или None, если id не распознан или сегмент пуст.
    class: — F1 класса; pair: — доля истинного класса, ушедшая в предсказанный (меньше — лучше);
    остальные — основная метрика задачи на строках сегмента."""
    y = np.asarray(data.y)
    cov = _covered(folds, len(y))
    kind, _, rest = str(seg_id).strip().partition(":")
    if kind == "class" and spec.is_clf:
        k = _class_index(spec, rest)
        if k is None:
            return None
        pred = oof.argmax(axis=1)
        return float(f1_score(y[cov], pred[cov], labels=[k], average="macro", zero_division=0)), True
    if kind == "pair" and spec.is_clf:
        a, _, b = rest.partition("->")
        ka, kb = _class_index(spec, a), _class_index(spec, b)
        if ka is None or kb is None:
            return None
        m = cov & (y == ka)
        if m.sum() == 0:
            return None
        return float((oof[m].argmax(axis=1) == kb).mean()), False
    m = segment_mask(seg_id, spec, data, shift)
    if m is None:
        return None
    m = m & cov
    val = _seg_metric(spec, y, oof, m)
    return None if val is None else (val, spec.greater_is_better)


# ------------------------------------------------------------------ отчёт

@dataclass
class ErrorReport:
    exp_id: str
    metric: str
    overall: float
    sections: list[tuple[str, list[str]]] = field(default_factory=list)   # (заголовок, строки)
    notes: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    all_ids: list[str] = field(default_factory=list)

    def segment_ids(self) -> list[str]:
        return list(self.all_ids)

    def has(self, seg_id: str | None) -> bool:
        return bool(seg_id) and str(seg_id).strip() in set(self.all_ids)

    def summary(self, max_chars: int = 3500) -> str:
        head = (f"Эксперимент {self.exp_id}, честный OOF после постобработки; {self.metric} = {self.overall:.5g}. "
                f"Доля потери минус доля строк = избыточный вклад сегмента.")
        parts = [head]
        used = len(head)
        for title, rows in self.sections:
            if not rows:
                continue
            block = [title] + rows
            for ln in block:
                if used + len(ln) + 1 > max_chars:
                    parts.append("... (обрезано)")
                    return "\n".join(parts)
                parts.append(ln)
                used += len(ln) + 1
        for ln in self.notes:
            if used + len(ln) + 1 > max_chars:
                break
            parts.append(ln)
            used += len(ln) + 1
        return "\n".join(parts)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {"exp_id": self.exp_id, "metric": self.metric, "overall": self.overall,
             "sections": [{"title": t, "rows": r} for t, r in self.sections], "notes": self.notes,
             "segments": self.segments, "all_ids": self.all_ids}
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _top_cols(cols: list[str], importances: dict | None, k: int) -> list[str]:
    if importances:
        cols = sorted(cols, key=lambda c: -importances.get(c, 0.0))
    return cols[:k]


def analyze_errors(spec, data, folds, res, shift=None, importances: dict | None = None) -> ErrorReport:
    y_all = np.asarray(data.y)
    n = len(y_all)
    cov = _covered(folds, n)
    oof_all = np.asarray(res.oof)
    rep = ErrorReport(exp_id=res.exp_id, metric=spec.metric, overall=float(res.score))
    ids: list[str] = []

    loss_all = _row_loss(spec, y_all, oof_all)
    loss_all = np.where(cov, loss_all, 0.0)
    total_loss = float(loss_all.sum()) or 1.0
    n_cov = int(cov.sum()) or 1

    # ---------------- классы и пары путаницы
    if spec.is_clf:
        y, pred = y_all[cov], oof_all[cov].argmax(axis=1)
        k = spec.n_classes
        p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
        cls_loss = (1.0 - f) / max(k, 1)
        tot = cls_loss.sum() or 1.0
        conflict = (shift.conflict_share_by_class if shift is not None else {}) or {}
        rows = []
        for j in np.argsort(f)[:12]:
            name = str(spec.classes[j])
            sid = f"class:{name}"
            ids.append(sid)
            cf = conflict.get(name)
            rows.append(f"- {sid} | F1 {f[j]:.3f} | P {p[j]:.3f} | R {r[j]:.3f} | строк {int(sup[j])} | "
                        f"противоречивых дубликатов {('%.0f%%' % (100 * cf)) if cf is not None else '-'} | "
                        f"доля потери {cls_loss[j] / tot:.2f}")
            rep.segments.append({"id": sid, "f1": float(f[j]), "support": int(sup[j]),
                                 "loss_share": float(cls_loss[j] / tot)})
        rep.sections.append(("Классы, от худшего F1 (id | F1 | precision | recall | строк | "
                             "доля строк-противоречивых дубликатов | доля потери macro F1):", rows))
        cm = confusion_matrix(y, pred, labels=list(range(k)))
        off = [(int(cm[a, b]), a, b) for a in range(k) for b in range(k) if a != b and cm[a, b] > 0]
        off.sort(reverse=True)
        rows = []
        for cnt, a, b in off[:5]:
            sid = f"pair:{spec.classes[a]}->{spec.classes[b]}"
            ids.append(sid)
            share = cnt / max(int(cm[a].sum()), 1)
            rows.append(f"- {sid} | {cnt} строк | {share:.0%} истинного класса")
            rep.segments.append({"id": sid, "count": cnt, "share_of_true": float(share)})
        rep.sections.append(("Пары путаницы (id | число | доля от истинного класса):", rows))
    else:
        # ---------------- бины таргета и крупнейшие промахи
        y = y_all[cov].astype(float)
        o = oof_all[cov].astype(float)
        err = o - y
        sq = err ** 2
        sse = float(sq.sum()) or 1.0
        bins = _quintile_bins(y_all.astype(float))[cov]
        rows = []
        for b in range(1, 6):
            m = bins == b
            if m.sum() == 0:
                continue
            sid = f"target:q{b}"
            ids.append(sid)
            rmse_b = float(np.sqrt(sq[m].mean()))
            bias = float(err[m].mean())
            share_rows, share_loss = m.mean(), sq[m].sum() / sse
            rows.append(f"- {sid} | строк {share_rows:.2f} | потери {share_loss:.2f} | RMSE {rmse_b:.4g} | "
                        f"смещение {bias:+.4g} (таргет {y[m].min():.4g}..{y[m].max():.4g})")
            rep.segments.append({"id": sid, "rows_share": float(share_rows), "loss_share": float(share_loss),
                                 "rmse": rmse_b, "bias": bias})
        rep.sections.append(("Квинтили таргета (id | доля строк | доля потери | RMSE | смещение = "
                             "среднее предсказание − таргет):", rows))
        kk = max(1, int(round(0.01 * len(sq))))
        top_share = float(np.sort(sq)[-kk:].sum() / sse)
        rep.notes.append(f"Крупнейшие промахи: 1% строк дают {top_share:.0%} суммы квадратов ошибок.")
        rep.segments.append({"id": "top1pct_errors", "loss_share": top_share})

    # ---------------- сегменты по признакам
    X = data.X.iloc[:n]
    cands = []
    cat_cols = _top_cols([c for c in spec.cat_cols if c in X.columns], importances, TOP_COLS)
    for col in cat_cols:
        keys = _keys(X[col])
        vc = keys[cov].value_counts()
        for val, cnt in vc.head(MAX_LEVELS).items():
            share_rows = cnt / n_cov
            if share_rows < MIN_ROWS_SHARE:
                continue
            m = (keys == val).values & cov
            cands.append((f"cat:{col}={val}", m))
    num_cols = _top_cols([c for c in spec.num_cols if c in X.columns], importances, TOP_COLS)
    for col in num_cols:
        bins = _quintile_bins(pd.to_numeric(X[col], errors="coerce").values)
        for b in range(0, 6):
            m = (bins == b) & cov
            if m.sum() / n_cov < MIN_ROWS_SHARE:
                continue
            cands.append((f"num:{col}:{'na' if b == 0 else 'q%d' % b}", m))
    seg_rows = []
    for sid, m in cands:
        share_rows = m.sum() / n_cov
        share_loss = loss_all[m].sum() / total_loss
        seg_rows.append((share_loss - share_rows, sid, share_rows, share_loss, m))
        ids.append(sid)
    seg_rows.sort(key=lambda t: -t[0])
    rows = []
    for exc, sid, sr, sl, m in seg_rows[:TOP_SEGMENTS]:
        if exc <= 0:
            break
        met = _seg_metric(spec, y_all, oof_all, m)
        rows.append(f"- {sid} | строк {sr:.2f} | потери {sl:.2f} | избыток {exc:+.2f} | "
                    f"{spec.metric} {met:.4g}" if met is not None else f"- {sid} | строк {sr:.2f} | потери {sl:.2f}")
        rep.segments.append({"id": sid, "rows_share": float(sr), "loss_share": float(sl), "excess": float(exc),
                             "metric": met})
    loss_word = "квадрат ошибки" if not spec.is_clf else "ошибки с весом редких классов"
    rep.sections.append((f"Сегменты по признакам с избыточной потерей (потеря — {loss_word}; "
                         f"id | доля строк | доля потери | избыток | метрика сегмента):", rows))

    # ---------------- строки, похожие на test
    if shift is not None and shift.adv_p is not None:
        m_top = segment_mask("testlike:top25", spec, data, shift)
        if m_top is not None:
            m_top = m_top & cov
            m_rest = (~m_top) & cov
            a, b = _seg_metric(spec, y_all, oof_all, m_top), _seg_metric(spec, y_all, oof_all, m_rest)
            if a is not None and b is not None:
                ids.append("testlike:top25")
                share_loss = loss_all[m_top].sum() / total_loss
                rep.sections.append(("Строки train, похожие на test (четверть с наибольшей вероятностью "
                                     "«test» по adversarial validation):",
                                     [f"- testlike:top25 | {spec.metric} {a:.4g} против {b:.4g} у остальных | "
                                      f"доля потери {share_loss:.2f}"]))
                rep.segments.append({"id": "testlike:top25", "metric": a, "metric_rest": b,
                                     "loss_share": float(share_loss)})

    # ---------------- по группам фолдов
    if getattr(folds, "kind", "random") in ("time", "group") and folds.names:
        rows = []
        for k_, (_, va) in enumerate(folds):
            sid = f"group:{folds.names[k_]}"
            ids.append(sid)
            val = res.fold_scores[k_] if k_ < len(res.fold_scores) else None
            if val is not None:
                rows.append(f"- {sid} | строк {len(va) / n_cov:.2f} | {spec.metric} {val:.4g}")
        rep.sections.append(("Метрика по группам фолдов (каждый фолд — целые группы, которых нет в обучении):", rows))

    rep.all_ids = list(dict.fromkeys(ids))
    return rep
