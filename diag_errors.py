"""Анализ ошибок узла по OOF: где модель теряет. Код, без LLM; сводка — для Исследователя.

Регрессия: ошибка по децилям таргета, по крупным категориям, профиль строк с крупнейшими ошибками.
Классификация: F1 по классам, топ пар путаницы, доли классов в OOF, факте и предсказаниях на test.
"""

import numpy as np
import pandas as pd

import task_metrics


def analyze(spec, y, oof, post, X: pd.DataFrame, te_pred=None, prof=None) -> tuple:
    if spec.task_type == "regression":
        return _regression(spec, np.asarray(y, float), np.asarray(oof, float), post, X)
    if spec.task_type in ("binary", "multiclass"):
        return _classification(spec, np.asarray(y, int), np.asarray(oof, float), post, te_pred)
    return {}, "Анализ ошибок для этого типа задачи не реализован."


def _cell(v, width: int = 24) -> str:
    try:
        if v is None or pd.isna(v):
            return "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() and abs(f) < 1e12 else f"{f:.4g}"
    return str(v)[:width]


def example_rows(spec, X: pd.DataFrame, y, oof=None, post=None, n: int = 5, order=None, max_cols: int = 40,
                 seed: int = 0) -> str:
    """Примеры строк поисковой части train для Исследователя, транспонированно.

    Столбцы r* — случайные строки, e* — строки с наибольшей ошибкой OOF текущего решения.
    Строки selection-holdout сюда не попадают: X и y — только поисковая часть.
    """
    y = np.asarray(y)
    tt = spec.task_type
    N = len(X)
    if N == 0:
        return "нет строк"
    rng = np.random.default_rng(seed)
    rand = list(rng.choice(N, size=min(n, N), replace=False))
    worst, pred_txt, extra = [], None, None
    names = [str(c) for c in spec.classes] if spec.classes else []
    if oof is not None and tt in ("regression", "binary", "multiclass"):
        p = task_metrics.apply_post(np.asarray(oof, float), tt, post)
        if tt == "regression":
            err = np.abs(y.astype(float) - p)
            pred_txt = [_cell(v) for v in p]
        else:
            lab = task_metrics.to_labels(p, tt, post)
            yi = y.astype(int)
            p_true = np.where(yi == 1, p, 1 - p) if tt == "binary" else p[np.arange(N), yi]
            err = 1 - p_true
            pred_txt = [names[k] if k < len(names) else str(k) for k in lab]
            extra = [f"{v:.2f}" for v in p_true]
        worst = [int(i) for i in np.argsort(-err)[: n + len(rand)] if i not in rand][:n]
    idx = rand + worst
    heads = [f"r{j + 1}" for j in range(len(rand))] + [f"e{j + 1}" for j in range(len(worst))]

    def target_cells():
        if tt in ("binary", "multiclass"):
            return [names[int(y[i])] if int(y[i]) < len(names) else str(y[i]) for i in idx]
        if y.ndim == 2:
            return [",".join(_cell(v, 6) for v in y[i][:4]) for i in idx]
        return [_cell(y[i]) for i in idx]

    tname = spec.targets[0] if len(spec.targets) == 1 else "+".join(spec.targets[:3])
    rows = [(f"{tname} (таргет)", target_cells())]
    if pred_txt is not None:
        rows.append(("прогноз OOF", [pred_txt[i] for i in idx]))
    if extra is not None:
        rows.append(("p(верный класс)", [extra[i] for i in idx]))
    cols = [c for c in (order or []) if c in X.columns]
    cols += [c for c in X.columns if c not in cols]
    rest = max(0, len(cols) - max_cols)
    for c in cols[:max_cols]:
        s = X[c]
        rows.append((str(c), [_cell(s.iloc[i]) for i in idx]))
    wn = min(30, max(len(r[0]) for r in rows))  # значения без выравнивания: так короче в токенах
    out = [f"{'колонка'[:wn]:<{wn}} | " + " | ".join(heads)]
    for name, cells in rows:
        out.append(f"{name[:wn]:<{wn}} | " + " | ".join(cells))
    head = (f"r1–r{len(rand)} — случайные строки train"
            + (f", e1–e{len(worst)} — строки с наибольшей ошибкой OOF текущего решения" if worst else "")
            + "; колонки по убыванию важности")
    if rest:
        out.append(f"... ещё {rest} колонок не показаны")
    return head + "\n" + "\n".join(out)


def _regression(spec, y, oof, post, X):
    p = task_metrics.apply_post(oof, "regression", post)
    r = y - p
    out = {"rmse": float(np.sqrt(np.mean(r**2))), "mae": float(np.mean(np.abs(r)))}
    lines = [f"Ошибка OOF: RMSE={out['rmse']:.4g}, MAE={out['mae']:.4g}"]
    qs = np.unique(np.quantile(y, np.linspace(0, 1, 11)))
    bins = np.clip(np.searchsorted(qs, y, side="right") - 1, 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append((f"[{qs[b]:.4g}; {qs[b + 1]:.4g}]", int(m.sum()), float(np.sqrt(np.mean(r[m] ** 2))),
                     float(np.mean(r[m])), float((r[m] ** 2).sum() / (r**2).sum())))
    out["by_target_decile"] = rows
    lines.append("По децилям таргета (диапазон: n, RMSE, смещение факт−прогноз, доля суммы квадратов ошибок):")
    for rng_, n, e, b, share in rows:
        lines.append(f"  {rng_}: n={n}, RMSE={e:.4g}, bias={b:+.4g}, доля={share:.0%}")
    cats = [c for c in spec.cat_cols if c in X.columns and 2 <= X[c].nunique() <= 100][:3]
    for c in cats:
        key = X[c].astype(str).to_numpy()
        df = pd.DataFrame({"k": key, "se": r**2, "r": r})
        g = df.groupby("k").agg(n=("se", "size"), sse=("se", "sum"), bias=("r", "mean"))
        g["rmse"] = np.sqrt(g["sse"] / g["n"])
        g["share"] = g["sse"] / g["sse"].sum()
        g = g.sort_values("share", ascending=False).head(6)
        for k, row in g.iterrows():
            if row.share >= 0.1 and row.rmse >= 1.3 * out["rmse"] and row.n >= 30:
                out.setdefault("segments", []).append((c, str(k)))
        lines.append(f"По {c} (топ по вкладу в ошибку):")
        for k, row in g.iterrows():
            lines.append(f"  {str(k)[:30]}: n={int(row.n)}, RMSE={row.rmse:.4g}, bias={row.bias:+.4g}, "
                         f"доля ошибки={row.share:.0%}")
    worst = np.abs(r) >= np.quantile(np.abs(r), 0.99)
    diffs = []
    for c in spec.num_cols:
        if c not in X.columns:
            continue
        s = X[c].astype(float)
        sd = s.std()
        if not sd or not np.isfinite(sd):
            continue
        diffs.append((c, float((s[worst].mean() - s.mean()) / sd)))
    diffs.sort(key=lambda t: -abs(t[1]))
    if diffs:
        lines.append("Худший 1% строк по ошибке отличается по признакам (стандартизованная разница средних): "
                     + ", ".join(f"{c} {d:+.2f}" for c, d in diffs[:6]))
    for c in cats:
        share_all = X[c].astype(str).value_counts(normalize=True)
        share_w = X.loc[worst, c].astype(str).value_counts(normalize=True)
        lift = (share_w / share_all.reindex(share_w.index)).sort_values(ascending=False)
        top = [(k, v) for k, v in lift.head(3).items() if v >= 3 and share_w[k] >= 0.05]
        if top:
            lines.append(f"  среди худших перепредставлены {c}: " + ", ".join(f"{k} (×{v:.1f})" for k, v in top))
    return out, "\n".join(lines)


def _classification(spec, y, oof, post, te_pred):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    lab = task_metrics.to_labels(task_metrics.apply_post(oof, spec.task_type, post), spec.task_type, post)
    K = max(2, spec.n_classes)
    prec, rec, f1, sup = precision_recall_fscore_support(y, lab, labels=list(range(K)), zero_division=0)
    names = [str(c) for c in spec.classes] if spec.classes else [str(k) for k in range(K)]
    out = {"per_class": {names[k]: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k]),
                                    "support": int(sup[k])} for k in range(K)}}
    lines = ["F1 по классам OOF (precision / recall / F1, support):"]
    for k in np.argsort(f1):
        lines.append(f"  {names[k]}: {prec[k]:.3f} / {rec[k]:.3f} / {f1[k]:.3f}, n={sup[k]}")
    cm = confusion_matrix(y, lab, labels=list(range(K)))
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("Чаще всего путаются (факт → прогноз: число, доля факта):")
        for n, i, j in pairs[:5]:
            lines.append(f"  {names[i]} → {names[j]}: {n} ({n / max(1, cm[i].sum()):.0%})")
    prior = np.bincount(y, minlength=K) / len(y)
    pred_oof = np.bincount(lab, minlength=K) / len(lab)
    line = "Доли классов (факт train / прогноз OOF" + (" / прогноз test" if te_pred is not None else "") + "): "
    if te_pred is not None:
        lab_te = task_metrics.to_labels(task_metrics.apply_post(te_pred, spec.task_type, post), spec.task_type, post)
        pred_te = np.bincount(lab_te, minlength=K) / len(lab_te)
        out["test_pred_share"] = pred_te.tolist()
        line += ", ".join(f"{names[k]} {prior[k]:.1%}/{pred_oof[k]:.1%}/{pred_te[k]:.1%}" for k in range(K))
    else:
        line += ", ".join(f"{names[k]} {prior[k]:.1%}/{pred_oof[k]:.1%}" for k in range(K))
    lines.append(line)
    return out, "\n".join(lines)
