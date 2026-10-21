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
