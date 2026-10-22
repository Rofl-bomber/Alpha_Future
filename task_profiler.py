"""Профиль данных, детектор схемы сплита и adversarial validation.

Профиль считается кодом без LLM. Для LLM он сжимается в текстовую сводку
(до ~3 тыс. токенов). Adversarial validation (классификатор «train против test»)
обучается в отдельном процессе через exp_runner, чтобы родитель не трогал OpenMP.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("agent")


def _unseen_frac(tr: pd.Series, te: pd.Series) -> float:
    seen = set(tr.dropna().astype(str).unique())
    te_s = te.dropna().astype(str)
    if len(te_s) == 0:
        return 0.0
    return float((~te_s.isin(seen)).mean())


def _id_runs(ids):
    if ids is None:
        return None
    v = pd.to_numeric(ids, errors="coerce")
    if v.isna().any():
        return None
    v = np.sort(v.to_numpy())
    return int(1 + (np.diff(v) != 1).sum())


def profile_data(spec, data, cfg) -> dict:
    X = data.X_all
    ntr = data.n_train
    Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
    prof = {"n_train": ntr, "n_test": len(Xte), "n_features": len(spec.feature_cols), "columns": {}}

    constant, group_cands, time_cands = [], [], []
    for c in spec.feature_cols:
        s = X[c]
        info = {"kind": ("cat" if c in spec.cat_cols else "num" if c in spec.num_cols else
                         "text" if c in spec.text_cols else "date"),
                "miss_train": round(float(Xtr[c].isna().mean()), 4),
                "miss_test": round(float(Xte[c].isna().mean()), 4),
                "nunique": int(s.nunique(dropna=True))}
        if info["nunique"] <= 1:
            constant.append(c)
        if info["kind"] == "num" and s.notna().any():
            v = Xtr[c].dropna()
            if len(v):
                info.update(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()),
                            skew=float(v.skew()) if len(v) > 2 else 0.0)
            te = Xte[c].dropna()
            if len(te) and len(v):
                info["test_above_train_max"] = round(float((te > v.max()).mean()), 4)
        low_card = info["nunique"] <= max(50, int(0.002 * len(X)))
        if info["kind"] in ("cat", "num") and low_card and info["nunique"] >= 2:
            uf = _unseen_frac(Xtr[c], Xte[c])
            info["unseen_in_test"] = round(uf, 4)
            if uf >= cfg.group_unseen_frac:
                ordered = info["kind"] == "num" and info.get("test_above_train_max", 0) >= 0.95
                group_cands.append({"col": c, "unseen": uf, "n_values": info["nunique"], "time_ordered": ordered})
        elif info["kind"] == "cat":
            info["unseen_in_test"] = round(_unseen_frac(Xtr[c], Xte[c]), 4)
        if info["kind"] == "date":
            try:
                dtr = pd.to_datetime(Xtr[c], errors="coerce", format="mixed")
                dte = pd.to_datetime(Xte[c], errors="coerce", format="mixed")
                if dtr.notna().any() and dte.notna().any():
                    above = float((dte > dtr.max()).mean())
                    info["test_after_train"] = round(above, 4)
                    if above >= 0.9:
                        time_cands.append({"col": c, "after": above})
            except Exception:
                pass
        if info["kind"] == "cat":
            vc = Xtr[c].astype(str).value_counts()
            info["top"] = [(str(k)[:30], int(v)) for k, v in vc.head(5).items()]
        prof["columns"][c] = info
    prof["constant"] = constant

    # дубли колонок (по хэшу значений)
    hashes = {}
    dup_cols = []
    for c in spec.num_cols:
        h = pd.util.hash_pandas_object(X[c], index=False).sum()
        if h in hashes:
            dup_cols.append((c, hashes[h]))
        else:
            hashes[h] = c
    prof["duplicate_cols"] = dup_cols[:20]

    # таргет
    y = data.y
    if spec.task_type == "regression":
        yv = y.astype(float)
        prof["target"] = {"mean": float(yv.mean()), "std": float(yv.std()), "min": float(yv.min()),
                          "max": float(yv.max()), "skew": float(pd.Series(yv).skew()),
                          "q": {q: float(np.quantile(yv, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
                          "nonneg": bool((yv >= 0).all())}
    elif spec.task_type in ("binary", "multiclass"):
        counts = np.bincount(y, minlength=spec.n_classes)
        prof["target"] = {"counts": {str(spec.classes[i]): int(n) for i, n in enumerate(counts)},
                          "prior": (counts / counts.sum()).round(5).tolist(),
                          "imbalance": float(counts.max() / max(1, counts[counts > 0].min()))}
    else:
        prof["target"] = {"shape": list(np.asarray(y).shape)}

    # пересечение строк train и test
    feat = [c for c in spec.feature_cols if c not in spec.text_cols]
    try:
        h_tr = pd.util.hash_pandas_object(Xtr[feat], index=False)
        h_te = pd.util.hash_pandas_object(Xte[feat], index=False)
        prof["test_rows_in_train"] = round(float(h_te.isin(set(h_tr)).mean()), 4)
        prof["dup_rows_train"] = round(float(h_tr.duplicated().mean()), 4)
    except Exception:
        prof["test_rows_in_train"] = None

    prof["test_id_runs"] = _id_runs(data.test_ids)
    prof["train_id_runs"] = _id_runs(data.train_ids)
    prof["group_candidates"] = sorted(group_cands, key=lambda g: (-g["unseen"], g["n_values"]))[:5]
    prof["time_candidates"] = time_cands

    # выбор схемы
    prof["group_col"], prof["time_col"], prof["time_ordered_groups"] = None, None, False
    if spec.time_hint and spec.time_hint in spec.feature_cols:
        prof["time_col"] = spec.time_hint
    elif time_cands:
        prof["time_col"] = time_cands[0]["col"]
    if prof["time_col"] is None:
        if spec.group_hint and spec.group_hint in spec.feature_cols:
            prof["group_col"] = spec.group_hint
        elif prof["group_candidates"]:
            g = prof["group_candidates"][0]
            prof["group_col"] = g["col"]
            prof["time_ordered_groups"] = bool(g["time_ordered"])
    return prof


def run_adversarial(spec, prof: dict, ctx) -> dict:
    """Adversarial validation в дочернем процессе. Колонки группы/времени исключаются."""
    import exp_runner

    exclude = [c for c in (prof.get("group_col"), prof.get("time_col")) if c]
    job = {"kind": "adversarial", "exclude": exclude}
    if ctx.cfg.unlimited:  # без внутренних лимитов: до конца фазы L1
        timeout = max(5.0, ctx.timer.left("l1"))
    else:
        timeout = min(300.0, max(60.0, ctx.timer.left("l1") * 0.3))
    res = exp_runner.run_job(job, ctx, timeout=timeout, tag="adversarial")
    if not res.get("ok"):
        log.warning("adversarial validation не удалась: %s", str(res.get("error"))[:300])
        return {"auc": None}
    out = {"auc": res["auc"], "top": res["top"], "excluded": exclude}
    if res.get("weights_path"):
        out["weights_path"] = res["weights_path"]
    return out


def summary_text(spec, prof: dict, max_cols: int = 80) -> str:
    """Сжатая сводка профиля для LLM."""
    lines = [f"Строк: train={prof['n_train']}, test={prof['n_test']}; признаков={prof['n_features']}",
             f"Задача: {spec.short()}"]
    t = prof.get("target", {})
    if spec.task_type == "regression" and t:
        lines.append(f"Таргет: mean={t['mean']:.4g}, std={t['std']:.4g}, min={t['min']:.4g}, max={t['max']:.4g}, "
                     f"skew={t['skew']:.2f}, q01={t['q'][0.01]:.4g}, q99={t['q'][0.99]:.4g}")
    elif "counts" in t:
        lines.append("Классы: " + ", ".join(f"{k}={v}" for k, v in t["counts"].items())
                     + f"; дисбаланс {t['imbalance']:.0f}x")
    if prof.get("constant"):
        lines.append(f"Константные колонки: {prof['constant'][:10]}")
    if prof.get("duplicate_cols"):
        lines.append(f"Дубли колонок: {prof['duplicate_cols'][:5]}")
    if prof.get("test_rows_in_train") is not None:
        lines.append(f"Доля строк test, совпадающих с train по признакам: {prof['test_rows_in_train']:.1%}; "
                     f"дубли внутри train: {prof.get('dup_rows_train', 0):.1%}")
    if prof.get("test_id_runs") is not None:
        lines.append(f"id теста идут {prof['test_id_runs']} сплошными блоками")
    if prof.get("group_col"):
        lines.append(f"Схема CV: группы по {prof['group_col']} (значения test не встречаются в train"
                     + (", test позже train" if prof.get("time_ordered_groups") else "") + ")")
    elif prof.get("time_col"):
        lines.append(f"Схема CV: по времени, колонка {prof['time_col']} (test позже train)")
    adv = prof.get("adversarial") or {}
    if adv.get("auc") is not None:
        lines.append(f"Adversarial AUC train vs test = {adv['auc']:.3f}; дрейфующие признаки: "
                     + ", ".join(f"{f}" for f, _ in adv.get("top", [])[:8]))
    lines.append("Колонки:")
    for c, info in list(prof["columns"].items())[:max_cols]:
        bits = [info["kind"], f"nuniq={info['nunique']}"]
        if info["miss_train"] or info["miss_test"]:
            bits.append(f"NaN tr/te={info['miss_train']:.0%}/{info['miss_test']:.0%}")
        if "skew" in info and abs(info["skew"]) > 2:
            bits.append(f"skew={info['skew']:.1f}")
        if info.get("unseen_in_test", 0) > 0.05:
            bits.append(f"невиданных в test={info['unseen_in_test']:.0%}")
        if "top" in info:
            bits.append("топ=" + ",".join(k for k, _ in info["top"][:3]))
        desc = spec.descriptions.get(c, "")
        lines.append(f"  {c}: " + "; ".join(bits) + (f" — {desc[:80]}" if desc else ""))
    if len(prof["columns"]) > max_cols:
        lines.append(f"  ... ещё {len(prof['columns']) - max_cols} колонок")
    return "\n".join(lines)
