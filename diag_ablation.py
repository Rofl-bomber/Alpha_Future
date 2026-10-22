"""Абляции блоков пайплайна в быстром режиме (3 фолда, высокий learning rate).

Для узла по очереди выключаются группы признаков и переключаются кодирование, модель, таргет.
Результат — таблица «блок · вариант · Δ ± SE» относительно базы на тех же фолдах.
Блок с наибольшей чувствительностью становится целью следующих Improve, а варианты,
которые сами улучшили результат, сразу идут в очередь гипотез.
"""

import math

import numpy as np

import search_operators


def variants(config: dict, spec, prof: dict) -> list:
    out = []
    feats = config.get("features", {})
    if feats.get("custom"):
        out.append(("features", "без LLM-признаков", {"features.custom": []}))
    extra_builtin = [o for o in feats.get("builtin", [])]
    if extra_builtin:
        out.append(("features", "без встроенных операций", {"features.builtin": []}))
    cats = [c for c in spec.cat_cols if c not in feats.get("drop", [])]
    if cats:
        out.append(("features", "без категориальных колонок", {"features.drop": list(feats.get("drop", [])) + cats}))
    adv = prof.get("adversarial") or {}
    if adv.get("auc") and adv["auc"] > 0.6:
        drift = [f for f, _ in adv.get("top", [])[:3] if f in spec.feature_cols and f not in feats.get("drop", [])]
        if drift:
            out.append(("features", f"без дрейфующих {drift}", {"features.drop": list(feats.get("drop", [])) + drift}))
    if spec.cat_cols:
        cur = config.get("preprocess", {}).get("cat_encoding", "native")
        for enc in ("native", "freq", "target_oof"):
            if enc != cur:
                out.append(("preprocess", f"кодирование {enc}", {"preprocess.cat_encoding": enc}))
    name = config.get("model", {}).get("name", "lgbm")
    alt = "catboost" if name != "catboost" else "lgbm"
    out.append(("model", f"модель {alt}", {"model.name": alt}))
    t = prof.get("target", {})
    tcfg = config.get("target", {})
    if spec.task_type == "regression":
        cur = tcfg.get("transform", "none")
        if cur == "none" and t.get("nonneg") and abs(t.get("skew", 0)) > 1:
            out.append(("target", "таргет log1p", {"target.transform": "log1p"}))
        elif cur != "none":
            out.append(("target", "таргет без преобразования", {"target.transform": "none"}))
    elif spec.task_type in ("binary", "multiclass") and t.get("imbalance", 1) > 5:
        cur = tcfg.get("class_weight", "none")
        out.append(("target", "веса классов " + ("none" if cur != "none" else "sqrt_balanced"),
                    {"target.class_weight": "none" if cur != "none" else "sqrt_balanced"}))
    return out


def run(node, spec, prof, evaluate_screen, gib: bool, time_left_fn, est_time: float) -> dict:
    """evaluate_screen(config, tag) → результат эксперимента в режиме screen (или None)."""
    base = evaluate_screen(node.config, f"{node.id}_abl_base")
    if not base or not base.get("ok") or base.get("truncated"):
        why = "обрезан по времени" if base and base.get("truncated") else "не удался"
        return {"rows": [], "text": f"Абляции не выполнены: базовый прогон {why}.", "focus": None, "improving": []}
    rows, improving = [], []
    for block, name, patch in variants(node.config, spec, prof):
        if time_left_fn() < est_time * 1.2:
            break
        cfg2 = search_operators.apply_patch(node.config, patch)
        res = evaluate_screen(cfg2, f"{node.id}_abl_{len(rows)}")
        if not res or not res.get("ok"):
            err = (res or {}).get("error", "") or "нет времени"
            rows.append({"block": block, "variant": name, "error": str(err).splitlines()[0][:200]})
            continue
        # сравнение на тех же фолдах; у обрезанного прогона — только на пройденных
        done = res.get("folds_done") or list(range(len(res["fold_scores"])))
        a = np.asarray(res["fold_scores"])
        b = np.asarray(base["fold_scores"])[done]
        d = (a - b) * (1 if gib else -1)
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
        row = {"block": block, "variant": name, "delta_gain": float(d.mean()), "se": se, "mean": res["mean"],
               "patch": patch, "truncated": bool(res.get("truncated")), "folds_done": done}
        rows.append(row)
        if not row["truncated"] and d.mean() > max(se, 0) and (d > 0).sum() >= math.ceil(0.6 * len(d)):
            improving.append(row)
    sens = {}
    for r in rows:
        if "delta_gain" in r and not r.get("truncated"):
            sens[r["block"]] = max(sens.get(r["block"], 0.0), abs(r["delta_gain"]))
    focus = max(sens, key=sens.get) if sens else None
    lines = [f"Абляции узла {node.id} (быстрый режим, база {base['mean']:.6g}; Δ>0 — вариант лучше базы):"]
    for r in rows:
        if "delta_gain" in r:
            note = (f" (обрезан по времени, фолды {[k + 1 for k in r['folds_done']]})" if r.get("truncated") else "")
            lines.append(f"  [{r['block']}] {r['variant']}: Δ={r['delta_gain']:+.4g} ± {r['se']:.2g}{note}")
        else:
            lines.append(f"  [{r['block']}] {r['variant']}: ошибка — {r.get('error', '')}")
    if focus:
        lines.append(f"Самый чувствительный блок: {focus}")
    return {"rows": rows, "text": "\n".join(lines), "focus": focus, "improving": improving, "base": base}
