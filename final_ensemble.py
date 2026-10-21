"""Постобработка под метрику, бленд и финальный выбор на selection-holdout.

- fit_post: подбор порогов, множителей классов, обрезки на OOF одного кандидата;
- caruana: жадный бленд с возвратом по OOF;
- select_final: единственное место, где читаются метки selection-holdout (через LabelVault);
  побеждает более сложный кандидат, только если он лучше более простого в доле бутстреп-выборок
  не меньше cfg.bootstrap_conf;
- check_prior_shift: коррекция сдвига приоров классов принимается, только если она улучшает
  метрику на holdout, пересэмплированном под оценённые доли классов test.
"""

import logging

import numpy as np

import task_metrics

log = logging.getLogger("agent")

_GRID = (0.3, 0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0)


def _better(a, b, gib):
    return task_metrics.gain(a, gib) > task_metrics.gain(b, gib) + 1e-12


def fit_post(oof, y, spec, post_cfg: dict, metric) -> dict:
    post_cfg = post_cfg or {}
    tt = spec.task_type
    y = np.asarray(y)
    oof = np.asarray(oof, dtype=float)
    post = {}
    if tt in task_metrics.REG_TYPES:
        if post_cfg.get("clip_to_train_range", True):
            post["clip"] = [np.min(y, axis=0).tolist(), np.max(y, axis=0).tolist()]
        return post
    if tt in ("binary", "multiclass"):
        post["train_prior"] = (np.bincount(y.astype(int), minlength=max(2, spec.n_classes)) / len(y)).tolist()
    if metric.need != "label":
        return post
    gib = metric.greater_is_better
    if tt == "binary" and post_cfg.get("threshold_opt", True):
        cands = np.unique(np.r_[np.quantile(oof, np.linspace(0.01, 0.99, 99)), 0.5])
        best_t, best_s = 0.5, metric.fn(y, (oof >= 0.5).astype(int))
        for t in cands:
            s = metric.fn(y, (oof >= t).astype(int))
            if _better(s, best_s, gib):
                best_t, best_s = float(t), s
        post["thr"] = best_t
    elif tt == "multiclass" and post_cfg.get("class_weights_opt", True):
        K = oof.shape[1]
        w = np.ones(K)
        best = metric.fn(y, (oof * w).argmax(axis=1))
        for _ in range(3):
            improved = False
            for k in range(K):
                for g in _GRID:
                    w2 = w.copy()
                    w2[k] *= g
                    s = metric.fn(y, (oof * w2).argmax(axis=1))
                    if _better(s, best, gib) and abs(task_metrics.gain(s, gib) - task_metrics.gain(best, gib)) > 1e-6:
                        w, best, improved = w2, s, True
            if not improved:
                break
        post["class_w"] = (w / w.mean()).tolist()
    elif tt == "multilabel" and post_cfg.get("threshold_opt", True):
        thr = []
        for j in range(oof.shape[1]):
            cands = np.unique(np.r_[np.quantile(oof[:, j], np.linspace(0.02, 0.98, 49)), 0.5])
            best_t, best_s = 0.5, None
            for t in cands:
                lab = (oof >= 0.5).astype(int)
                lab[:, j] = (oof[:, j] >= t).astype(int)
                s = metric.fn(y, lab)
                if best_s is None or _better(s, best_s, gib):
                    best_t, best_s = float(t), s
            thr.append(best_t)
        post["thr"] = thr
    return post


def blend(preds: list, weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = None
    for p, wi in zip(preds, w):
        if wi == 0:
            continue
        out = wi * np.asarray(p, dtype=float) if out is None else out + wi * np.asarray(p, dtype=float)
    return out


def caruana(oofs: list, y, spec, metric, max_steps: int = 30) -> np.ndarray:
    """Жадный отбор с возвратом (Caruana et al., 2004). Возвращает веса кандидатов."""
    n = len(oofs)
    gib = metric.greater_is_better
    single = [task_metrics.score(metric, y, o, spec.task_type) for o in oofs]
    start = int(np.argmax([task_metrics.gain(s, gib) for s in single]))
    counts = np.zeros(n)
    counts[start] = 1
    cur = np.asarray(oofs[start], dtype=float).copy()
    best_counts, best_score = counts.copy(), single[start]
    for step in range(1, max_steps):
        step_best, step_i = None, None
        for i in range(n):
            cand = (cur * step + oofs[i]) / (step + 1)
            s = task_metrics.score(metric, y, cand, spec.task_type)
            if step_best is None or _better(s, step_best, gib):
                step_best, step_i = s, i
        counts[step_i] += 1
        cur = (cur * step + oofs[step_i]) / (step + 1)
        if _better(step_best, best_score, gib):
            best_score, best_counts = step_best, counts.copy()
        elif step - best_counts.sum() >= 5:
            break
    return best_counts / best_counts.sum()


def select_final(cands: list, vault, spec, metric, cfg) -> tuple:
    """cands: список словарей {name, complexity, ho, post}. Возвращает (победитель, отчёт)."""
    report = {}
    for c in cands:
        report[c["name"]] = {"holdout": vault.score(metric, c["ho"], spec.task_type, c["post"])}
    order = sorted(cands, key=lambda c: c["complexity"])
    winner = order[0]
    for c in order[1:]:
        p = vault.prob_better(metric, c["ho"], c["post"], winner["ho"], winner["post"], spec.task_type,
                              n_boot=cfg.bootstrap_n, seed=cfg.seed)
        report[c["name"]]["p_better_than_" + winner["name"]] = round(p, 3)
        if p >= cfg.bootstrap_conf:
            winner = c
    return winner, report


def check_prior_shift(te_pred, ho_pred, post, vault, spec, metric, min_tv: float = 0.03) -> dict:
    """Оценить доли классов в test EM-алгоритмом и проверить коррекцию на пересэмплированном holdout."""
    if spec.task_type not in ("binary", "multiclass") or not post or post.get("train_prior") is None:
        return {"apply": False, "reason": "не классификация"}
    train_prior = np.asarray(post["train_prior"], dtype=float)
    est = task_metrics.em_priors(te_pred, train_prior)
    tv = float(0.5 * np.abs(est - train_prior).sum())
    if tv < min_tv:
        return {"apply": False, "tv": tv, "reason": "сдвиг приоров мал"}
    res = vault.shift_check(metric, ho_pred, post, est, train_prior, spec.task_type)
    apply = res["win_frac"] >= 0.7 and res["mean_gain"] > 0
    return {"apply": bool(apply), "tv": tv, "prior": est.tolist(), **res}
