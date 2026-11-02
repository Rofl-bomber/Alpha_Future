"""OOF-based ensembling: Caruana hill climbing and level-2 stacking."""
from __future__ import annotations

import numpy as np

import mla_metrics as MT
from mla_postprocess import metric_with_decision
from mla_utils import LOG


def blend_objective(metric: str, task: str):
    """Smooth objective (higher=better) used to choose blend weights."""
    m = MT.get(metric)
    if task == "regression":
        return lambda y, p: MT.utility(metric, m.fn(y, p))
    if m.needs == "proba":
        return lambda y, P: MT.utility(metric, MT.score(metric, y, P))
    # label metrics: optimize log loss of the blend (stable surrogate); decision rule fitted afterwards
    return lambda y, P: -MT.score("logloss", y, P)


def hill_climb(preds: list, y: np.ndarray, objective, max_steps: int = 80, init_top: int = 1) -> tuple[np.ndarray, float]:
    """Greedy forward selection with replacement (Caruana et al., 2004). Returns (weights, objective)."""
    M = len(preds)
    scores = [objective(y, p) for p in preds]
    order = np.argsort(scores)[::-1]
    counts = np.zeros(M)
    for i in order[:init_top]:
        counts[i] += 1
    cur = sum(preds[i] * counts[i] for i in range(M)) / counts.sum()
    best = objective(y, cur)
    for _ in range(max_steps):
        n = counts.sum()
        cand_best, cand_i = best, -1
        for i in range(M):
            p = (cur * n + preds[i]) / (n + 1)
            s = objective(y, p)
            if s > cand_best + 1e-12:
                cand_best, cand_i = s, i
        if cand_i < 0:
            break
        counts[cand_i] += 1
        cur = (cur * n + preds[cand_i]) / (n + 1)
        best = cand_best
    w = counts / counts.sum()
    return w, best


def stack_level2(oofs: list, tests: list, y: np.ndarray, task: str, n_classes: int, folds: list, objective, seed: int = 0,
                 time_budget: float = 120.0):
    """Regularised linear stacker with honest level-2 OOF (same folds). Returns (oof2, test2, info) or None.
    Aborts (returns None) when the projected cost exceeds time_budget."""
    import time
    t_start = time.time()
    try:
        if task == "regression":
            from sklearn.linear_model import Ridge
            Z = np.column_stack(oofs)
            Zt = np.column_stack(tests)
            best = None
            for alpha in (0.1, 1.0, 10.0, 100.0):
                oof2 = np.zeros(len(y))
                for tr, va in folds:
                    r = Ridge(alpha=alpha, positive=True).fit(Z[tr], y[tr])
                    oof2[va] = r.predict(Z[va])
                s = objective(y, oof2)
                if best is None or s > best[0]:
                    best = (s, alpha, oof2)
            r = Ridge(alpha=best[1], positive=True).fit(Z, y)
            return best[2], r.predict(Zt), {"alpha": best[1], "coef": r.coef_.tolist(), "objective": best[0]}
        from sklearn.linear_model import LogisticRegression
        eps = 1e-6
        if task == "binary":
            feats = [np.log(np.clip(p[:, 1], eps, 1 - eps) / np.clip(p[:, 0], eps, 1)) for p in oofs]
            feats_t = [np.log(np.clip(p[:, 1], eps, 1 - eps) / np.clip(p[:, 0], eps, 1)) for p in tests]
            Z, Zt = np.column_stack(feats), np.column_stack(feats_t)
        else:
            Z = np.column_stack([np.log(np.clip(p, eps, 1)) for p in oofs])
            Zt = np.column_stack([np.log(np.clip(p, eps, 1)) for p in tests])
        best = None
        grid = (0.03, 0.3) if Z.shape[0] * Z.shape[1] * max(2, n_classes) < 2e7 else (0.1,)
        n_fits = len(grid) * len(folds) + 1
        for C in grid:
            oof2 = np.zeros((len(y), n_classes))
            for i, (tr, va) in enumerate(folds):
                if len(np.unique(y[tr])) < n_classes:
                    return None
                t0 = time.time()
                lr = LogisticRegression(C=C, max_iter=300, tol=1e-3)
                lr.fit(Z[tr], y[tr])
                oof2[va] = lr.predict_proba(Z[va])
                if (time.time() - t0) * n_fits > time_budget or time.time() - t_start > time_budget:
                    LOG.info("stacking skipped (too slow for the remaining time)")
                    return None
            s = objective(y, oof2)
            if best is None or s > best[0]:
                best = (s, C, oof2)
        lr = LogisticRegression(C=best[1], max_iter=300, tol=1e-3).fit(Z, y)
        return best[2], lr.predict_proba(Zt), {"C": best[1], "objective": best[0]}
    except Exception as e:
        LOG.warning(f"stacking failed: {e}")
        return None
