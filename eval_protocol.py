"""Протокол оценки: поисковые фолды, selection-holdout и сейф меток.

Разбиение делается один раз за прогон и не меняется. Все кандидаты сравниваются на
одних и тех же фолдах. Метки selection-holdout хранятся только в LabelVault в процессе
оркестратора: дочерние процессы и LLM их не получают. Читает их один модуль —
final_ensemble.select_final — один раз, в конце.
"""

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit

import task_metrics

log = logging.getLogger("agent")


@dataclass
class Splits:
    scheme: str
    group_col: str | None
    search_idx: np.ndarray
    holdout_idx: np.ndarray
    folds: np.ndarray
    folds_screen: np.ndarray
    folds_alt: np.ndarray
    holdout_weights: np.ndarray | None = None
    note: str = ""

    @property
    def n_folds(self) -> int:
        return int(self.folds.max()) + 1

    def fold_array(self, kind: str) -> np.ndarray:
        return {"full": self.folds, "screen": self.folds_screen, "alt": self.folds_alt}[kind]

    def save(self, path):
        np.savez(path, search_idx=self.search_idx, holdout_idx=self.holdout_idx, folds=self.folds,
                 folds_screen=self.folds_screen, folds_alt=self.folds_alt,
                 holdout_weights=self.holdout_weights if self.holdout_weights is not None else np.array([]),
                 meta=np.array([self.scheme, self.group_col or "", self.note], dtype=object))

    @classmethod
    def load(cls, path) -> "Splits":
        z = np.load(path, allow_pickle=True)
        meta = z["meta"]
        hw = z["holdout_weights"]
        return cls(scheme=str(meta[0]), group_col=str(meta[1]) or None, search_idx=z["search_idx"],
                   holdout_idx=z["holdout_idx"], folds=z["folds"], folds_screen=z["folds_screen"],
                   folds_alt=z["folds_alt"], holdout_weights=hw if hw.size else None, note=str(meta[2]))

    def describe(self) -> str:
        sizes = np.bincount(self.folds)
        return (f"схема={self.scheme}" + (f" по {self.group_col}" if self.group_col else "")
                + f"; поиск={len(self.search_idx)} строк, {self.n_folds} фолдов {sizes.tolist()}; "
                f"selection-holdout={len(self.holdout_idx)} строк"
                + ("; holdout взвешен по сходству с test" if self.holdout_weights is not None else "")
                + (f"; {self.note}" if self.note else ""))


def _strata(y, task_type, min_count=4):
    y = np.asarray(y)
    if task_type in ("binary", "multiclass"):
        s = y.astype(int).copy()
    elif task_type == "multilabel":
        s = y.sum(axis=1).astype(int)
    else:
        v = y if y.ndim == 1 else y[:, 0]
        qs = np.unique(np.quantile(v, np.linspace(0, 1, 11)[1:-1]))
        s = np.searchsorted(qs, v)
    counts = pd.Series(s).value_counts()
    rare = counts[counts < min_count].index
    if len(rare):
        s = np.where(np.isin(s, rare), -1, s)
        if (s == -1).sum() < 2:
            s = np.where(s == -1, counts.idxmax(), s)
    return s


def _choose_holdout_groups(groups: pd.Series, frac: float, time_ordered: bool, rng):
    sizes = groups.value_counts()
    n = len(groups)
    names = list(sizes.index)
    if time_ordered:
        try:
            names = sorted(names, key=float)
        except (TypeError, ValueError):
            names = sorted(names, key=str)
    best, best_score = None, None
    if len(names) <= 15:
        cands = itertools.chain.from_iterable(itertools.combinations(names, r) for r in (1, 2, 3))
    else:
        cands = []
        for _ in range(200):
            perm = rng.permutation(len(names))
            chosen, tot = [], 0
            for i in perm:
                chosen.append(names[i])
                tot += sizes[names[i]]
                if tot >= frac * n:
                    break
            cands.append(tuple(chosen))
    for subset in cands:
        if len(names) - len(subset) < 2:
            continue
        f = sizes[list(subset)].sum() / n
        score = abs(f - frac) + (1.0 if f > 2.5 * frac or f < 0.3 * frac else 0.0)
        if time_ordered:  # при равенстве — более поздние группы
            score -= 1e-6 * max(names.index(g) for g in subset)
        if best_score is None or score < best_score:
            best, best_score = subset, score
    return best


def make_splits(spec, data, prof: dict, cfg) -> tuple:
    """Вернуть (Splits, LabelVault)."""
    rng = np.random.default_rng(cfg.seed)
    y = data.y
    n = len(y)
    frac = cfg.holdout_frac_large if n > cfg.large_rows else cfg.holdout_frac
    if n < 1000:
        frac = max(frac, 0.2)
    Xtr = data.X_all.iloc[: data.n_train]
    K = cfg.n_folds
    strata = _strata(y, spec.task_type)
    scheme, group_col, note = "stratified", None, ""
    holdout_idx = None

    if prof.get("group_col"):
        group_col = prof["group_col"]
        groups = Xtr[group_col].astype(str).fillna("__nan__").reset_index(drop=True)
        chosen = _choose_holdout_groups(groups, frac, prof.get("time_ordered_groups", False), rng)
        if chosen and groups.nunique() - len(chosen) >= 2:
            scheme = "group"
            holdout_idx = np.where(groups.isin(chosen).to_numpy())[0]
            note = f"группы holdout: {list(chosen)}"
        else:
            note = f"групп {groups.nunique()} — мало для группового сплита"
            group_col = None
    elif prof.get("time_col"):
        t = pd.to_datetime(Xtr[prof["time_col"]], errors="coerce", format="mixed")
        if t.notna().mean() > 0.9:
            scheme = "time"
            order = np.argsort(t.fillna(t.min()).to_numpy(), kind="stable")
            n_ho = int(round(frac * n))
            holdout_idx = np.sort(order[n - n_ho:])
            group_col = prof["time_col"]

    if holdout_idx is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=cfg.seed)
        try:
            _, holdout_idx = next(sss.split(np.zeros(n), strata))
        except ValueError:
            holdout_idx = rng.choice(n, size=int(round(frac * n)), replace=False)
        holdout_idx = np.sort(holdout_idx)
        # редкие классы, у которых все объекты ушли в holdout, возвращаем в поиск
        if spec.task_type in ("binary", "multiclass"):
            mask = np.ones(n, bool)
            mask[holdout_idx] = False
            for k in range(spec.n_classes):
                if (y[mask] == k).sum() == 0 and (y == k).sum() > 0:
                    move = holdout_idx[y[holdout_idx] == k]
                    holdout_idx = np.setdiff1d(holdout_idx, move)
    search_idx = np.setdiff1d(np.arange(n), holdout_idx)
    s_strata = strata[search_idx]

    def strat_folds(k, seed):
        f = np.zeros(len(search_idx), int)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (_, va) in enumerate(skf.split(np.zeros(len(search_idx)), s_strata)):
            f[va] = i
        return f

    if scheme == "group":
        g = Xtr[group_col].astype(str).fillna("__nan__").to_numpy()[search_idx]
        ng = len(np.unique(g))

        def group_folds(k, seed=None):
            k = max(2, min(k, ng))
            if seed is None:
                f = np.zeros(len(search_idx), int)
                for i, (_, va) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(g)), groups=g)):
                    f[va] = i
                return f
            uniq = np.unique(g)
            perm = np.random.default_rng(seed).permutation(len(uniq))
            mapping = {u: perm[j] % k for j, u in enumerate(uniq)}
            return np.array([mapping[v] for v in g])

        folds = group_folds(K)
        folds_screen = group_folds(min(cfg.screen_folds, K))
        folds_alt = group_folds(min(cfg.screen_folds, K), seed=cfg.seed + 7)
    elif scheme == "time":
        t = pd.to_datetime(Xtr[group_col], errors="coerce", format="mixed").to_numpy()[search_idx]
        order = np.argsort(t, kind="stable")

        def block_folds(k):
            f = np.zeros(len(search_idx), int)
            for i, chunk in enumerate(np.array_split(order, k)):
                f[chunk] = i
            return f

        folds = block_folds(K)
        folds_screen = block_folds(cfg.screen_folds)
        folds_alt = strat_folds(cfg.screen_folds, cfg.seed + 7)
    else:
        folds = strat_folds(K, cfg.seed)
        folds_screen = strat_folds(cfg.screen_folds, cfg.seed + 1)
        folds_alt = strat_folds(cfg.screen_folds, cfg.seed + 7)

    splits = Splits(scheme=scheme, group_col=group_col, search_idx=search_idx, holdout_idx=holdout_idx,
                    folds=folds, folds_screen=folds_screen, folds_alt=folds_alt, note=note)

    adv = prof.get("adversarial") or {}
    if scheme == "stratified" and adv.get("weights_path") and adv.get("auc") is not None \
            and cfg.adv_shift_auc <= adv["auc"] <= cfg.adv_weight_max_auc:
        odds = np.load(adv["weights_path"])
        w = np.clip(odds[holdout_idx], 0.1, 10.0)
        splits.holdout_weights = w / w.mean()

    vault = LabelVault(y[holdout_idx], splits.holdout_weights)
    return splits, vault


class LabelVault:
    """Сейф меток selection-holdout. Метки не покидают процесс оркестратора до финального выбора."""

    def __init__(self, y_holdout, weights=None):
        self.__y = np.asarray(y_holdout)
        self.__w = weights
        self.opened = False

    def __len__(self):
        return len(self.__y)

    def score(self, metric, pred, task_type, post=None) -> float:
        self.opened = True
        return task_metrics.score(metric, self.__y, pred, task_type, post, sample_weight=self.__w)

    def prob_better(self, metric, pred_a, post_a, pred_b, post_b, task_type, n_boot=300, seed=0) -> float:
        """Доля бутстреп-выборок holdout, где кандидат A лучше B (парный бутстреп)."""
        self.opened = True
        rng = np.random.default_rng(seed)
        y, w = self.__y, self.__w
        out_a = task_metrics.to_needed(pred_a, task_type, metric.need, post_a)
        out_b = task_metrics.to_needed(pred_b, task_type, metric.need, post_b)
        n = len(y)
        wins = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            ww = None if w is None else w[idx]
            sa = metric.fn(y[idx], out_a[idx], ww)
            sb = metric.fn(y[idx], out_b[idx], ww)
            if task_metrics.gain(sa, metric.greater_is_better) > task_metrics.gain(sb, metric.greater_is_better):
                wins += 1
        return wins / n_boot

    def shift_check(self, metric, ho_pred, post, est_prior, train_prior, task_type, n_rep=20, seed=0) -> dict:
        """Пересэмплировать holdout под оценённые доли классов test и сравнить метрику без и с коррекцией."""
        self.opened = True
        y = self.__y.astype(int)
        n = len(y)
        rng = np.random.default_rng(seed)
        K = len(est_prior)
        ho_prior = np.bincount(y, minlength=K) / n
        w = np.asarray(est_prior)[y] / np.maximum(ho_prior[y], 1e-12)
        prob = w / w.sum()
        gains = []
        for _ in range(n_rep):
            idx = rng.choice(n, size=n, replace=True, p=prob)
            p_s = np.asarray(ho_pred)[idx]
            base = metric.fn(y[idx], task_metrics.to_needed(p_s, task_type, metric.need, post, "oof"))
            est_r = task_metrics.em_priors(p_s, train_prior)
            post2 = dict(post, prior=list(est_r))
            corr = metric.fn(y[idx], task_metrics.to_needed(p_s, task_type, metric.need, post2, "test"))
            gains.append(task_metrics.gain(corr, metric.greater_is_better)
                         - task_metrics.gain(base, metric.greater_is_better))
        g = np.asarray(gains)
        return {"win_frac": float((g > 0).mean()), "mean_gain": float(g.mean())}

    def release(self):
        """Метки для refit — только после того, как финальный выбор сделан."""
        return self.__y.copy()
