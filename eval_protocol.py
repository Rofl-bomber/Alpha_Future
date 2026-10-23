"""Протокол оценки: поисковые фолды, selection-holdout и сейф меток.

Разбиение делается один раз за прогон и не меняется. Все кандидаты сравниваются на
одних и тех же фолдах. Основной путь — разбиение Аналитика: holdout он задаёт кодом,
фолды харнесс строит по типу и колонкам (random / stratified / group / ordered) или берёт
готовые (custom); всё приходит через submit_split в Python-сессии (exp_kernel) и
проверяется здесь (check_split). Запасной путь — схема профайлера (make_splits).
Метки selection-holdout хранятся только в LabelVault в процессе оркестратора: дочерние
процессы их не получают. Читает их один модуль — final_ensemble.select_final — в конце.
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

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
    screen_k: int = 0  # >0: быстрый режим оценивает только первые screen_k фолдов основного разбиения

    @property
    def n_folds(self) -> int:
        return int(self.folds.max()) + 1

    def fold_array(self, kind: str):
        """Массив фолдов для режима; для alt — None, если альтернативного разбиения нет."""
        arr = {"full": self.folds, "screen": self.folds_screen, "alt": self.folds_alt}[kind]
        return arr if arr is not None and len(arr) else None

    def eval_folds(self, kind: str) -> list:
        """Какие фолды режим оценивает (обучение — на всех остальных строках поиска)."""
        arr = self.fold_array(kind)
        if arr is None:
            return []
        K = int(arr.max()) + 1
        if kind == "screen" and self.screen_k:
            return list(range(min(self.screen_k, K)))
        return list(range(K))

    def save(self, path):
        alt = self.folds_alt if self.folds_alt is not None else np.array([], dtype=int)
        np.savez(path, search_idx=self.search_idx, holdout_idx=self.holdout_idx, folds=self.folds,
                 folds_screen=self.folds_screen, folds_alt=alt,
                 holdout_weights=self.holdout_weights if self.holdout_weights is not None else np.array([]),
                 meta=np.array([self.scheme, self.group_col or "", self.note, str(self.screen_k)], dtype=object))

    @classmethod
    def load(cls, path) -> "Splits":
        z = np.load(path, allow_pickle=True)
        meta = z["meta"]
        hw = z["holdout_weights"]
        alt = z["folds_alt"]
        return cls(scheme=str(meta[0]), group_col=str(meta[1]) or None, search_idx=z["search_idx"],
                   holdout_idx=z["holdout_idx"], folds=z["folds"], folds_screen=z["folds_screen"],
                   folds_alt=alt if alt.size else None, holdout_weights=hw if hw.size else None,
                   note=str(meta[2]), screen_k=int(meta[3]) if len(meta) > 3 else 0)

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


# ---------------------------------------------------------------- разбиение Аналитика

SPLIT_TYPES = ("random", "stratified", "group", "ordered", "custom")
SPLIT_FILE = "analyst_split.npz"


def group_codes(frame: pd.DataFrame, columns: list) -> np.ndarray:
    """Коды групп по одной или нескольким колонкам (составной ключ); пропуск — отдельное значение."""
    key = None
    for c in columns:
        s = frame[c].astype(object).where(frame[c].notna(), "__nan__").astype(str)
        key = s if key is None else key + "\x1f" + s
    return pd.factorize(key)[0]


def _order_values(s: pd.Series) -> np.ndarray:
    """Значения колонки порядка как числа: число, дата или категория по возрастанию."""
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().mean() >= 0.9:
        return v.to_numpy(dtype=float)
    d = pd.to_datetime(s, errors="coerce", format="mixed")
    if d.notna().mean() >= 0.9:
        return (d - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)
    cats = sorted(s.dropna().astype(str).unique())
    pos = {c: i for i, c in enumerate(cats)}
    return s.astype(str).map(pos).astype(float).to_numpy()


def build_folds(kind: str, frame: pd.DataFrame, y, task_type: str, columns, k: int, seed: int) -> np.ndarray:
    """Фолды по типу схемы для строк frame (поисковая часть): массив номеров 0..K-1."""
    n = len(frame)
    columns = [c for c in (columns or []) if c in frame.columns]
    if kind == "random":
        from sklearn.model_selection import KFold
        f = np.zeros(n, int)
        for i, (_, va) in enumerate(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n))):
            f[va] = i
        return f
    if kind == "group":
        if not columns:
            raise ValueError("для type='group' нужны columns — колонки ключа группы")
        g = group_codes(frame, columns)
        ng = len(np.unique(g))
        if ng < 2:
            raise ValueError("все строки попали в одну группу")
        kk = max(2, min(k, ng))
        perm = np.random.default_rng(seed).permutation(ng)
        sizes = np.bincount(g, minlength=ng)
        # жадно раскладываем группы по фолдам от крупных к мелким, при равенстве — случайный порядок
        order = sorted(range(ng), key=lambda j: (-sizes[j], perm[j]))
        load = np.zeros(kk)
        assign = np.zeros(ng, int)
        for j in order:
            t = int(np.argmin(load))
            assign[j] = t
            load[t] += sizes[j]
        return assign[g]
    if kind == "ordered":
        if not columns:
            raise ValueError("для type='ordered' нужна колонка порядка в columns")
        v = _order_values(frame[columns[0]])
        if np.isnan(v).any():
            raise ValueError(f"в колонке порядка {columns[0]} есть пропуски или нечисловые значения")
        uniq = np.unique(v)
        if len(uniq) < 2:
            raise ValueError(f"в колонке порядка {columns[0]} меньше двух значений")
        blocks = np.array_split(uniq, min(k, len(uniq)))  # фолд = период или подряд идущие периоды
        f = np.zeros(n, int)
        for i, b in enumerate(blocks):
            f[np.isin(v, b)] = i
        return f
    # stratified — по умолчанию
    strata = _strata(np.asarray(y), task_type)
    f = np.zeros(n, int)
    try:
        for i, (_, va) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                                    .split(np.zeros(n), strata)):
            f[va] = i
    except ValueError:
        return build_folds("random", frame, y, task_type, columns, k, seed)
    return f


def resolve_rows(sel, train: pd.DataFrame, id_col) -> np.ndarray:
    """Выбор строк train → булева маска: маска той же длины, иначе id (если есть) или номера строк."""
    n = len(train)
    if isinstance(sel, (pd.Series, pd.Index)):
        sel = sel.to_numpy()
    arr = np.asarray(sel)
    if arr.dtype == bool:
        if arr.shape[0] != n:
            raise ValueError(f"маска holdout длины {arr.shape[0]}, а строк в train {n}")
        return arr.astype(bool)
    vals = arr.ravel()
    if id_col and id_col in train.columns:
        ids = train[id_col]
        mask = ids.astype(str).isin(set(map(str, vals))).to_numpy()
        if mask.sum() == 0:
            raise ValueError(f"ни одно значение не совпало с колонкой id {id_col}")
        return mask
    pos = vals.astype(int)
    if pos.min() < 0 or pos.max() >= n:
        raise ValueError("номера строк вне train")
    mask = np.zeros(n, bool)
    mask[pos] = True
    return mask


def check_split(holdout: np.ndarray, folds: np.ndarray, valid: np.ndarray, y_valid, task_type: str,
                n_classes: int, min_frac: float, max_frac: float) -> tuple:
    """Проверить разбиение по строкам сырого train. Возвращает (ошибки, сводка).

    holdout — булева маска; folds — номер фолда (≥0 для строк поиска); valid — строки с таргетом;
    y_valid — таргет строк valid (классы — любые метки, регрессия — числа).
    """
    errors = []
    n_valid = int(valid.sum())
    ho = holdout & valid
    search = valid & ~holdout
    frac = ho.sum() / max(1, n_valid)
    if ho.sum() == 0:
        errors.append("holdout пустой")
    elif not (min_frac <= frac <= max_frac):
        errors.append(f"доля holdout {frac:.1%} вне допустимых {min_frac:.0%}–{max_frac:.0%} строк с таргетом")
    f = folds[search]
    if (f < 0).any():
        errors.append(f"{int((f < 0).sum())} строк поиска без фолда")
    uniq = np.unique(f[f >= 0])
    if len(uniq) < 2:
        errors.append("нужно хотя бы 2 фолда")
    sizes = {int(u): int((f == u).sum()) for u in uniq}
    min_size = max(20, int(0.02 * search.sum()))
    small = [u for u, c in sizes.items() if c < min_size]
    if small:
        errors.append(f"фолды-крошки (меньше {min_size} строк): {small}")
    yv = pd.Series(np.asarray(y_valid))
    y_full = pd.Series([None] * len(valid), dtype=object)
    y_full[np.where(valid)[0]] = yv.to_numpy()
    lines = [f"строк с таргетом {n_valid}; holdout {int(ho.sum())} ({frac:.1%}); поиск {int(search.sum())}, "
             f"фолдов {len(uniq)}: " + ", ".join(f"{u}={c}" for u, c in sizes.items())]
    if task_type in ("binary", "multiclass"):
        ys = y_full[search].astype(str).to_numpy()
        classes = sorted(set(y_full[valid].astype(str)))
        missing_search = [c for c in classes if (ys == c).sum() == 0]
        if missing_search:
            errors.append(f"классы {missing_search[:10]} есть только в holdout — в поиске их нет")
        for u in uniq:
            tr_cls = set(ys[f != u])
            lost = [c for c in classes if c not in tr_cls and c not in missing_search]
            if lost:
                errors.append(f"в обучающей части фолда {u} нет классов {lost[:10]}")
        yh = y_full[ho].astype(str)
        lines.append("доли классов: поиск " + ", ".join(f"{c}={(ys == c).mean():.3f}" for c in classes[:12])
                     + "; holdout " + ", ".join(f"{c}={(yh == c).mean():.3f}" for c in classes[:12]))
    elif task_type == "regression":
        yn = pd.to_numeric(y_full, errors="coerce")
        parts = [f"holdout {yn[ho].mean():.4g}"] + [f"фолд {u} {yn[np.where(search)[0][f == u]].mean():.4g}"
                                                    for u in uniq]
        lines.append("среднее таргета: " + ", ".join(parts))
    return errors, "\n".join(lines)


def save_analyst_split(workdir, holdout, folds, folds_alt, meta: dict):
    np.savez(Path(workdir) / SPLIT_FILE, holdout=holdout.astype(bool), folds=folds.astype(int),
             folds_alt=folds_alt.astype(int) if folds_alt is not None else np.array([], dtype=int),
             meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object))


def splits_from_analyst(spec, data, raw_valid: np.ndarray, workdir, cfg) -> tuple | None:
    """Разбиение Аналитика (файл submit_split) → (Splits, LabelVault) или None, если файла нет или он плох."""
    path = Path(workdir) / SPLIT_FILE
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    holdout, folds_raw = z["holdout"], z["folds"]
    alt_raw = z["folds_alt"] if z["folds_alt"].size else None
    if len(holdout) != len(raw_valid) or len(folds_raw) != len(raw_valid):
        log.warning("разбиение Аналитика: длина массивов не совпадает с train — не используется")
        return None
    pos = np.where(raw_valid)[0]  # позиции строк данных в сыром train
    ho_data = holdout[pos]
    holdout_idx = np.where(ho_data)[0]
    search_idx = np.where(~ho_data)[0]
    f = folds_raw[pos][search_idx]
    if len(search_idx) == 0 or (f < 0).any():
        log.warning("разбиение Аналитика: у строк поиска нет фолдов — не используется")
        return None
    _, folds = np.unique(f, return_inverse=True)
    folds_alt = None
    if alt_raw is not None:
        a = alt_raw[pos][search_idx]
        if (a >= 0).all() and len(np.unique(a)) >= 2:
            folds_alt = np.unique(a, return_inverse=True)[1]
    kind = meta.get("type", "custom")
    cols = meta.get("columns") or []
    splits = Splits(scheme=f"analyst:{kind}", group_col=",".join(cols) or None, search_idx=search_idx,
                    holdout_idx=holdout_idx, folds=folds, folds_screen=folds.copy(), folds_alt=folds_alt,
                    note="разбиение Аналитика (submit_split)", screen_k=cfg.screen_folds)
    return splits, LabelVault(data.y[holdout_idx])


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
