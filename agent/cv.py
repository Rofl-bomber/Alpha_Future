"""Фолды задаются один раз за прогон и общие для всех экспериментов.

Схема выбирается по профилю сдвига (shift.ShiftReport, ТЗ 1.5):
- есть колонка сплита -> GroupKFold по ней (n_splits = min(n_folds, число групп));
- колонки сплита нет, но много дубликатов -> StratifiedGroupKFold / GroupKFold по хэшу строки;
- иначе StratifiedKFold / KFold.

inner_split — внутренний сплит обучающей части фолда для early stopping (ТЗ 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


@dataclass
class Folds:
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_id: np.ndarray
    scheme: str
    kind: str = "random"                 # random | group | time | dup | forward
    groups: np.ndarray | None = None     # группа каждой строки train (для групповых схем)
    group_col: str | None = None
    names: list[str] = field(default_factory=list)   # что лежит в валидационной части каждого фолда

    def __len__(self) -> int:
        return len(self.splits)

    def __iter__(self):
        return iter(self.splits)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _strat_labels(y: np.ndarray, n_classes: int, n_folds: int) -> np.ndarray:
    strat = np.asarray(y).copy()
    counts = np.bincount(strat, minlength=max(n_classes, 1))
    rare = np.where((counts > 0) & (counts < n_folds))[0]
    if len(rare):
        strat[np.isin(strat, rare)] = -1  # редкие классы стратифицируем вместе
    return strat


def _fold_id(n: int, splits) -> np.ndarray:
    fold_id = np.full(n, -1, dtype=int)
    for k, (_, va) in enumerate(splits):
        fold_id[va] = k
    return fold_id


def _group_keys(data, col: str) -> pd.Series:
    vals = data.X[col].iloc[: data.n_train].reset_index(drop=True)
    return vals.astype(object).where(vals.notna(), "__nan__")


def make_folds(spec, data, shift=None, n_folds: int = 5, seed: int = 42) -> Folds:
    y = np.asarray(data.y)
    n = len(y)
    idx = np.arange(n)

    if shift is not None and getattr(shift, "split_col", None):
        col = shift.split_col
        keys = _group_keys(data, col)
        codes, uniq = pd.factorize(keys)
        k = int(min(n_folds, len(uniq)))
        splits = list(GroupKFold(n_splits=k).split(idx, groups=codes))
        names = []
        for _, va in splits:
            vs = sorted({_fmt(v) for v in keys.values[va]})
            names.append(f"{col}=" + ",".join(vs[:5]) + ("…" if len(vs) > 5 else ""))
        kind = shift.split_kind or "group"
        scheme = f"GroupKFold({k}) по {col}: в test только значения {col}, которых нет в train (сплит типа {kind})"
        return Folds(splits=splits, fold_id=_fold_id(n, splits), scheme=scheme, kind=kind,
                     groups=codes.astype(np.int64), group_col=col, names=names)

    if shift is not None and getattr(shift, "dup_grouping", False) and shift.dup_groups is not None:
        rng = np.random.RandomState(seed)
        g0 = np.asarray(shift.dup_groups, dtype=np.int64)
        perm = rng.permutation(int(g0.max()) + 1)
        groups = perm[g0]  # случайная перенумерация групп: порядок групп не зависит от порядка строк
        if spec.is_clf:
            strat = _strat_labels(y, spec.n_classes, n_folds)
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(idx, strat, groups))
            scheme = (f"StratifiedGroupKFold({n_folds}, seed={seed}) по хэшу строки: "
                      f"дубликаты ({shift.dup_share_train:.0%} строк train) не разносятся по разным фолдам")
        else:
            splits = list(GroupKFold(n_splits=n_folds).split(idx, groups=groups))
            scheme = (f"GroupKFold({n_folds}) по хэшу строки: "
                      f"дубликаты ({shift.dup_share_train:.0%} строк train) не разносятся по разным фолдам")
        return Folds(splits=splits, fold_id=_fold_id(n, splits), scheme=scheme, kind="dup", groups=groups,
                     names=[f"fold{k}" for k in range(len(splits))])

    if spec.is_clf:
        strat = _strat_labels(y, spec.n_classes, n_folds)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx, strat))
        scheme = f"StratifiedKFold({n_folds}, seed={seed})"
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx))
        scheme = f"KFold({n_folds}, seed={seed})"
    return Folds(splits=splits, fold_id=_fold_id(n, splits), scheme=scheme, kind="random",
                 names=[f"fold{k}" for k in range(len(splits))])


def forward_split(spec, data, shift) -> Folds | None:
    """Один сплит «вперёд» для сплита типа time: обучение на всех группах, кроме последней,
    проверка на последней. Только для сверки в журнале L1 (ТЗ 1.5)."""
    if shift is None or shift.split_kind != "time" or not shift.split_col:
        return None
    col = shift.split_col
    v = data.X[col].iloc[: data.n_train].astype(float).values
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return None
    last = np.nanmax(v) if _is_after(shift) else np.nanmin(v)
    va = np.where(ok & (v == last))[0]
    tr = np.where(~(ok & (v == last)))[0]
    if len(va) == 0 or len(tr) == 0:
        return None
    splits = [(tr, va)]
    return Folds(splits=splits, fold_id=_fold_id(len(v), splits),
                 scheme=f"вперёд: обучение без {col}={_fmt(last)}, проверка на {col}={_fmt(last)}",
                 kind="forward", group_col=col, names=[f"{col}={_fmt(last)}"])


def _is_after(shift) -> bool:
    """test лежит после train (иначе — до)."""
    try:
        return float(shift.split_test_values[0]) > float(shift.split_train_values[-1])
    except (ValueError, IndexError):
        return True


def inner_split(tr: np.ndarray, y: np.ndarray, folds: Folds, frac: float, seed: int,
                is_clf: bool) -> tuple[np.ndarray, np.ndarray]:
    """Позиции внутри tr: (для обучения, для early stopping). ТЗ 4.1.

    При групповой схеме сплит идёт по тем же группам, если их в обучающей части >= 3:
    мало групп (< 20) — одна целая группа с размером, ближайшим к frac; много — случайные группы до frac.
    Иначе — стратифицированный (классификация) или случайный сплит."""
    n = len(tr)
    pos = np.arange(n)
    if frac <= 0 or n < 20:
        return pos, pos
    rng = np.random.RandomState(seed)
    target = max(1, int(round(frac * n)))
    yy = np.asarray(y)[tr] if is_clf else None
    mask = None
    if folds.groups is not None:
        g = np.asarray(folds.groups)[tr]
        uniq, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        if len(uniq) >= 3:
            if len(uniq) < 20:
                j = int(np.argmin(np.abs(cnt - target)))
                es_mask = inv == j
            else:
                chosen, acc = [], 0
                for j in rng.permutation(len(uniq)):
                    if acc >= target:
                        break
                    chosen.append(j)
                    acc += int(cnt[j])
                es_mask = np.isin(inv, chosen)
            if 0 < es_mask.sum() < n:
                mask = es_mask
    if mask is None:
        if is_clf:
            parts = []
            for c in np.unique(yy):
                p = np.where(yy == c)[0]
                m = int(round(frac * len(p)))
                if m >= 1 and len(p) - m >= 1:
                    parts.append(rng.choice(p, m, replace=False))
            es = np.concatenate(parts) if parts else rng.choice(n, target, replace=False)
        else:
            es = rng.choice(n, target, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[es] = True
    if is_clf:
        # классы, которых не осталось в обучающей части, возвращаем из выборки для early stopping:
        # иначе модели (LightGBM, CatBoost) падают на незнакомой метке в eval
        fit_classes = set(np.unique(yy[~mask]).tolist())
        missing = [c for c in np.unique(yy[mask]).tolist() if c not in fit_classes]
        if missing:
            mask = mask & ~np.isin(yy, missing)
    if mask.sum() == 0 or mask.all():
        return pos, pos
    return pos[~mask], pos[mask]
