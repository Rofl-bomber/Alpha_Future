"""Профиль сдвига train/test: статистики колонок по частям, детектор сплита, дубликаты,
adversarial validation. Всё детерминированно, без LLM; результат выбирает схему фолдов (cv.make_folds)
и уходит Исследователю отдельным разделом профиля."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOW_CARD = 50          # «низкокардинальная» колонка: не больше стольких уникальных значений в train
NAN_KEY = "__nan__"

# пороги флагов колонок (ТЗ, блок 1.1)
OUT_OF_RANGE_MIN = 0.5
UNSEEN_MIN = 0.2
MISSING_GAP_MIN = 0.10
SHARE_SHIFT_MIN = 0.15
ADV_SHIFT_MIN = 0.6     # AUC, с которого показываем дрейфующие признаки
ADV_STRONG = 0.75


def fmt_val(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        return f"{float(v):.6g}"
    return str(v)[:40]


def _keys(s: pd.Series) -> pd.Series:
    """Значения как ключи (пропуск — отдельное значение NAN_KEY)."""
    return s.astype(object).where(s.notna(), NAN_KEY)


def _num_stats(v: pd.Series) -> dict:
    v = v.astype(float)
    if not v.notna().any():
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------ колонки

def column_info(spec, data) -> dict[str, dict]:
    """Статистики каждой колонки отдельно для train и test (ТЗ 1.1)."""
    X, n_tr = data.X, data.n_train
    out: dict[str, dict] = {}
    for c in spec.feature_cols:
        s_tr, s_te = X[c].iloc[:n_tr], X[c].iloc[n_tr:]
        kind = "cat" if c in spec.cat_cols else "num"
        info: dict = {
            "kind": kind,
            "missing_train": float(s_tr.isna().mean()) if n_tr else 0.0,
            "missing_test": float(s_te.isna().mean()) if len(s_te) else 0.0,
            "nunique_train": int(s_tr.nunique()),
            "nunique": int(X[c].nunique()),
            "unseen": 0.0,
            "tvd": None,
            "top": [],
            "out_of_range": 0.0,
        }
        if kind == "num":
            vtr, vte = s_tr.astype(float), s_te.astype(float)
            info["train"] = _num_stats(vtr)
            info["test"] = _num_stats(vte)
            te_nn = vte.dropna()
            if vtr.notna().any() and len(te_nn):
                lo, hi = float(vtr.min()), float(vtr.max())
                info["out_of_range"] = float(((te_nn < lo) | (te_nn > hi)).mean())
        low = kind == "cat" or info["nunique_train"] <= LOW_CARD
        info["low_card"] = bool(low)
        if low and len(s_te) and n_tr:
            k_tr, k_te = _keys(s_tr), _keys(s_te)
            tr_set = set(s_tr.dropna().unique())
            te_nn = s_te.dropna()
            info["unseen"] = float((~te_nn.isin(tr_set)).sum() / len(s_te)) if len(te_nn) else 0.0
            p = k_tr.value_counts(normalize=True)
            q = k_te.value_counts(normalize=True)
            idx = p.index.union(q.index, sort=False)
            pv = p.reindex(idx, fill_value=0.0).values.astype(float)
            qv = q.reindex(idx, fill_value=0.0).values.astype(float)
            info["tvd"] = float(0.5 * np.abs(pv - qv).sum())
            info["top"] = [(fmt_val(v), float(p.get(v, 0.0)), float(q.get(v, 0.0))) for v in p.index[:5]]
        out[c] = info
    return out


# ------------------------------------------------------------------ детектор сплита

def detect_split(spec, data, info: dict, unseen_min: float = 0.95) -> tuple[str | None, str | None, float]:
    """Колонка, по которой train и test разрезаны целыми группами (ТЗ 1.2)."""
    cands = []
    for c in spec.feature_cols:
        i = info[c]
        ntr = i["nunique_train"]
        if not (2 <= ntr <= LOW_CARD):
            continue
        if ntr >= 3 and i["unseen"] >= unseen_min:
            cands.append((i["unseen"], -ntr, c))
    if not cands:
        return None, None, 0.0
    cands.sort(reverse=True)
    unseen, _, col = cands[0]
    kind = "group"
    if col in spec.num_cols:
        n_tr = data.n_train
        vtr = data.X[col].iloc[:n_tr].astype(float).dropna()
        vte = data.X[col].iloc[n_tr:].astype(float).dropna()
        if len(vtr) and len(vte) and (vte.min() > vtr.max() or vte.max() < vtr.min()):
            kind = "time"
    return col, kind, float(unseen)


# ------------------------------------------------------------------ дубликаты

def row_hashes(spec, data) -> np.ndarray:
    X = data.X[spec.feature_cols]
    return pd.util.hash_pandas_object(X.astype(str), index=False).values


def duplicates(spec, data) -> dict:
    """Дубликаты по всем признакам (без id и таргета) и противоречивые метки (ТЗ 1.3)."""
    n_tr = data.n_train
    h = row_hashes(spec, data)
    h_tr, h_te = h[:n_tr], h[n_tr:]
    groups = pd.factorize(pd.Series(h_tr))[0].astype(np.int64)
    out = {
        "groups": groups,
        "share_train": float(pd.Series(h_tr).duplicated().mean()) if n_tr else 0.0,
        "share_test": float(np.isin(h_te, h_tr).mean()) if len(h_te) else 0.0,
        "conflict_groups": 0,
        "conflict_rows_share": 0.0,
        "conflict_by_class": {},
        "conflict_mask": np.zeros(n_tr, dtype=bool),
    }
    if spec.is_clf and n_tr:
        y = np.asarray(data.y)
        df = pd.DataFrame({"g": groups, "y": y})
        nun = df.groupby("g")["y"].nunique()
        bad = nun.index[nun > 1].values
        mask = np.isin(groups, bad)
        out["conflict_groups"] = int(len(bad))
        out["conflict_rows_share"] = float(mask.mean())
        out["conflict_mask"] = mask
        by_class = {}
        for k in range(spec.n_classes):
            m = y == k
            if m.any():
                by_class[str(spec.classes[k])] = float(mask[m].mean())
        out["conflict_by_class"] = by_class
    return out


# ------------------------------------------------------------------ adversarial validation

def adversarial(spec, data, exclude: list[str], max_rows: int = 50000, seed: int = 42,
                n_threads: int = 8) -> dict | None:
    """LightGBM train против test (ТЗ 1.4). Возвращает AUC, топ признаков по gain и
    OOF-вероятность «похожести на test» для каждой строки train."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cols = [c for c in spec.feature_cols if c not in set(exclude)]
    n_tr, n_te = data.n_train, data.n_test
    if not cols or n_tr < 50 or n_te < 50:
        return None
    X = data.X[cols].copy()
    for c in cols:
        if c in spec.cat_cols:
            X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None))
    safe = [f"f{i}" for i in range(len(cols))]
    X.columns = safe
    rng = np.random.RandomState(seed)
    tr_all = np.arange(n_tr)
    te_all = np.arange(n_tr, n_tr + n_te)
    tr_s = np.sort(rng.choice(tr_all, max_rows, replace=False)) if n_tr > max_rows else tr_all
    te_s = np.sort(rng.choice(te_all, max_rows, replace=False)) if n_te > max_rows else te_all
    idx = np.concatenate([tr_s, te_s])
    yy = np.r_[np.zeros(len(tr_s)), np.ones(len(te_s))].astype(int)
    Xs = X.iloc[idx].reset_index(drop=True)
    params = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_jobs=n_threads, random_state=seed, verbose=-1)
    oof = np.zeros(len(idx))
    gains = np.zeros(len(cols))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(yy)), yy):
        m = lgb.LGBMClassifier(**params).fit(Xs.iloc[a], yy[a])
        oof[b] = m.predict_proba(Xs.iloc[b])[:, 1]
        gains += m.booster_.feature_importance(importance_type="gain")
    auc = float(roc_auc_score(yy, oof))
    p_train = np.full(n_tr, np.nan)
    p_train[tr_s] = oof[: len(tr_s)]
    rest = np.setdiff1d(tr_all, tr_s)
    if len(rest):
        m = lgb.LGBMClassifier(**params).fit(Xs, yy)
        p_train[rest] = m.predict_proba(X.iloc[rest])[:, 1]
    tot = gains.sum() or 1.0
    order = np.argsort(-gains)
    top = [(cols[j], round(float(gains[j] / tot), 4)) for j in order[:10]]
    return {"auc": auc, "top": top, "p_train": p_train, "excluded": list(exclude)}


def adv_grade(auc: float | None) -> str:
    if auc is None:
        return "не посчитан"
    if auc < ADV_SHIFT_MIN:
        return "сдвига нет"
    if auc <= ADV_STRONG:
        return "умеренный сдвиг"
    return "сильный сдвиг"


# ------------------------------------------------------------------ отчёт

@dataclass
class ShiftReport:
    split_col: str | None = None
    split_kind: str | None = None          # time | group
    unseen_share: float = 0.0
    split_train_values: list = field(default_factory=list)
    split_test_values: list = field(default_factory=list)
    dup_share_train: float = 0.0
    dup_share_test: float = 0.0
    dup_grouping: bool = False             # группировать фолды по хэшу строки
    conflict_groups: int = 0
    conflict_rows_share: float = 0.0
    conflict_share_by_class: dict = field(default_factory=dict)
    adv_auc: float | None = None
    adv_top: list = field(default_factory=list)
    adv_excluded: list = field(default_factory=list)
    col_info: dict = field(default_factory=dict)
    col_flags: dict = field(default_factory=dict)
    cv_desc: str = ""
    time_s: float = 0.0
    notes: list = field(default_factory=list)
    # массивы — в shift.json не пишутся
    dup_groups: np.ndarray | None = None
    conflict_mask: np.ndarray | None = None
    adv_p: np.ndarray | None = None

    def flagged(self, flag: str, exclude_split: bool = False) -> list[str]:
        return [c for c, fl in self.col_flags.items() if flag in fl and not (exclude_split and c == self.split_col)]

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("dup_groups", "conflict_mask", "adv_p")}
        return d

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def summary(self, max_chars: int = 1500) -> str:
        lines: list[str] = []
        if self.cv_desc:
            lines.append(f"- Схема фолдов: {self.cv_desc}.")
        if self.split_col:
            tr_v = ", ".join(self.split_train_values[:8])
            te_v = ", ".join(self.split_test_values[:8])
            lines.append(f"- Колонка сплита {self.split_col} (тип {self.split_kind}): в test значения [{te_v}], "
                         f"в train [{tr_v}]; новых значений в test {self.unseen_share:.0%}. Колонка остаётся "
                         f"признаком, но в test её значения модель не видела — кандидат на удаление (features.drop).")
        if self.adv_auc is not None:
            excl = f", без {', '.join(self.adv_excluded)}" if self.adv_excluded else ""
            line = f"- Adversarial validation (train против test{excl}): AUC {self.adv_auc:.2f} — {adv_grade(self.adv_auc)}."
            if self.adv_auc >= ADV_SHIFT_MIN and self.adv_top:
                line += " Дрейфующие признаки (доля gain): " + ", ".join(f"{c} {s:.2f}" for c, s in self.adv_top[:5]) + "."
            lines.append(line)
        gap = self.flagged("MISSING_GAP")
        if gap:
            lines.append("- Пропуски train/test различаются: " + ", ".join(
                f"{c} {self.col_info[c]['missing_train']:.2f}/{self.col_info[c]['missing_test']:.2f}" for c in gap[:5]) + ".")
        share = self.flagged("SHARE_SHIFT", exclude_split=True)
        if share:
            share.sort(key=lambda c: -(self.col_info[c].get("tvd") or 0.0))
            lines.append("- Доли значений различаются (TVD): " + ", ".join(
                f"{c} {self.col_info[c]['tvd']:.2f}" for c in share[:5]) + ".")
        unseen = self.flagged("UNSEEN", exclude_split=True)
        if unseen:
            lines.append("- Новые значения в test: " + ", ".join(
                f"{c} {self.col_info[c]['unseen']:.0%} строк" for c in unseen[:5]) + ".")
        oor = self.flagged("OUT_OF_RANGE", exclude_split=True)
        if oor:
            lines.append("- Значения test вне диапазона train: " + ", ".join(
                f"{c} {self.col_info[c]['out_of_range']:.0%}" for c in oor[:5]) + ".")
        if self.dup_share_train >= 0.001 or self.dup_share_test >= 0.001:
            line = (f"- Дубликаты: {self.dup_share_train:.1%} строк train — копии других строк train; "
                    f"{self.dup_share_test:.1%} строк test есть в train.")
            if self.dup_grouping:
                line += " Фолды сгруппированы по хэшу строки, копии не попадают в разные фолды."
            lines.append(line)
        if self.conflict_groups:
            top = sorted(self.conflict_share_by_class.items(), key=lambda kv: -kv[1])[:6]
            lines.append(f"- Одинаковые признаки, разные метки: {self.conflict_groups} групп "
                         f"({self.conflict_rows_share:.1%} строк train); доля таких строк по классам: "
                         + ", ".join(f"{k} {v:.0%}" for k, v in top)
                         + ". Для этих строк признаки не различают классы — это потолок качества.")
        if not lines:
            lines.append("- Признаков сдвига не найдено.")
        out, used = [], 0
        for ln in lines:
            if used + len(ln) + 1 > max_chars:
                break
            out.append(ln)
            used += len(ln) + 1
        return "\n".join(out)


def detect_shift(spec, data, cfg) -> ShiftReport:
    """Полный профиль сдвига (ТЗ, блок 1). cfg: split_unseen, dup_min, adv_rows, seed, n_threads."""
    t0 = time.time()
    rep = ShiftReport()
    info = column_info(spec, data)
    rep.col_info = info

    col, kind, unseen = detect_split(spec, data, info, cfg.split_unseen)
    if col:
        n_tr = data.n_train
        rep.split_col, rep.split_kind, rep.unseen_share = col, kind, unseen
        vtr = data.X[col].iloc[:n_tr].dropna().unique()
        vte = data.X[col].iloc[n_tr:].dropna().unique()
        key = (lambda v: float(v)) if col in spec.num_cols else (lambda v: str(v))
        rep.split_train_values = [fmt_val(v) for v in sorted(vtr, key=key)][:20]
        rep.split_test_values = [fmt_val(v) for v in sorted(vte, key=key)][:20]

    try:
        d = duplicates(spec, data)
        rep.dup_share_train, rep.dup_share_test = d["share_train"], d["share_test"]
        rep.dup_groups = d["groups"]
        rep.dup_grouping = d["share_train"] >= cfg.dup_min
        rep.conflict_groups, rep.conflict_rows_share = d["conflict_groups"], d["conflict_rows_share"]
        rep.conflict_share_by_class, rep.conflict_mask = d["conflict_by_class"], d["conflict_mask"]
    except Exception as e:  # профиль не должен ронять прогон
        rep.notes.append(f"дубликаты не посчитаны: {type(e).__name__}: {e}")

    try:
        adv = adversarial(spec, data, [col] if col else [], cfg.adv_rows, cfg.seed, cfg.n_threads)
        if adv is not None:
            rep.adv_auc, rep.adv_top, rep.adv_p, rep.adv_excluded = adv["auc"], adv["top"], adv["p_train"], adv["excluded"]
    except Exception as e:
        rep.notes.append(f"adversarial validation не посчитана: {type(e).__name__}: {e}")

    drift = {c for c, _ in rep.adv_top[:5]} if (rep.adv_auc or 0.0) >= ADV_SHIFT_MIN else set()
    for c, i in info.items():
        fl = []
        if c == rep.split_col:
            fl.append("SPLIT")
        if i.get("out_of_range", 0.0) >= OUT_OF_RANGE_MIN:
            fl.append("OUT_OF_RANGE")
        if i.get("low_card") and i.get("unseen", 0.0) >= UNSEEN_MIN:
            fl.append("UNSEEN")
        if abs(i["missing_train"] - i["missing_test"]) >= MISSING_GAP_MIN:
            fl.append("MISSING_GAP")
        if i.get("tvd") is not None and i["tvd"] >= SHARE_SHIFT_MIN:
            fl.append("SHARE_SHIFT")
        if c in drift:
            fl.append("DRIFT")
        if fl:
            rep.col_flags[c] = fl
    rep.time_s = round(time.time() - t0, 2)
    return rep
