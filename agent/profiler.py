"""Профиль данных: типы, пропуски, кардинальность, распределение таргета,
важности признаков из L1. Сводка для Исследователя — до ~3 тыс. токенов."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColProfile:
    name: str
    kind: str                 # num | cat
    missing_train: float
    missing_test: float
    nunique: int
    examples: list
    stats: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class Profile:
    spec_brief: str
    n_train: int
    n_test: int
    columns: list[ColProfile]
    target: dict
    importances: dict[str, float] = field(default_factory=dict)
    max_chars: int = 9000

    def set_importances(self, imps: dict | None):
        self.importances = dict(sorted((imps or {}).items(), key=lambda kv: -kv[1]))

    def top_numeric(self, k: int = 5) -> list[str]:
        nums = [c.name for c in self.columns if c.kind == "num"]
        if self.importances:
            nums.sort(key=lambda c: -self.importances.get(c, 0.0))
        return nums[:k]

    def top_categorical(self, k: int = 3) -> list[str]:
        cats = [c.name for c in self.columns if c.kind == "cat"]
        if self.importances:
            cats.sort(key=lambda c: -self.importances.get(c, 0.0))
        return cats[:k]

    def summary(self, with_brief: bool = True) -> str:
        lines = ([self.spec_brief] if with_brief else []) + [f"Строк: train {self.n_train}, test {self.n_test}."]
        t = self.target
        if t.get("kind") == "regression":
            lines.append("Таргет: mean={mean:.4g}, std={std:.4g}, min={min:.4g}, p50={p50:.4g}, max={max:.4g}, "
                         "skew={skew:.3g}, доля <=0: {nonpos:.3f}".format(**t))
        else:
            dist = ", ".join(f"{k}: {v:.3f}" for k, v in t["dist"].items())
            lines.append(f"Таргет, доли классов: {dist}")
        if self.importances:
            top = list(self.importances.items())[:15]
            lines.append("Важности L1 (gain, доля): " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        lines.append("Колонки (имя | тип | пропуски train/test | уникальных | статистика/примеры | описание):")
        cols = list(self.columns)
        if self.importances:
            cols.sort(key=lambda c: -self.importances.get(c.name, 0.0))
        header = "\n".join(lines)
        body, used = [], len(header)
        for c in cols:
            if c.kind == "num":
                st = c.stats
                info = f"mean={st['mean']:.4g} std={st['std']:.4g} min={st['min']:.4g} max={st['max']:.4g}"
            else:
                info = "примеры: " + ", ".join(map(str, c.examples[:4]))
            desc = (c.description or "")[:90]
            row = f"- {c.name} | {c.kind} | {c.missing_train:.2f}/{c.missing_test:.2f} | {c.nunique} | {info} | {desc}"
            if used + len(row) + 1 > self.max_chars:
                body.append(f"... ещё {len(cols) - len(body)} колонок не показано")
                break
            body.append(row)
            used += len(row) + 1
        return header + "\n" + "\n".join(body)


def _fmt_val(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)[:30]


def profile(spec, data, max_chars: int = 9000) -> Profile:
    X, n_tr = data.X, data.n_train
    Xtr, Xte = X.iloc[:n_tr], X.iloc[n_tr:]
    cols = []
    for c in spec.feature_cols:
        kind = "cat" if c in spec.cat_cols else "num"
        s = X[c]
        cp = ColProfile(
            name=c, kind=kind,
            missing_train=float(Xtr[c].isna().mean()), missing_test=float(Xte[c].isna().mean()) if len(Xte) else 0.0,
            nunique=int(s.nunique()),
            examples=[_fmt_val(v) for v in s.dropna().value_counts().index[:5]],
            description=spec.descriptions.get(c, ""),
        )
        if kind == "num":
            v = s.astype(float)
            cp.stats = {"mean": float(np.nanmean(v)) if v.notna().any() else float("nan"),
                        "std": float(np.nanstd(v)) if v.notna().any() else float("nan"),
                        "min": float(np.nanmin(v)) if v.notna().any() else float("nan"),
                        "max": float(np.nanmax(v)) if v.notna().any() else float("nan")}
        cols.append(cp)
    if spec.is_clf:
        vc = pd.Series(data.y).value_counts(normalize=True).sort_index()
        target = {"kind": "classification", "dist": {str(spec.classes[i]): float(p) for i, p in vc.items()}}
    else:
        y = pd.Series(data.y)
        target = {"kind": "regression", "mean": y.mean(), "std": y.std(), "min": y.min(), "p50": y.median(),
                  "max": y.max(), "skew": float(y.skew()), "nonpos": float((y <= 0).mean())}
    return Profile(spec_brief=spec.brief(), n_train=n_tr, n_test=len(Xte), columns=cols, target=target,
                   max_chars=max_chars)
