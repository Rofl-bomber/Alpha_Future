"""Профиль данных: типы, пропуски, кардинальность, распределение таргета, важности признаков из L1.
Статистики колонок считаются отдельно для train и test (ТЗ 1.1), флаги сдвига — из shift.ShiftReport.
Сводка для Исследователя — до ~3 тыс. токенов плюс раздел «Сдвиг train/test»."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .shift import column_info


@dataclass
class ColProfile:
    name: str
    kind: str                 # num | cat
    missing_train: float
    missing_test: float
    nunique: int
    examples: list
    stats_train: dict = field(default_factory=dict)
    stats_test: dict = field(default_factory=dict)
    top: list = field(default_factory=list)          # [(значение, доля train, доля test)]
    unseen: float = 0.0
    low_card: bool = False
    flags: list = field(default_factory=list)
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
    shift: object = None

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

    @staticmethod
    def _range(st: dict) -> str:
        if not st or st.get("mean") is None or (isinstance(st["mean"], float) and np.isnan(st["mean"])):
            return "пусто"
        return f"mean={st['mean']:.4g} [{st['min']:.4g}..{st['max']:.4g}]"

    def _row(self, c: ColProfile) -> str:
        if c.kind == "num":
            info = f"train {self._range(c.stats_train)} / test {self._range(c.stats_test)}"
            if c.low_card and c.top:
                info += "; топ (train/test): " + ", ".join(f"{v} {a:.2f}/{b:.2f}" for v, a, b in c.top[:3])
        else:
            info = "топ (train/test): " + ", ".join(f"{v} {a:.2f}/{b:.2f}" for v, a, b in c.top[:5])
        if c.low_card and c.unseen > 0:
            info += f"; новых в test {c.unseen:.0%}"
        flags = ",".join(c.flags) if c.flags else "-"
        desc = (c.description or "")[:90]
        return (f"- {c.name} | {c.kind} | {c.missing_train:.2f}/{c.missing_test:.2f} | {c.nunique} | "
                f"{info} | {flags} | {desc}")

    def summary(self, with_brief: bool = True) -> str:
        lines = ([self.spec_brief] if with_brief else []) + [f"Строк: train {self.n_train}, test {self.n_test}."]
        t = self.target
        if t.get("kind") == "regression":
            lines.append("Таргет: mean={mean:.4g}, std={std:.4g}, min={min:.4g}, p50={p50:.4g}, max={max:.4g}, "
                         "skew={skew:.3g}, доля <=0: {nonpos:.3f}".format(**t))
        else:
            dist = ", ".join(f"{k}: {v:.3f}" for k, v in t["dist"].items())
            lines.append(f"Таргет, доли классов: {dist}")
        if self.shift is not None:
            lines.append("Сдвиг train/test:")
            lines.append(self.shift.summary())
        if self.importances:
            top = list(self.importances.items())[:15]
            lines.append("Важности L1 (gain, доля): " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        lines.append("Колонки (имя | тип | пропуски train/test | уникальных | статистика train / test | флаги | описание):")
        cols = list(self.columns)
        if self.importances:
            cols.sort(key=lambda c: -self.importances.get(c.name, 0.0))
        # порядок: топ-15 по важности, затем колонки с флагами сдвига, затем остальные —
        # чтобы при обрезке по max_chars не потерять колонки со сдвигом
        flagged = [c for c in cols if c.flags]
        rest = [c for c in cols if not c.flags]
        top_imp = cols[:15]
        ordered = top_imp + [c for c in flagged if c not in top_imp] + [c for c in rest if c not in top_imp]
        header = "\n".join(lines)
        body, used = [], len(header)
        for c in ordered:
            row = self._row(c)
            if used + len(row) + 1 > self.max_chars:
                body.append(f"... ещё {len(ordered) - len(body)} колонок не показано")
                break
            body.append(row)
            used += len(row) + 1
        return header + "\n" + "\n".join(body)


def _fmt_val(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)[:30]


def profile(spec, data, max_chars: int = 9000, shift=None) -> Profile:
    X, n_tr = data.X, data.n_train
    info = shift.col_info if (shift is not None and shift.col_info) else column_info(spec, data)
    flags = shift.col_flags if shift is not None else {}
    cols = []
    for c in spec.feature_cols:
        i = info[c]
        s = X[c]
        cp = ColProfile(
            name=c, kind=i["kind"],
            missing_train=i["missing_train"], missing_test=i["missing_test"],
            nunique=i["nunique"],
            examples=[_fmt_val(v) for v in s.dropna().value_counts().index[:5]],
            stats_train=i.get("train", {}), stats_test=i.get("test", {}),
            top=i.get("top", []), unseen=i.get("unseen", 0.0), low_card=i.get("low_card", False),
            flags=list(flags.get(c, [])),
            description=spec.descriptions.get(c, ""),
        )
        cols.append(cp)
    if spec.is_clf:
        vc = pd.Series(data.y).value_counts(normalize=True).sort_index()
        target = {"kind": "classification", "dist": {str(spec.classes[i]): float(p) for i, p in vc.items()}}
    else:
        y = pd.Series(data.y)
        target = {"kind": "regression", "mean": y.mean(), "std": y.std(), "min": y.min(), "p50": y.median(),
                  "max": y.max(), "skew": float(y.skew()), "nonpos": float((y <= 0).mean())}
    return Profile(spec_brief=spec.brief(), n_train=n_tr, n_test=len(X) - n_tr, columns=cols, target=target,
                   max_chars=max_chars, shift=shift)
