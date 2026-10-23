"""Журнал экспериментов (journal.jsonl): память агента между итерациями и материал для защиты.

Одна строка на эксперимент: подэтап (stage), гипотеза, ссылка на гипотезу Аналитика (ref), патч,
что изменено (change), скоры по фолдам, дельта к родителю, время, статус и ошибка. Для LLM журнал
сжимается до непринятых идей и свежих ошибок: текущий подэтап подробно, прошлые — сводкой;
принятые изменения Исследователь видит в блоке «текущее решение».
"""

import json
import threading
import time
from pathlib import Path

import numpy as np


class Journal:
    def __init__(self, path, greater_is_better: bool):
        self.path = Path(path)
        self.gib = greater_is_better
        self.records = []
        self._lock = threading.Lock()

    def add(self, rec: dict) -> dict:
        rec = dict(rec)
        rec.setdefault("t", round(time.time(), 1))
        with self._lock:
            self.records.append(rec)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")
        return rec

    def _gain(self, v):
        return v if self.gib else -v

    def best(self, status=("accepted", "anchor")):
        ok = [r for r in self.records if r.get("status") in status and r.get("mean") is not None]
        return max(ok, key=lambda r: self._gain(r["mean"])) if ok else None

    def summary(self, stage: str | None = None, k_tried: int = 16, k_errors: int = 4) -> str:
        """Что пробовали и что не принято — для Исследователя: текущий подэтап подробно, прошлые — сводкой.

        Принятые изменения здесь не перечисляются: они в блоке «текущее решение» (цепочка).
        Для каждой записи — к какому решению применяли («на nXXX»), что изменили и выигрыш по фолдам.
        """
        bad = ("rejected", "screen_rejected", "flagged", "truncated")
        cur = [r for r in self.records if r.get("status") in bad and (stage is None or r.get("stage") == stage)]
        old = [r for r in self.records if r.get("status") in bad and stage is not None and r.get("stage") != stage]
        errors = [r for r in self.records if r.get("status") in ("failed", "invalid")
                  and (stage is None or r.get("stage") == stage)][-k_errors:]
        if not cur and not errors and not old:
            return "Пока ничего не отклонено."
        lines = []
        if cur:
            lines.append("Не принято в этом подэтапе («на nXXX» — к какому решению применяли; выигрыш > 0 — лучше "
                         "того решения, но ниже порога шума; в скобках — фолды с улучшением из всех):")
            for r in cur[-k_tried:]:
                lines.append("  - " + self._line(r))
        if old:
            lines.append(f"Не принято в прошлых подэтапах ({len(old)}), последние:")
            for r in old[-6:]:
                lines.append(f"  - [{r.get('stage')}] {r['node']}: {_short(r.get('change') or r.get('desc'), 120)} → "
                             + (f"{r['delta']:+.4g}" if isinstance(r.get("delta"), (int, float)) else r.get("status")))
        if errors:
            lines.append("Не удалось выполнить:")
            for r in errors:
                lines.append(f"  - {r['node']}: {_short(r.get('desc'), 100)} → {r.get('status')}: "
                             f"{_short(r.get('reason') or r.get('error'), 160)}")
        return "\n".join(lines)

    @staticmethod
    def _line(r) -> str:
        d = r.get("delta")
        k = len(r.get("fold_scores") or [])
        gain = (f"выигрыш {d:+.4g}" + (f" ({r['wins']}/{k})" if r.get("wins") is not None and k else "")
                if isinstance(d, (int, float)) else "без оценки")
        idea = _short(r.get("desc"), 90)
        change = _short(r.get("change") or r.get("desc"), 220)
        return (f"{r['node']} на {r.get('parent')}: {change} → {gain} [{r.get('status')}"
                + (f": {_short(r.get('reason'), 80)}" if r.get("reason") else "") + "]"
                + (f" (проверка {r['ref']})" if r.get("ref") else "")
                + (f" — идея: {idea}" if r.get("change") and idea and idea != change else ""))

    def stats(self) -> dict:
        st = {}
        for r in self.records:
            st[r.get("status")] = st.get(r.get("status"), 0) + 1
        return st

    def times(self, family: str, mode: str = "full"):
        """Медианное время прогона семейства (обрезанные прогоны тоже учитываются: время они заняли)."""
        ts = [r["time_s"] for r in self.records if r.get("family") == family and r.get("mode") == mode
              and r.get("time_s")]
        return float(np.median(ts)) if ts else None


def _short(text, n=100):
    if not text:
        return ""
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
