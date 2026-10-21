"""Журнал экспериментов (journal.jsonl): память агента между итерациями и материал для защиты.

Одна строка на эксперимент: гипотеза, патч, скоры по фолдам, дельта к родителю, время,
статус и ошибка. Для LLM журнал сжимается: топ результатов, последние записи,
проваленные идеи и свежие ошибки.
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

    def summary(self, k_top: int = 10, k_last: int = 6, k_failed: int = 10) -> str:
        full = [r for r in self.records if r.get("mode") == "full" and r.get("mean") is not None]
        top = sorted(full, key=lambda r: -self._gain(r["mean"]))[:k_top]
        lines = ["Лучшие эксперименты (полная оценка):"]
        for r in top:
            lines.append(f"  {r['node']}: {r['mean']:.6g}±{r.get('std', 0):.3g} [{r.get('status')}] "
                         f"{r.get('op')}: {_short(r.get('desc'))}")
        lines.append("Последние записи:")
        for r in self.records[-k_last:]:
            d = r.get("delta")
            lines.append(f"  {r['node']} {r.get('op')} [{r.get('status')}]"
                         + (f" Δ={d:+.4g}" if isinstance(d, (int, float)) else "")
                         + f": {_short(r.get('desc'))}" + (f" — ошибка: {_short(r.get('error'), 120)}"
                                                           if r.get('error') else ""))
        failed = [r for r in self.records if r.get("status") in ("rejected", "screen_rejected", "failed",
                                                                  "flagged")][-k_failed:]
        if failed:
            lines.append("Не сработало (не повторять в том же виде):")
            for r in failed:
                d = r.get("delta")
                lines.append(f"  - {_short(r.get('desc'), 140)} → {r.get('status')}"
                             + (f", Δ={d:+.4g}" if isinstance(d, (int, float)) else "")
                             + (f", {_short(r.get('reason'), 80)}" if r.get("reason") else ""))
        return "\n".join(lines)

    def stats(self) -> dict:
        st = {}
        for r in self.records:
            st[r.get("status")] = st.get(r.get("status"), 0) + 1
        return st

    def times(self, family: str, mode: str = "full"):
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
