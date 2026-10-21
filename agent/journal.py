"""journal.jsonl — одна строка на эксперимент (и на отфильтрованную гипотезу);
llm_log.jsonl — все вызовы LLM. OOF и test-предсказания всех экспериментов — в preds/."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

KEPT = ("baseline", "keep")
FAILED = ("discard", "critic_reject", "engineer_failed", "error", "filtered")
NOISE = ("noise",)   # выигрыш есть, но в пределах шума по фолдам (1 <= t < t_min): не «провал», а слабый сигнал


def _fmt_t(e: dict) -> str:
    t = e.get("t")
    if t is None:
        return "-" if e.get("se") is None else "se=0"
    return f"{t:.1f}"


def _fmt_seg(e: dict) -> str:
    seg = e.get("target_segment")
    if not seg:
        return "-"
    sd = e.get("segment_delta")
    return f"{seg} {sd:+.4g}" if isinstance(sd, (int, float)) else f"{seg} ?"


def _short(obj, n=160) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


class Journal:
    def __init__(self, run_dir: Path, greater_is_better: bool):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "journal.jsonl"
        self.preds_dir = self.run_dir / "preds"
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.gib = greater_is_better
        self.entries: list[dict] = []
        self._results: dict = {}   # exp_id -> ExpResult (в памяти, для бленда)

    # ------------------------------------------------------------ запись
    def add(self, hypothesis: dict | None, result=None, status: str = "", **extra) -> dict:
        e = {"ts": round(time.time(), 2), "status": status}
        if hypothesis is not None:
            e["hypothesis"] = hypothesis
            e["kind"] = hypothesis.get("kind")
            e["patch"] = hypothesis.get("patch")
        if result is not None:
            e.update({
                "exp_id": result.exp_id, "model": result.config["model"]["name"],
                "config": result.config, "fold_scores": result.fold_scores, "score": result.score,
                "raw_score": result.raw_score, "oof_score": result.oof_score,
                "score_insample": getattr(result, "score_insample", None), "pp_params": result.pp_params,
                "time_s": result.time_s, "error": result.error, "reused_from": result.reused_from,
                "n_features": len(result.feature_names),
            })
            if result.ok:
                self._results[result.exp_id] = result
                np.savez_compressed(self.preds_dir / f"{result.exp_id}.npz", oof_raw=result.oof_raw,
                                    test_raw=result.test_raw, oof=result.oof, test=result.test)
        e.update(extra)
        self.entries.append(e)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        return e

    # ------------------------------------------------------------ чтение
    def _better(self, a: float, b: float) -> bool:
        return a > b if self.gib else a < b

    def evaluated(self) -> list[dict]:
        return [e for e in self.entries if e.get("score") is not None]

    def best(self) -> dict | None:
        cands = [e for e in self.entries if e["status"] in KEPT and e.get("score") is not None]
        if not cands:
            return None
        best = cands[0]
        for e in cands[1:]:
            if self._better(e["score"], best["score"]):
                best = e
        return best

    def result(self, exp_id: str):
        return self._results.get(exp_id)

    def ranked(self, statuses=None) -> list[dict]:
        ev = [e for e in self.evaluated() if statuses is None or e["status"] in statuses]
        return sorted(ev, key=lambda e: e["score"], reverse=self.gib)

    def summary(self, k: int = 10) -> str:
        lines = []
        b = self.best()
        base = next((e for e in self.entries if e["status"] == "baseline"), None)
        if b:
            lines.append(f"Текущий лучший: {b['exp_id']} score={b['score']:.6g}"
                         + (f" (L1 {base['score']:.6g})" if base and base.get("score") is not None else ""))
        top = self.ranked()[:k]
        if top:
            lines.append(f"Топ-{len(top)} экспериментов (exp_id | тип | модель | статус | score | дельта | t | "
                         f"сегмент | идея):")
            for e in top:
                h = e.get("hypothesis") or {}
                d = e.get("delta")
                lines.append(f"- {e['exp_id']} | {e.get('kind') or 'L1'} | {e.get('model')} | {e['status']} | "
                             f"{e['score']:.6g} | {('%+.4g' % d) if d is not None else '-'} | {_fmt_t(e)} | "
                             f"{_fmt_seg(e)} | {_short(h.get('rationale', ''), 90)} | "
                             f"patch={_short(e.get('patch') or {}, 120)}")
        noise = [e for e in self.entries if e["status"] in NOISE][-8:]
        if noise:
            lines.append("На грани шума (выигрыш есть, но t ниже порога; идею можно усилить или совместить с другой):")
            for e in noise:
                h = e.get("hypothesis") or {}
                what = h.get("feature_spec") or e.get("patch")
                d = e.get("delta")
                lines.append(f"- {e.get('exp_id')} [{e.get('kind')}] {_short(what, 120)}"
                             + (f" (дельта {d:+.4g}, t {_fmt_t(e)})" if d is not None else "")
                             + (f"; сегмент {_fmt_seg(e)}" if e.get("target_segment") else ""))
        failed = [e for e in self.entries if e["status"] in FAILED][-15:]
        if failed:
            lines.append("Неудачные идеи (не повторять):")
            for e in failed:
                h = e.get("hypothesis") or {}
                what = h.get("feature_spec") or e.get("patch")
                reason = e.get("reason") or ""
                d = e.get("delta")
                lines.append(f"- [{e.get('kind')}] {_short(what, 120)} -> {e['status']}"
                             + (f" (дельта {d:+.4g}, t {_fmt_t(e)})" if d is not None else "")
                             + (f"; сегмент {_fmt_seg(e)}" if e.get("target_segment") else "")
                             + (f": {_short(reason, 100)}" if reason else ""))
        errs = [e for e in self.entries if e.get("error")][-3:]
        if errs:
            lines.append("Последние ошибки:")
            for e in errs:
                lines.append(f"- {e.get('exp_id', '?')} [{e.get('kind')}] {_short(str(e['error']).splitlines()[0], 200)}")
        return "\n".join(lines) if lines else "Журнал пуст."


class LLMLog:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "llm_log.jsonl"
        self.tokens: dict[str, dict] = {}

    def log(self, role: str, messages: list, response: str, usage: dict, latency: float, cached: bool,
            attempt: int, valid: bool | None, error: str | None = None, call_id: str = ""):
        t = self.tokens.setdefault(role, {"calls": 0, "prompt": 0, "completion": 0, "cached_calls": 0})
        t["calls"] += 1
        if cached:
            t["cached_calls"] += 1
        else:
            t["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
            t["completion"] += int(usage.get("completion_tokens", 0) or 0)
        rec = {"ts": round(time.time(), 2), "call_id": call_id, "role": role, "attempt": attempt,
               "messages": messages, "response": response, "usage": usage, "latency_s": round(latency, 3),
               "cached": cached, "valid": valid, "error": error}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
