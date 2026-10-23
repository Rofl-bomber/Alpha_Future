"""Отчёт Аналитика во время поиска: статичная часть (S), реестр гипотез (D2) и журнал изменений.

Статичная часть — то, что пишет Аналитик: часть A (полная и короткая) и JSON (validation,
dependencies, traps, hypotheses, rejected, open_questions). Внутри подэтапа она не меняется;
меняет её только повторный вызов Аналитика (apply_delta).

Реестр — статус каждой гипотезы (H..), отклонённой идеи (R..) и ловушки (T..): открыта /
проверена в узле X с Δ, принята или отклонена экспериментом / снята Аналитиком. Статус ведёт
харнесс без LLM: узел, у гипотезы которого есть ref, обновляет статус пункта (on_result).
Пометки Исследователя «устарело» записываются как замечания, статус не меняют.

Каждое изменение пишется в report_changes.jsonl: что, кем, почему. Ничего не удаляется молча:
снятый пункт остаётся в реестре со статусом и причиной, но не идёт в промпт как открытый.
"""

import json
import time
from pathlib import Path

OPEN = "открыта"


class Registry:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.items = {}  # id → {kind, what, evidence, estimate, status, results, notes, source}
        self.order = []
        self.json = {}
        self.part_a = ""
        self.short = ""
        self.updates = []  # тексты обновлений Аналитика (повторные вызовы)
        self.log_path = self.workdir / "report_changes.jsonl"

    # ---------------------------------------------------------------- загрузка
    def load_report(self, report: dict | None, short: str = ""):
        report = report or {}
        self.part_a = report.get("part_a", "") or ""
        self.short = short or ""
        data = dict(report.get("json") or {})
        traps = []
        for i, t in enumerate(data.get("traps") or [], 1):
            t = dict(t) if isinstance(t, dict) else {"issue": str(t)}
            t.setdefault("id", f"T{i}")
            traps.append(t)
        data["traps"] = traps
        self.json = data
        for h in data.get("hypotheses") or []:
            self._add(h, "H", "Аналитик")
        for r in data.get("rejected") or []:
            self._add(r, "R", "Аналитик")
        for t in traps:
            self._add(t, "T", "Аналитик")
        self._log("load", {"items": len(self.items)}, "Аналитик")
        self.save()

    def _add(self, item: dict, kind: str, source: str):
        iid = str(item.get("id") or f"{kind}{len(self.order) + 1}")
        while iid in self.items:
            iid += "_"
        item["id"] = iid
        what = item.get("what") or item.get("issue") or ""
        if kind == "T" and item.get("columns"):
            what = f"{item.get('columns')}: {what}"
        self.items[iid] = {"kind": kind, "what": str(what), "evidence": str(item.get("evidence") or ""),
                           "estimate": item.get("preliminary_estimate"), "reason": str(item.get("reason") or ""),
                           "status": OPEN, "results": [], "notes": [], "source": source}
        self.order.append(iid)
        return iid

    @property
    def available(self) -> bool:
        return bool(self.items or self.part_a)

    # ---------------------------------------------------------------- статусы
    def on_result(self, ref, node_id: str, status: str, delta, stage: str):
        """Результат эксперимента, проверявшего пункт ref (H.., R.., T..)."""
        if not ref or str(ref) not in self.items:
            return
        it = self.items[str(ref)]
        d = f"Δ={delta:+.4g}" if isinstance(delta, (int, float)) else "без оценки"
        it["results"].append({"node": node_id, "status": status, "delta": delta, "stage": stage})
        if status == "accepted":
            it["status"] = f"принята в {node_id} ({d})"
        elif not any(r["status"] == "accepted" for r in it["results"]):
            word = {"rejected": "отклонена экспериментом", "screen_rejected": "отклонена на скрининге",
                    "failed": "не реализована", "invalid": "не реализована", "flagged": "отклонена проверкой",
                    "truncated": "не досчитана"}.get(status, status)
            it["status"] = f"{word} в {node_id} ({d})"
        self._log("result", {"id": ref, "node": node_id, "status": status, "delta": delta}, "харнесс")
        self.save()

    def mark_obsolete(self, marks: list, source: str = "Исследователь"):
        for m in marks or []:
            if not isinstance(m, dict):
                m = {"item": str(m)}
            iid = str(m.get("item") or m.get("id") or "").strip()
            reason = str(m.get("reason") or "")[:300]
            if iid in self.items:
                self.items[iid]["notes"].append(f"{source}: устарело — {reason}")
            self._log("obsolete", {"item": iid, "reason": reason}, source)
        if marks:
            self.save()

    # ---------------------------------------------------------------- дельта Аналитика
    def apply_delta(self, delta: dict, source: str):
        if not delta:
            return 0
        n = 0
        for r in delta.get("retire") or []:
            r = r if isinstance(r, dict) else {"id": str(r)}
            iid = str(r.get("id", ""))
            if iid in self.items:
                self.items[iid]["status"] = f"снята Аналитиком: {str(r.get('reason') or '')[:200]}"
                self._log("retire", r, source)
                n += 1
        for u in delta.get("update") or []:
            if not isinstance(u, dict):
                continue
            iid = str(u.get("id", ""))
            if iid in self.items:
                it = self.items[iid]
                if u.get("evidence"):
                    it["evidence"] = str(u["evidence"])
                if u.get("preliminary_estimate") is not None:
                    it["estimate"] = u["preliminary_estimate"]
                if u.get("what"):
                    it["what"] = str(u["what"])
                self._log("update", u, source)
                n += 1
        add = delta.get("add") or {}
        for h in add.get("hypotheses") or []:
            h = dict(h) if isinstance(h, dict) else {"what": str(h)}
            self.json.setdefault("hypotheses", []).append(h)
            self._log("add", {"id": self._add(h, "H", source), "what": h.get("what")}, source)
            n += 1
        for t in add.get("traps") or []:
            t = dict(t) if isinstance(t, dict) else {"issue": str(t)}
            t.setdefault("id", f"T{sum(1 for i in self.items.values() if i['kind'] == 'T') + 1}")
            self.json.setdefault("traps", []).append(t)
            self._log("add", {"id": self._add(t, "T", source), "what": t.get("issue")}, source)
            n += 1
        for d in add.get("dependencies") or []:
            self.json.setdefault("dependencies", []).append(d)
            self._log("add_dependency", d, source)
            n += 1
        for q in add.get("open_questions") or []:
            self.json.setdefault("open_questions", []).append(q)
            n += 1
        if delta.get("notes"):
            self.updates.append(f"{source}: {delta['notes']}")
        self.save()
        return n

    # ---------------------------------------------------------------- тексты для промптов
    def json_text(self, limit: int = 9000) -> str:
        """JSON Аналитика в текущей версии: со статусами пунктов, без снятых."""
        if not self.json:
            return "Отчёта Аналитика нет."
        data = dict(self.json)
        for key in ("hypotheses", "rejected", "traps"):
            out = []
            for h in data.get(key) or []:
                h = dict(h)
                it = self.items.get(str(h.get("id")))
                if it is not None:
                    if it["status"].startswith("снята"):
                        continue
                    h["status"] = it["status"]
                    if it["evidence"]:
                        h["evidence"] = it["evidence"]
                    if it["estimate"] is not None:
                        h["preliminary_estimate"] = it["estimate"]
                out.append(h)
            data[key] = out
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= limit else text[:limit] + " …[обрезано; полностью — read_report]"

    def text_open(self, max_closed: int = 40) -> str:
        """D2: открытые пункты подробно, закрытые — одной строкой с результатом."""
        if not self.items:
            return "Реестр пуст: отчёта Аналитика нет."
        open_lines, closed = [], []
        kind_name = {"H": "гипотеза", "R": "отклонённая идея", "T": "ловушка"}
        for iid in self.order:
            it = self.items[iid]
            notes = ("; замечания: " + " | ".join(it["notes"][-2:])) if it["notes"] else ""
            if it["status"] == OPEN:
                est = it["estimate"]
                open_lines.append(f"- {iid} [{kind_name[it['kind']]}, открыта] {it['what'][:300]}"
                                  + (f"; основание: {it['evidence'][:200]}" if it["evidence"] else "")
                                  + (f"; причина отклонения: {it['reason'][:200]}" if it["reason"] else "")
                                  + (f"; предварительная оценка (на сырых данных до принятых изменений): {est}"
                                     if est not in (None, "", "null") else "") + notes)
            else:
                closed.append(f"- {iid} [{kind_name[it['kind']]}] {it['what'][:120]} → {it['status']}{notes}")
        lines = ["Открытые пункты:"] + (open_lines or ["  нет"])
        if closed:
            lines.append("Закрытые пункты:")
            lines += closed[-max_closed:]
        if self.updates:
            lines.append("Обновления Аналитика во время поиска:")
            lines += [f"  - {u[:500]}" for u in self.updates[-4:]]
        return "\n".join(lines)

    def full_report(self, limit: int = 40000) -> str:
        text = self.part_a or "Части A нет."
        if self.updates:
            text += "\n\n# Обновления Аналитика во время поиска\n" + "\n".join(f"- {u}" for u in self.updates)
        text += "\n\n# Реестр\n" + self.text_open()
        return text[:limit]

    def item_text(self, iid) -> str:
        it = self.items.get(str(iid or ""))
        if not it:
            return ""
        return f"{iid}: {it['what']}; основание: {it['evidence']}" + (f"; причина: {it['reason']}" if it["reason"] else "")

    def analyst_notes(self, ref=None, limit: int = 3000) -> str:
        """Для Инженера: пункт, на который ссылается гипотеза, плюс ключи (dependencies) и ловушки."""
        parts = []
        if ref and str(ref) in self.items:
            parts.append("Пункт отчёта, который проверяет гипотеза: " + self.item_text(ref))
        deps = self.json.get("dependencies") or []
        if deps:
            parts.append("Зависимости и ключи: " + json.dumps(deps, ensure_ascii=False)[:1500])
        traps = self.json.get("traps") or []
        if traps:
            parts.append("Ловушки: " + json.dumps(traps, ensure_ascii=False)[:1200])
        return "\n".join(parts)[:limit] or "нет"

    # ---------------------------------------------------------------- журнал
    def _log(self, op: str, data, source: str):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 1), "op": op, "source": source, "data": data},
                                   ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def save(self):
        try:
            (self.workdir / "registry.json").write_text(
                json.dumps({"items": self.items, "order": self.order, "json": self.json, "updates": self.updates},
                           ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass
