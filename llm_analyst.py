"""Аналитик: анализ данных в постоянной Python-сессии, приём отчёта, короткая часть A, повторные вызовы.

Анализ (run_analysis). Аналитик пишет код шагами (Thought + один блок ```python```), харнесс
исполняет его в сессии (exp_kernel) и возвращает вывод. Ограничений со стороны харнесса нет,
кроме числа шагов и таймаута шага из промпта и общего потолка времени этапа: по потолку харнесс
просит финальный отчёт. Механика шага:
- исполняется ПЕРВЫЙ блок кода ответа, остальные отбрасываются с предупреждением;
- «Final:» вместе с кодом — сначала исполняется код, модель видит вывод и присылает отчёт снова;
- нет зафиксированного разбиения (submit_split) — один раз просим вызвать его до «Final:»;
- длинная история сжимается порциями: старые выводы сокращаются, свежие идут полностью
  (порциями — чтобы префикс промпта долго не менялся и кэш сервера работал).
Отчёт делится на часть A (текст) и часть B (JSON, проверяется pydantic-схемой; при ошибке —
до двух просьб прислать только исправленный JSON).

Короткая часть A (short_report) — отдельный вызов LLM; числа короткой версии сверяются с полной,
при неудаче — разделы 1–4 с обрезкой.

Повторный вызов (run_delta) — после подэтапов Preprocessing и Features и на плато: та же сессия,
текущее решение уже подгружено; на выходе дельта {retire, update, add, notes}.
"""

import json
import logging
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

import eval_protocol
import llm_client
import llm_prompts

log = logging.getLogger("agent")

PY_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
FINAL_RE = re.compile(r"(?:^|\n)\s*(?:\*\*)?Final(?:\*\*)?\s*:", re.I)
JSON_BLOCK = re.compile(r"```(?:json|JSON)\s*\n(.*?)```", re.S)
COMPRESS_AT_CHARS = 160_000  # история длиннее — старые обмены сокращаются (порцией)


# ---------------------------------------------------------------- схемы отчёта


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return [v]
    return [v]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class Validation(_Lenient):
    type: str = "stratified"
    columns: list = Field(default_factory=list)
    holdout: str = ""
    rejected_schemes: list = Field(default_factory=list)

    @field_validator("columns", "rejected_schemes", mode="before")
    @classmethod
    def _lst(cls, v):
        return _as_list(v)

    @field_validator("type", "holdout", mode="before")
    @classmethod
    def _txt(cls, v):
        return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))


class AnalystJSON(_Lenient):
    validation: Validation = Field(default_factory=Validation)
    dependencies: list = Field(default_factory=list)
    traps: list = Field(default_factory=list)
    hypotheses: list = Field(default_factory=list)
    rejected: list = Field(default_factory=list)
    open_questions: list = Field(default_factory=list)

    @field_validator("dependencies", "traps", "hypotheses", "rejected", "open_questions", mode="before")
    @classmethod
    def _lst(cls, v):
        return _as_list(v)


class DeltaAdd(_Lenient):
    hypotheses: list = Field(default_factory=list)
    dependencies: list = Field(default_factory=list)
    traps: list = Field(default_factory=list)
    open_questions: list = Field(default_factory=list)

    @field_validator("hypotheses", "dependencies", "traps", "open_questions", mode="before")
    @classmethod
    def _lst(cls, v):
        return _as_list(v)


class DeltaOut(_Lenient):
    retire: list = Field(default_factory=list)
    update: list = Field(default_factory=list)
    add: DeltaAdd = Field(default_factory=DeltaAdd)
    notes: str = ""

    @field_validator("retire", "update", mode="before")
    @classmethod
    def _lst(cls, v):
        return _as_list(v)

    @field_validator("notes", mode="before")
    @classmethod
    def _txt(cls, v):
        return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))

    @field_validator("add", mode="before")
    @classmethod
    def _add(cls, v):
        return v if isinstance(v, dict) else {}


def normalize_ids(obj: dict) -> dict:
    """Проставить недостающие id гипотез (H..) и отклонённых идей (R..)."""
    for key, prefix in (("hypotheses", "H"), ("rejected", "R")):
        items, used = [], set()
        for i, h in enumerate(obj.get(key) or [], 1):
            h = dict(h) if isinstance(h, dict) else {"what": str(h)}
            hid = str(h.get("id") or f"{prefix}{i}")
            while hid in used:
                hid = f"{hid}_"
            used.add(hid)
            h["id"] = hid
            items.append(h)
        obj[key] = items
    return obj


# ---------------------------------------------------------------- разбор шагов


def parse_step(text: str) -> dict:
    """{kind: code|final|none, code, n_blocks, final, code_before_final}."""
    m = FINAL_RE.search(text or "")
    if m:
        before, after = text[: m.start()], text[m.end():]
        blocks = [b for b in PY_BLOCK.findall(before) if not b.lstrip().startswith("{")]
        return {"kind": "final", "final": after.strip(), "code": blocks[0].strip() if blocks else "",
                "n_blocks": len(blocks)}
    blocks = [b for b in PY_BLOCK.findall(text or "") if not b.lstrip().startswith("{")]
    if blocks:
        return {"kind": "code", "code": blocks[0].strip(), "n_blocks": len(blocks)}
    return {"kind": "none"}


def split_report(final_text: str) -> tuple:
    """Отчёт → (часть A, JSON-строка части B или None)."""
    blocks = list(JSON_BLOCK.finditer(final_text))
    if blocks:
        b = blocks[-1]
        part_a = final_text[: b.start()]
        raw = b.group(1)
    else:
        i, j = final_text.rfind("\n{"), final_text.rfind("}")
        if i == -1 or j < i:
            return final_text.strip(), None
        part_a, raw = final_text[:i], final_text[i: j + 1]
    part_a = re.sub(r"(?is)\n\s*(?:#+\s*)?(?:\*\*)?Part\s*B.*$", "", part_a).strip()
    return part_a, raw


def parse_json(raw: str, schema):
    obj = llm_client.extract_json("```json\n" + raw + "\n```") if raw else None
    if obj is None:
        raise ValueError("JSON не найден")
    return schema.model_validate(obj).model_dump()


# ---------------------------------------------------------------- общий цикл шагов


class _Loop:
    """Цикл «LLM пишет код — сессия исполняет» с журналом шагов и сжатием длинной истории."""

    def __init__(self, client, kernel, cfg, role: str, log_path: Path, step_timeout: float):
        self.client, self.kernel, self.cfg, self.role = client, kernel, cfg, role
        self.log_path = log_path
        self.step_timeout = step_timeout
        self.messages = []
        self.frozen = 0  # сколько первых обменов уже сокращено

    def _view(self):
        """Сообщения для запроса: старые (до границы frozen) сокращены, свежие — полностью.

        Граница сдвигается порциями, когда история переросла COMPRESS_AT_CHARS: префикс промпта долго
        остаётся неизменным, и кэш сервера продолжает работать.
        """
        msgs = self.messages
        size = sum(len(m["content"]) for m in msgs)
        keep = max(4, 2 * int(self.cfg.analysis_keep_full))
        if size > COMPRESS_AT_CHARS and len(msgs) - 2 - keep > self.frozen:
            self.frozen = len(msgs) - 2 - keep
        out = list(msgs[:2])
        for i, m in enumerate(msgs[2:]):
            if i < self.frozen:
                out.append({"role": m["role"], "content": _shorten(m["content"], 1200 if m["role"] == "assistant"
                                                                    else 500)})
            else:
                out.append(m)
        return out

    def ask(self, deadline: float, max_tokens: int, temperature: float) -> str:
        left = deadline - time.time()
        t0 = time.time()
        text = self.client.chat(self._view(), role=self.role, max_tokens=max_tokens, temperature=temperature,
                                strip=False, timeout=max(10.0, min(self.cfg.llm_timeout_s, left)))
        self._write({"t": round(t0, 1), "dt": round(time.time() - t0, 1), "kind": "llm", "content": text})
        return text

    def run_code(self, code: str, label: str) -> str:
        t0 = time.time()
        out = self.kernel.exec(code, timeout=self.step_timeout, label=label)
        self._write({"t": round(t0, 1), "dt": round(time.time() - t0, 1), "kind": "exec", "label": label,
                     "code": code, "output": out[:6000]})
        return out

    def push(self, assistant: str, user: str):
        self.messages += [{"role": "assistant", "content": assistant}, {"role": "user", "content": user}]

    def _write(self, rec):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _shorten(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n… [сокращено харнессом: ещё {len(text) - n} символов]"


# ---------------------------------------------------------------- анализ данных


def run_analysis(client, kernel, cfg, workdir, task_md: str, spec_text: str, files_text: str, stats: str,
                 deadline: float) -> dict | None:
    """Анализ данных до финального отчёта. Возвращает {part_a, json, split, steps} или None."""
    workdir = Path(workdir)
    system = llm_prompts.ANALYST_SYSTEM.format(max_out=cfg.analysis_max_out, timeout=int(cfg.analysis_step_timeout_s),
                                               steps=cfg.analysis_steps, ho_min=cfg.holdout_min_frac,
                                               ho_max=cfg.holdout_max_frac)
    user = llm_prompts.ANALYST_USER.format(task_md=task_md[:14000], spec=spec_text, files=files_text,
                                           stats=stats[:30000])
    loop = _Loop(client, kernel, cfg, "analyst", workdir / "analyst_log.jsonl", cfg.analysis_step_timeout_s)
    loop.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    split_file = workdir / eval_protocol.SPLIT_FILE
    asked_split = False
    final_text = None
    step_times = []
    steps = 0
    forced = False
    llm_errors = 0
    span = max(60.0, deadline - time.time())
    while steps < cfg.analysis_steps:
        med = sorted(step_times)[len(step_times) // 2] if step_times else 60.0
        margin = min(max(90.0, 2.0 * med), 0.25 * span)  # запас на финальный отчёт
        if time.time() > deadline - margin:
            forced = True
            break
        t0 = time.time()
        try:
            text = loop.ask(deadline, cfg.analysis_step_max_tokens, cfg.analysis_temperature)
        except llm_client.LLMError as e:
            log.warning("Аналитик: вызов LLM не удался: %s", str(e)[:200])
            llm_errors += 1
            if llm_errors >= 3 or time.time() > deadline - 30:
                break
            continue
        llm_errors = 0
        steps += 1
        st = parse_step(text)
        if st["kind"] == "final":
            if st["code"]:
                out = loop.run_code(st["code"], f"step{steps}")
                loop.push(text, llm_prompts.ANALYST_OUTPUT.format(output=out) + "\n\n"
                          + llm_prompts.ANALYST_FINAL_WITH_CODE)
            elif not split_file.exists() and not asked_split:
                asked_split = True
                loop.push(text, llm_prompts.ANALYST_NEED_SPLIT)
            else:
                final_text = st["final"]
                loop.messages.append({"role": "assistant", "content": text})
                break
        elif st["kind"] == "code":
            out = loop.run_code(st["code"], f"step{steps}")
            fb = llm_prompts.ANALYST_OUTPUT.format(output=out)
            if st["n_blocks"] > 1:
                fb += llm_prompts.ANALYST_EXTRA_BLOCKS.format(n=st["n_blocks"])
            if steps == cfg.analysis_steps - 1:
                fb += llm_prompts.ANALYST_LAST_STEP
            loop.push(text, fb)
        else:
            loop.push(text, llm_prompts.ANALYST_BAD_FORMAT)
        step_times.append(time.time() - t0)
        log.info("Аналитик: шаг %d (%s) за %.0f с", steps, st["kind"], time.time() - t0)
    if final_text is None:
        log.info("Аналитик: %s — прошу финальный отчёт", "потолок времени" if forced else "шаги закончились")
        loop.messages.append({"role": "user", "content": llm_prompts.ANALYST_FORCE_FINAL})
        try:
            text = loop.ask(deadline + 600, cfg.analysis_step_max_tokens, cfg.analysis_temperature)
            st = parse_step(text)
            final_text = st["final"] if st["kind"] == "final" else text
            loop.messages.append({"role": "assistant", "content": text})
        except llm_client.LLMError as e:
            log.warning("Аналитик не прислал отчёт: %s", str(e)[:200])
            return None
    part_a, raw = split_report(final_text)
    data, err = None, None
    try:
        data = parse_json(raw, AnalystJSON)
    except Exception as e:  # noqa: BLE001
        err = str(e)[:400]
    for _ in range(2):
        if data is not None:
            break
        loop.messages.append({"role": "user", "content": llm_prompts.ANALYST_FIX_JSON.format(error=err)})
        try:
            text = loop.ask(deadline + 600, 8000, 0.2)
            loop.messages.append({"role": "assistant", "content": text})
            blocks = JSON_BLOCK.findall(text)
            data = parse_json(blocks[-1] if blocks else text, AnalystJSON)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:400]
    if data is None:
        log.warning("Аналитик: часть B не разобрана (%s) — JSON пустой", err)
        data = AnalystJSON().model_dump()
    data = normalize_ids(data)
    (workdir / "analyst_report.md").write_text(final_text, encoding="utf-8")
    (workdir / "analyst_report.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"part_a": part_a, "json": data, "split": split_file.exists(), "steps": steps}


# ---------------------------------------------------------------- короткая часть A

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers_kept(short: str, full: str) -> float:
    nums = [n.replace(",", ".") for n in _NUM.findall(short) if len(n.replace(",", "").replace(".", "")) >= 2]
    if not nums:
        return 1.0
    full_n = full.replace(",", ".")
    return sum(1 for n in nums if n in full_n) / len(nums)


def fallback_short(part_a: str, limit_chars: int) -> str:
    """Разделы 1–4 с обрезкой каждого — если сжатие LLM не удалось."""
    parts = re.split(r"(?m)^(?=\s*(?:#+\s*)?(?:\*\*)?[1-6][.)]\s)", part_a)
    keep = [p.strip() for p in parts if re.match(r"\s*(?:#+\s*)?(?:\*\*)?[1-4][.)]", p)]
    if not keep:
        return part_a[:limit_chars]
    per = max(300, limit_chars // len(keep))
    return "\n\n".join(_shorten(p, per) for p in keep)


def short_report(client, cfg, part_a: str) -> str:
    limit_chars = int(cfg.short_report_tokens * 3.2)
    if not part_a.strip():
        return ""
    if len(part_a) <= limit_chars:
        return part_a
    if client is None or not client.available:
        return fallback_short(part_a, limit_chars)
    msgs = [{"role": "system", "content": llm_prompts.SHORT_SYSTEM.format(tokens=cfg.short_report_tokens)},
            {"role": "user", "content": llm_prompts.SHORT_USER.format(report=part_a)}]
    for attempt in range(2):
        try:
            text = client.chat(msgs, role="summarizer", max_tokens=int(cfg.short_report_tokens * 2), thinking=False,
                               temperature=0.2)
        except llm_client.LLMError as e:
            log.warning("короткая часть A: LLM не ответила (%s)", str(e)[:200])
            break
        kept = _numbers_kept(text, part_a)
        if kept >= 0.9 and len(text) <= limit_chars * 1.5:
            return text.strip()
        log.info("короткая часть A: сохранено чисел %.0f%%, длина %d — %s", 100 * kept, len(text),
                 "повтор" if attempt == 0 else "беру разделы 1–4")
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": "Some numbers are not from the report or the text is too long. "
                                                   "Use only numbers that appear in the report; keep it shorter."}]
    return fallback_short(part_a, limit_chars)


# ---------------------------------------------------------------- повторный вызов


def run_delta(client, kernel, cfg, workdir, context: dict, deadline: float, reason: str) -> dict | None:
    """Повторный вызов Аналитика: дельта к отчёту {retire, update, add, notes} или None."""
    workdir = Path(workdir)
    system = llm_prompts.DELTA_SYSTEM.format(max_out=cfg.analysis_max_out, timeout=int(cfg.analysis_step_timeout_s),
                                             steps=cfg.recall_steps)
    user = llm_prompts.DELTA_USER.format(reason=reason, **context)
    loop = _Loop(client, kernel, cfg, "analyst_delta", workdir / "analyst_delta_log.jsonl",
                 cfg.analysis_step_timeout_s)
    loop.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    final_text = None
    steps = 0
    while steps < cfg.recall_steps and time.time() < deadline - 30:
        try:
            text = loop.ask(deadline, cfg.analysis_step_max_tokens, cfg.analysis_temperature)
        except llm_client.LLMError as e:
            log.warning("Аналитик (дельта): вызов LLM не удался: %s", str(e)[:200])
            break
        steps += 1
        st = parse_step(text)
        if st["kind"] == "final":
            if st["code"]:
                out = loop.run_code(st["code"], f"delta{steps}")
                loop.push(text, llm_prompts.ANALYST_OUTPUT.format(output=out) + "\n\n"
                          + "Your reply contained code and Final; the code was run. Reply with 'Final:' and the JSON.")
                continue
            final_text = st["final"]
            break
        if st["kind"] == "code":
            out = loop.run_code(st["code"], f"delta{steps}")
            fb = llm_prompts.ANALYST_OUTPUT.format(output=out)
            if st["n_blocks"] > 1:
                fb += llm_prompts.ANALYST_EXTRA_BLOCKS.format(n=st["n_blocks"])
            if steps == cfg.recall_steps - 1:
                fb += "\n\nThis was your last code step. The next reply must be 'Final:' with the JSON."
            loop.push(text, fb)
        else:
            loop.push(text, llm_prompts.ANALYST_BAD_FORMAT)
    if final_text is None:
        loop.messages.append({"role": "user", "content": "Time is over. Reply now with 'Final:' and the JSON delta."})
        try:
            text = loop.ask(time.time() + 300, 8000, 0.2)
            st = parse_step(text)
            final_text = st["final"] if st["kind"] == "final" else text
        except llm_client.LLMError:
            return None
    blocks = JSON_BLOCK.findall(final_text)
    try:
        return parse_json(blocks[-1] if blocks else final_text, DeltaOut)
    except Exception as e:  # noqa: BLE001
        log.warning("Аналитик (дельта): JSON не разобран: %s", str(e)[:200])
        return None
