"""Клиент LLM (Qwen3.8-27B через OpenAI-совместимый API): ретраи, JSON, учёт токенов, журнал вызовов.

Переменные окружения как в baseline.py: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL.
Ответ модели очищается от блоков размышлений. JSON: сначала response_format (если сервер
умеет), затем извлечение блока, починка и проверка pydantic-схемой с повторным запросом.
Если лимит токенов на прогон исчерпан — LLMBudgetExceeded, оркестратор переходит
на детерминированные операторы. У Аналитика (анализ, сжатие отчёта, повторные вызовы) отдельный
лимит token_limit_analyst. В режиме AGENT_UNLIMITED лимитов токенов нет.
"""

import ast
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("agent")


class LLMError(Exception):
    pass


class LLMBudgetExceeded(LLMError):
    pass


THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


ANALYST_ROLES = ("analyst", "analyst_delta", "summarizer")


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def strip_paired_think(text: str) -> str:
    """Убрать только парные блоки <think>…</think>; одиночный «</think>» внутри ответа не трогать.

    Модель иногда пишет несколько шагов в одном ответе через фиктивный «</think>»: для циклов с кодом
    важен ПЕРВЫЙ шаг, а strip_think оставил бы только последний.
    """
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if text.lstrip().startswith("</think>"):
        text = text.lstrip()[len("</think>"):]
    return text.strip()


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def extract_json(text: str):
    """Достать JSON-объект из ответа модели (блок ```json, затем первая «{» … последняя «}»)."""
    text = strip_think(text)
    cands = re.findall(r"```(?:json|JSON)?\s*([\[{].*?[\]}])\s*```", text, re.S)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i: j + 1])
    for c in cands:
        for attempt in (c, _repair(c)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        try:
            obj = ast.literal_eval(_repair(c).replace("true", "True").replace("false", "False").replace("null", "None"))
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
    raise ValueError("в ответе не найден корректный JSON")


class LLMClient:
    def __init__(self, cfg, workdir: Path, deadline_fn=None):
        self.cfg = cfg
        self.model = os.environ.get("OPENAI_MODEL", "")
        self.available = bool(self.model and os.environ.get("OPENAI_API_KEY"))
        self.client = None
        if self.available:
            from openai import OpenAI

            self.client = OpenAI(timeout=cfg.llm_timeout_s, max_retries=0)
        self.log_path = Path(workdir) / "llm_log.jsonl"
        self.deadline_fn = deadline_fn
        self.lock = threading.Lock()
        self.usage = {}
        self.total_in = 0
        self.total_out = 0
        self.json_mode_ok = None
        self.thinking_ctrl_ok = None
        cache = os.environ.get("AGENT_LLM_CACHE")
        self.cache_dir = Path(cache) if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ учёт
    def _account(self, role, pin, pout, dt, ok=True):
        with self.lock:
            u = self.usage.setdefault(role, {"calls": 0, "in": 0, "out": 0, "errors": 0, "time_s": 0.0})
            u["calls"] += 1
            u["in"] += pin
            u["out"] += pout
            u["time_s"] += dt
            if not ok:
                u["errors"] += 1
            self.total_in += pin
            self.total_out += pout

    def _totals(self, analyst: bool):
        with self.lock:
            rows = [u for r, u in self.usage.items() if (r in ANALYST_ROLES) == analyst]
        return sum(u["in"] for u in rows), sum(u["out"] for u in rows)

    def budget_left(self, role: str | None = None) -> bool:
        if self.cfg.unlimited:
            return True
        if role in ANALYST_ROLES:
            pin, pout = self._totals(True)
            return (pin + pout) < self.cfg.token_limit_analyst
        pin, pout = self._totals(False)
        return (pin + pout) < self.cfg.token_limit_total and pout < self.cfg.token_limit_out

    def _log(self, rec):
        try:
            with self.lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ вызовы
    def _time_left(self, timeout):
        """Сколько секунд можно ждать ответа. В режиме без ограничений — до дедлайна вызывающей стороны."""
        t_left = float("inf") if self.cfg.unlimited else self.cfg.llm_timeout_s
        if self.deadline_fn:
            t_left = min(t_left, self.deadline_fn() - time.time())
        if timeout:
            t_left = min(t_left, timeout)
        return t_left

    def chat(self, messages, role: str, max_tokens: int | None = 4000, temperature=None, json_mode=False,
             thinking=None, timeout=None, strip: bool = True) -> str:
        """Один ответ модели.

        Сетевые и серверные ошибки — до cfg.llm_retries попыток с паузой. Если ответ обрезан по длине
        (finish_reason="length") или пуст, делается ровно один повтор: без размышлений и, если лимит
        длины задан, с удвоенным max_tokens. В режиме AGENT_UNLIMITED max_tokens не передаётся.
        """
        if not self.available:
            raise LLMError("LLM не настроена: нет OPENAI_MODEL или OPENAI_API_KEY")
        if not self.budget_left(role):
            raise LLMBudgetExceeded(f"исчерпан лимит токенов ({role})")
        if self.cfg.unlimited:
            max_tokens = None
        temperature = self.cfg.temperature if temperature is None else temperature
        key = None
        if self.cache_dir:
            key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, json_mode],
                                            ensure_ascii=False).encode()).hexdigest()
            hit = self.cache_dir / f"{key}.json"
            if hit.exists():
                return json.loads(hit.read_text(encoding="utf-8"))["text"]
        last_err = None
        errors = 0
        short_retry_done = False  # единственный повтор для обрезанного или пустого ответа
        while True:
            t_left = self._time_left(timeout)
            if t_left < 5:
                raise LLMError((last_err + "; " if last_err else "") + "нет времени на вызов LLM")
            kwargs = dict(model=self.model, messages=messages, temperature=temperature)
            if max_tokens:
                kwargs["max_tokens"] = int(max_tokens)
            extra = {}
            if json_mode and self.json_mode_ok is not False:
                kwargs["response_format"] = {"type": "json_object"}
            if thinking is False and self.thinking_ctrl_ok is not False:
                # vLLM/SGLang читают chat_template_kwargs, часть облачных провайдеров — enable_thinking
                extra["chat_template_kwargs"] = {"enable_thinking": False}
                extra["enable_thinking"] = False
            if extra:
                kwargs["extra_body"] = extra
            t0 = time.time()
            try:
                timeout_s = None if t_left == float("inf") else t_left
                resp = self.client.with_options(timeout=timeout_s).chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                name = type(e).__name__
                last_err = f"{name}: {str(e)[:300]}"
                self._account(role, 0, 0, dt, ok=False)
                self._log({"t": round(t0, 1), "role": role, "dt": round(dt, 1), "error": last_err,
                           "max_tokens": max_tokens, "thinking": thinking})
                if name == "BadRequestError":
                    changed = False
                    if "response_format" in kwargs and self.json_mode_ok is not True:
                        self.json_mode_ok = False
                        changed = True
                    if "extra_body" in kwargs and self.thinking_ctrl_ok is not True:
                        self.thinking_ctrl_ok = False
                        changed = True
                    if changed:
                        continue
                    raise LLMError(last_err)
                if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                    raise LLMError(last_err)
                errors += 1
                if errors >= self.cfg.llm_retries:
                    raise LLMError(last_err)
                time.sleep(min(20.0, 2.0 * (2 ** (errors - 1))))
                continue
            dt = time.time() - t0
            if "response_format" in kwargs:
                self.json_mode_ok = True
            if "extra_body" in kwargs:
                self.thinking_ctrl_ok = True
            choice = resp.choices[0]
            msg = choice.message
            finish = getattr(choice, "finish_reason", None)
            raw = msg.content or ""
            text = strip_think(raw) if strip else strip_paired_think(raw)
            reasoning = getattr(msg, "reasoning_content", None) or ""
            usage = getattr(resp, "usage", None)
            pin = getattr(usage, "prompt_tokens", 0) or 0
            pout = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            r_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
            self._account(role, pin, pout, dt)
            self._log({"t": round(t0, 1), "role": role, "dt": round(dt, 1), "in": pin, "out": pout,
                       "reasoning_tokens": r_tokens, "reasoning_chars": len(reasoning) + (len(raw) - len(text)),
                       "finish_reason": finish, "max_tokens": max_tokens, "thinking": thinking,
                       "prompt_tail": messages[-1]["content"][-1500:], "response": text[:6000]})
            truncated = finish == "length"
            if (truncated or not text) and not short_retry_done:
                short_retry_done = True
                why = "обрезан по длине" if truncated else "пустой"
                log.warning("LLM (%s): ответ %s (out=%d, finish_reason=%s) — один повтор без размышлений%s",
                            role, why, pout, finish, f", max_tokens={2 * max_tokens}" if max_tokens else "")
                thinking = False
                if max_tokens:
                    max_tokens = 2 * int(max_tokens)
                last_err = f"ответ {why} (finish_reason={finish})"
                continue
            if not text:
                raise LLMError(f"пустой ответ после повтора (finish_reason={finish}, out={pout})")
            if key:
                (self.cache_dir / f"{key}.json").write_text(json.dumps({"text": text}, ensure_ascii=False),
                                                            encoding="utf-8")
            return text

    def chat_json(self, messages, role: str, schema=None, max_tokens: int | None = 4000, retries: int = 2,
                  thinking=None, timeout=None):
        """Вызов с ответом в JSON; при ошибке разбора или схемы — повторный запрос с текстом ошибки."""
        msgs = list(messages)
        err = None
        for _ in range(retries + 1):
            text = self.chat(msgs, role, max_tokens=max_tokens, json_mode=True, thinking=thinking, timeout=timeout)
            try:
                obj = extract_json(text)
                return schema.model_validate(obj) if schema is not None else obj
            except Exception as e:  # noqa: BLE001
                err = str(e)[:500]
                msgs = msgs + [{"role": "assistant", "content": text[-3000:]},
                               {"role": "user", "content": "Ответ не разобран: " + err
                                + "\nВерни только один корректный JSON-объект по схеме, без текста вокруг."}]
        raise LLMError(f"не удалось получить корректный JSON: {err}")

    def usage_summary(self) -> dict:
        with self.lock:
            return {"total_in": self.total_in, "total_out": self.total_out, "by_role": dict(self.usage)}
