"""Клиент LLM (Qwen3.8-27B через OpenAI-совместимый API): ретраи, JSON, учёт токенов, журнал вызовов.

Переменные окружения как в baseline.py: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL.
Ответ модели очищается от блоков размышлений. JSON: сначала response_format (если сервер
умеет), затем извлечение блока, починка и проверка pydantic-схемой с повторным запросом.
Если лимит токенов на прогон исчерпан — LLMBudgetExceeded, оркестратор переходит
на детерминированные операторы.
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


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
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

    def budget_left(self) -> bool:
        return (self.total_in + self.total_out) < self.cfg.token_limit_total and self.total_out < self.cfg.token_limit_out

    def _log(self, rec):
        try:
            with self.lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ вызовы
    def chat(self, messages, role: str, max_tokens: int = 4000, temperature=None, json_mode=False,
             thinking=None, timeout=None) -> str:
        if not self.available:
            raise LLMError("LLM не настроена: нет OPENAI_MODEL или OPENAI_API_KEY")
        if not self.budget_left():
            raise LLMBudgetExceeded("исчерпан лимит токенов на прогон")
        temperature = self.cfg.temperature if temperature is None else temperature
        key = None
        if self.cache_dir:
            key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, json_mode],
                                            ensure_ascii=False).encode()).hexdigest()
            hit = self.cache_dir / f"{key}.json"
            if hit.exists():
                return json.loads(hit.read_text(encoding="utf-8"))["text"]
        last_err = None
        for attempt in range(self.cfg.llm_retries):
            t_left = self.cfg.llm_timeout_s
            if self.deadline_fn:
                t_left = min(t_left, self.deadline_fn() - time.time())
            if timeout:
                t_left = min(t_left, timeout)
            if t_left < 5:
                raise LLMError("нет времени на вызов LLM")
            kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            extra = {}
            if json_mode and self.json_mode_ok is not False:
                kwargs["response_format"] = {"type": "json_object"}
            if thinking is False and self.thinking_ctrl_ok is not False:
                extra["chat_template_kwargs"] = {"enable_thinking": False}
            if extra:
                kwargs["extra_body"] = extra
            t0 = time.time()
            try:
                resp = self.client.with_options(timeout=t_left).chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                name = type(e).__name__
                msg = str(e)[:300]
                self._account(role, 0, 0, dt, ok=False)
                last_err = f"{name}: {msg}"
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
                time.sleep(min(20.0, 2.0 * (2 ** attempt)))
                continue
            dt = time.time() - t0
            if "response_format" in kwargs:
                self.json_mode_ok = True
            if "extra_body" in kwargs:
                self.thinking_ctrl_ok = True
            msg = resp.choices[0].message
            text = strip_think(msg.content or "")
            usage = getattr(resp, "usage", None)
            pin = getattr(usage, "prompt_tokens", 0) or 0
            pout = getattr(usage, "completion_tokens", 0) or 0
            self._account(role, pin, pout, dt)
            self._log({"t": round(t0, 1), "role": role, "dt": round(dt, 1), "in": pin, "out": pout,
                       "prompt_tail": messages[-1]["content"][-1500:], "response": text[:6000]})
            if not text:
                last_err = "пустой ответ"
                continue
            if key:
                (self.cache_dir / f"{key}.json").write_text(json.dumps({"text": text}, ensure_ascii=False),
                                                            encoding="utf-8")
            return text
        raise LLMError(last_err or "LLM не ответила")

    def chat_json(self, messages, role: str, schema=None, max_tokens: int = 4000, retries: int = 2, thinking=None):
        """Вызов с ответом в JSON; при ошибке разбора или схемы — повторный запрос с текстом ошибки."""
        msgs = list(messages)
        err = None
        for _ in range(retries + 1):
            text = self.chat(msgs, role, max_tokens=max_tokens, json_mode=True, thinking=thinking)
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
