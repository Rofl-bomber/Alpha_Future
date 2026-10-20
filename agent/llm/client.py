"""LLM-клиент: OpenAI SDK (как в baseline.py), извлечение JSON, pydantic-валидация,
один повтор с текстом ошибки, счётчик токенов по ролям, кэш ответов по хэшу промпта, mock-режим."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..journal import LLMLog

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:  # открывающий тег мог быть в шаблоне чата
        text = text.split("</think>", 1)[1]
    return text.strip()


def _balanced(text: str, start: int) -> str | None:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> Any:
    """JSON из ответа: блок ```json, иначе первый сбалансированный {...} или [...]."""
    t = strip_think(text)
    cands = re.findall(r"```(?:json)?\s*(.*?)```", t, re.S)
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if starts:  # только с первой скобки: не вытаскиваем внутренние фрагменты оборванного ответа
        b = _balanced(t, min(starts))
        if b:
            cands.append(b)
    last_err = None
    for c in cands:
        c = c.strip()
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as e:
                last_err = e
    raise ValueError(f"не удалось извлечь JSON из ответа: {last_err or 'JSON не найден'}")


def extract_code(text: str) -> str:
    t = strip_think(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    for b in blocks:
        if "def make_features" in b:
            return b.strip()
    if blocks:
        return blocks[0].strip()
    i = t.find("def make_features")
    if i >= 0:
        start = t.rfind("\n", 0, max(0, t.rfind("import", 0, i))) if "import" in t[:i] else i
        return t[max(start, 0):].strip()
    raise ValueError("в ответе нет блока кода с make_features")


class LLMClient:
    def __init__(self, cfg, log: LLMLog, cache_dir: Path, mock=None):
        self.cfg = cfg
        self.log = log
        self.mock = mock
        self.model = os.environ.get("OPENAI_MODEL", "mock" if mock else "")
        self.cache_dir = Path(cache_dir)
        if cfg.llm_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._n = 0
        if mock is None:
            from openai import OpenAI  # клиент как в baseline: ключ и base_url из окружения
            self._client = OpenAI(timeout=cfg.llm_timeout, max_retries=2)
        self.extra_body = json.loads(cfg.llm_extra_body) if cfg.llm_extra_body else None

    # ------------------------------------------------------------ низкий уровень
    def _cache_key(self, role: str, messages: list, temperature: float) -> str:
        payload = json.dumps({"model": self.model, "role": role, "messages": messages, "t": temperature,
                              "extra": self.extra_body}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def complete(self, role: str, messages: list[dict], context: dict | None = None, attempt: int = 0,
                 temperature: float | None = None) -> tuple[str, dict, float, bool]:
        temperature = self.cfg.temperature if temperature is None else temperature
        key = self._cache_key(role, messages, temperature)
        cpath = self.cache_dir / f"{key}.json"
        if self.cfg.llm_cache and self.mock is None and cpath.exists():
            d = json.loads(cpath.read_text(encoding="utf-8"))
            return d["text"], d.get("usage", {}), 0.0, True
        t0 = time.time()
        if self.mock is not None:
            text, usage = self.mock.complete(role, messages, context or {}, attempt)
        else:
            kw: dict = dict(model=self.model, messages=messages, temperature=temperature)
            if self.cfg.llm_max_tokens:
                kw["max_tokens"] = self.cfg.llm_max_tokens
            if self.extra_body:
                kw["extra_body"] = self.extra_body
            resp = self._client.chat.completions.create(**kw)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0)} if u else {}
        latency = time.time() - t0
        if self.cfg.llm_cache and self.mock is None:
            cpath.write_text(json.dumps({"text": text, "usage": usage}, ensure_ascii=False), encoding="utf-8")
        return text, usage, latency, False

    # ------------------------------------------------------------ JSON-роли
    def call_json(self, role: str, system: str, user: str, schema: type[BaseModel], context: dict | None = None,
                  retry_template: str = "") -> BaseModel | None:
        """Один повтор с текстом ошибки. Возвращает валидированный объект или None."""
        self._n += 1
        call_id = f"{role}-{self._n:04d}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            try:
                text, usage, lat, cached = self.complete(role, messages, context, attempt)
            except Exception as e:  # сеть/эндпоинт: без backoff, просто фиксируем
                self.log.log(role, messages, "", {}, 0.0, False, attempt, False, f"{type(e).__name__}: {e}", call_id)
                return None
            try:
                obj = schema.model_validate(extract_json(text))
                self.log.log(role, messages, text, usage, lat, cached, attempt, True, None, call_id)
                return obj
            except (ValueError, ValidationError) as e:
                err = str(e)[:1500]
                self.log.log(role, messages, text, usage, lat, cached, attempt, False, err, call_id)
                fix = (retry_template or "Ответ не прошёл проверку: {error}\nВерни только исправленный JSON.").replace(
                    "{error}", err)
                messages = messages + [{"role": "assistant", "content": strip_think(text)[:6000]},
                                       {"role": "user", "content": fix}]
        return None

    def call_text(self, role: str, messages: list[dict], context: dict | None = None, attempt: int = 0,
                  call_id: str = "") -> str | None:
        if not call_id:
            self._n += 1
            call_id = f"{role}-{self._n:04d}"
        try:
            text, usage, lat, cached = self.complete(role, messages, context, attempt)
        except Exception as e:
            self.log.log(role, messages, "", {}, 0.0, False, attempt, False, f"{type(e).__name__}: {e}", call_id)
            return None
        self.log.log(role, messages, text, usage, lat, cached, attempt, None, None, call_id)
        return text

    def new_call_id(self, role: str) -> str:
        self._n += 1
        return f"{role}-{self._n:04d}"
