"""LLM client for an OpenAI-compatible Qwen endpoint.

Features: thinking on/off via chat_template_kwargs (auto-detected), <think> stripping, retries with backoff,
call/token budgets, deadline-aware timeouts, JSON and code extraction with repair, JSONL call log.
"""
from __future__ import annotations

import ast
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from mla_utils import LOG, dumps, truncate


class LLMError(Exception):
    pass


class BudgetExceeded(LLMError):
    pass


class Truncated(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    thinking: bool


THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    reasoning = " ".join(m.group(0) for m in THINK_RX.finditer(text))
    out = THINK_RX.sub("", text)
    low = out.lower()
    if "</think>" in low:  # opening tag was in the prompt template
        i = low.rfind("</think>")
        reasoning += out[:i]
        out = out[i + len("</think>"):]
    elif "<think>" in low:  # unterminated thinking (truncated)
        i = low.find("<think>")
        reasoning += out[i:]
        out = out[:i]
    return out.strip(), reasoning


# ---------------------------------------------------------------------------------------------
# JSON / code extraction
# ---------------------------------------------------------------------------------------------
def _balanced_spans(text: str, open_ch: str, close_ch: str):
    spans = []
    depth, start, in_str, esc, quote = 0, None, False, False, ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"',):
            in_str, quote = True, ch
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])
                start = None
    return spans


def _repair(s: str) -> str:
    s = re.sub(r"//[^\n\"]*(?=\n)", "", s)  # line comments outside strings (best effort)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s


def _try_load(s: str):
    for cand in (s, _repair(s)):
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        py = _repair(s).replace("true", "True").replace("false", "False").replace("null", "None")
        return ast.literal_eval(py)
    except Exception:
        return None


def extract_json(text: str, expect: type = dict):
    if not text:
        raise ValueError("empty response")
    cands = []
    for m in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL):
        cands.append(m.group(1).strip())
    if expect is dict:
        cands += sorted(_balanced_spans(text, "{", "}"), key=len, reverse=True)
    else:
        cands += sorted(_balanced_spans(text, "[", "]"), key=len, reverse=True)
    cands.append(text.strip())
    for c in cands:
        obj = _try_load(c)
        if isinstance(obj, expect):
            return obj
    raise ValueError("no valid JSON object found")


def extract_code(text: str, func: str = "transform") -> str:
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", text or "", re.DOTALL)
    with_func = [b for b in blocks if re.search(rf"def\s+{func}\s*\(", b)]
    if with_func:
        return with_func[-1].strip()
    if blocks:
        return max(blocks, key=len).strip()
    if text and re.search(rf"def\s+{func}\s*\(", text):
        i = min([text.find(k) for k in ("import ", "def ") if text.find(k) >= 0] or [0])
        return text[i:].strip()
    raise ValueError("no python code found")


# ---------------------------------------------------------------------------------------------
class LLMClient:
    DEFAULT_THINKING = {"analyst": True, "researcher": True, "developer": True, "debugger": True, "metric": True,
                        "health": False, "repair": False}

    def __init__(self, cfg, deadline, log_path: Path | None = None):
        from openai import OpenAI
        self.cfg = cfg
        self.deadline = deadline
        self.client = OpenAI(api_key=cfg.api_key or "EMPTY", base_url=cfg.base_url or None, timeout=cfg.llm_timeout, max_retries=0)
        self.model = cfg.model
        self.sem = threading.Semaphore(cfg.llm_concurrency)
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.template_kwargs_ok: bool | None = None
        self.available: bool | None = None
        self.log_path = log_path
        self.per_purpose: dict = {}
        self.disabled_reason = ""

    # ------------------------------------------------------------------ budget
    def budget_left(self) -> bool:
        return (self.calls < self.cfg.llm_max_calls and
                self.prompt_tokens + self.completion_tokens < self.cfg.llm_max_total_tokens and
                self.available is not False)

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "failures": self.failures, "per_purpose": self.per_purpose, "available": self.available,
                "template_kwargs_ok": self.template_kwargs_ok, "model": self.model}

    def _log(self, rec: dict):
        if not self.log_path:
            return
        try:
            with self.lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(dumps(rec) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ health
    def discover_model(self):
        if self.model:
            return
        try:
            ms = self.client.models.list()
            ids = [m.id for m in ms.data]
            qwen = [i for i in ids if "qwen" in i.lower()]
            self.model = (qwen or ids)[0]
            LOG.info(f"LLM model discovered: {self.model}")
        except Exception as e:
            raise LLMError(f"model discovery failed: {e}")

    def health_check(self, attempts: int = 3) -> bool:
        for a in range(attempts):
            try:
                self.discover_model()
                try:
                    r = self.chat([{"role": "user", "content": "Reply with the single word: OK"}], purpose="health",
                                  thinking=False, max_tokens=512, timeout=min(180.0, self.cfg.llm_timeout), _health=True)
                    reply, el = r.text, r.elapsed
                except Truncated:
                    reply, el = "<truncated reasoning>", 0.0  # the endpoint works; the model just thinks a lot
                self.available = True
                LOG.info(f"LLM available ({self.model}); reply={truncate(reply, 40)!r} in {el:.1f}s; "
                         f"template_kwargs={'yes' if self.template_kwargs_ok else 'no'}")
                return True
            except Exception as e:
                LOG.warning(f"LLM health check {a + 1}/{attempts} failed: {type(e).__name__}: {truncate(str(e), 300)}")
                if a < attempts - 1 and self.deadline.remaining() > 300:
                    time.sleep(min(20 * (a + 1), 45))
        self.available = False
        self.disabled_reason = "health check failed"
        return False

    # ------------------------------------------------------------------ chat
    def _thinking_for(self, purpose: str, thinking):
        mode = self.cfg.llm_thinking
        if mode == "on":
            return True
        if mode == "off":
            return False
        if thinking is not None:
            return thinking
        return self.DEFAULT_THINKING.get(purpose, False)

    def chat(self, messages: list, purpose: str = "chat", thinking=None, max_tokens: int | None = None,
             temperature: float | None = None, top_p: float | None = None, timeout: float | None = None,
             _health: bool = False, **_ignored) -> LLMResponse:
        if not _health:
            if self.available is False:
                raise LLMError("LLM unavailable")
            if not self.budget_left():
                raise BudgetExceeded("LLM budget exhausted")
        think = self._thinking_for(purpose, thinking)
        role_cap = (self.cfg.extra.get("role_tokens") or {}).get(purpose)
        if role_cap:
            max_tokens = role_cap
        max_tokens = int(min(max_tokens or self.cfg.llm_max_tokens_per_call, self.cfg.llm_max_tokens_per_call))
        if temperature is None:
            temperature = 0.6 if think else 0.7
        if top_p is None:
            top_p = 0.95 if think else 0.8
        last_err = None
        for attempt in range(4):
            remaining = self.deadline.hard_remaining() - 20
            to = min(timeout or self.cfg.llm_timeout, remaining)
            if to < 15:
                raise LLMError("no time left for an LLM call")
            msgs = messages
            if not think and self.template_kwargs_ok is False and msgs and msgs[-1].get("role") == "user":
                # Qwen3 soft switch when the server ignores/rejects chat_template_kwargs
                msgs = list(msgs[:-1]) + [{"role": "user", "content": str(msgs[-1]["content"]) + "\n/no_think"}]
            kwargs = dict(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=to)
            if self.template_kwargs_ok is not False:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(think)}}
            t0 = time.time()
            try:
                with self.sem:
                    resp = self.client.chat.completions.create(**kwargs)
                el = time.time() - t0
                if self.template_kwargs_ok is None and "extra_body" in kwargs:
                    self.template_kwargs_ok = True
                ch = resp.choices[0]
                msg = ch.message
                raw = msg.content or ""
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning") or ""
                text, r2 = strip_think(raw)
                reasoning = (reasoning or "") + r2
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0) or sum(len(str(m.get("content", ""))) for m in messages) // 3
                ct = int(getattr(usage, "completion_tokens", 0) or 0) or (len(raw) + len(reasoning)) // 3
                with self.lock:
                    self.calls += 1
                    self.prompt_tokens += pt
                    self.completion_tokens += ct
                    self.consecutive_failures = 0
                    pp = self.per_purpose.setdefault(purpose, {"calls": 0, "seconds": 0.0, "completion_tokens": 0})
                    pp["calls"] += 1
                    pp["seconds"] += el
                    pp["completion_tokens"] += ct
                finish = ch.finish_reason or ""
                self._log({"t": time.time(), "purpose": purpose, "thinking": think, "elapsed": round(el, 1), "pt": pt, "ct": ct,
                           "finish": finish, "text": truncate(text, 6000), "reasoning_len": len(reasoning)})
                if not text.strip():
                    if finish == "length":
                        raise Truncated(f"output truncated during reasoning ({ct} tokens)")
                    raise LLMError("empty response")
                return LLMResponse(text, reasoning, finish, pt, ct, el, think)
            except Truncated:
                raise
            except Exception as e:
                name = type(e).__name__
                msg = str(e)
                status = getattr(e, "status_code", None)
                last_err = e
                if status == 400 and "extra_body" in kwargs and self.template_kwargs_ok is None:
                    LOG.info("LLM endpoint rejected chat_template_kwargs; retrying without them")
                    self.template_kwargs_ok = False
                    continue
                if status in (401, 403, 404):
                    with self.lock:
                        self.failures += 1
                    raise LLMError(f"{name}: {truncate(msg, 300)}")
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError") or status is None
                with self.lock:
                    self.failures += 1
                    self.consecutive_failures += 1
                self._log({"t": time.time(), "purpose": purpose, "error": f"{name}: {truncate(msg, 500)}", "attempt": attempt})
                if not retryable or attempt == 3 or _health:
                    break
                time.sleep(min(60.0, 4.0 * (2 ** attempt)))
        if self.consecutive_failures >= 8:
            LOG.warning("LLM: too many consecutive failures; disabling LLM for the rest of the run")
            self.available = False
            self.disabled_reason = "consecutive failures"
        raise LLMError(f"LLM call failed: {type(last_err).__name__}: {truncate(str(last_err), 300)}")

    def chat_json(self, messages: list, purpose: str, expect: type = dict, **kw):
        try:
            r = self.chat(messages, purpose=purpose, **kw)
        except Truncated:
            kw2 = dict(kw)
            kw2["thinking"] = False
            r = self.chat(messages, purpose=purpose, **kw2)
        try:
            return extract_json(r.text, expect), r
        except Exception as e:
            fix = messages + [{"role": "assistant", "content": truncate(r.text, 12000)},
                              {"role": "user", "content": f"Your reply could not be parsed as JSON ({e}). Reply again with ONLY the corrected JSON {'object' if expect is dict else 'array'}, no prose, no code fences."}]
            r2 = self.chat(fix, purpose="repair", thinking=False, max_tokens=kw.get("max_tokens"))
            return extract_json(r2.text, expect), r2

    def chat_code(self, messages: list, purpose: str, func: str = "transform", **kw):
        try:
            r = self.chat(messages, purpose=purpose, **kw)
        except Truncated:
            kw2 = dict(kw)
            kw2["thinking"] = False
            r = self.chat(messages, purpose=purpose, **kw2)
        return extract_code(r.text, func), r
