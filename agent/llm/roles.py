"""Три LLM-роли: Исследователь, Инженер, Критик. Каждая — свой промпт и строгий формат ответа."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .client import LLMClient, extract_code, strip_think

PROMPTS = Path(__file__).parent / "prompts"


def _tpl(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def render(name: str, **kw) -> str:
    return Template(_tpl(name)).safe_substitute(**{k: str(v) for k, v in kw.items()})


# ------------------------------------------------------------------ схемы

class Hypothesis(BaseModel):
    id: str = ""
    kind: Literal["config", "feature_code", "model", "postprocess"]
    rationale: str = ""
    patch: dict[str, Any] = Field(default_factory=dict)
    feature_spec: str | None = None
    expected: str = ""
    cost: str = "low"

    @field_validator("id", "rationale", "expected", "cost", mode="before")
    @classmethod
    def _str(cls, v):
        return "" if v is None else str(v)

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "feature_code" and not (self.feature_spec or "").strip():
            raise ValueError("для kind=feature_code обязательно поле feature_spec")
        if self.kind != "feature_code" and not self.patch:
            raise ValueError(f"гипотеза {self.id or '?'}: пустой patch")
        return self


class ResearcherOut(BaseModel):
    """Невалидные отдельные гипотезы не роняют ответ целиком: они уходят в журнал как filtered.
    Повтор с текстом ошибки нужен, только если валидных гипотез нет вовсе."""
    hypotheses: list[dict] = Field(min_length=1, max_length=10)

    @model_validator(mode="before")
    @classmethod
    def _wrap(cls, data):
        if isinstance(data, list):
            return {"hypotheses": data}
        return data

    @model_validator(mode="after")
    def _any_valid(self):
        valid, invalid = self.split()
        if not valid:
            raise ValueError("нет ни одной валидной гипотезы: " + " | ".join(x["error"][:400] for x in invalid))
        return self

    def split(self) -> tuple[list[Hypothesis], list[dict]]:
        valid, invalid = [], []
        for raw in self.hypotheses:
            try:
                valid.append(Hypothesis.model_validate(raw))
            except ValidationError as e:
                invalid.append({"raw": raw, "error": " ".join(str(e).split())[:600]})
        return valid, invalid


class JumpVerdict(BaseModel):
    verdict: Literal["accept", "reject"]
    reason: str = ""


class FinalChoice(BaseModel):
    choice: str
    reason: str = ""


# ------------------------------------------------------------------ роли

class Researcher:
    role = "researcher"

    def __init__(self, client: LLMClient):
        self.client = client

    def propose(self, ctx: dict) -> tuple[list[Hypothesis], list[dict]] | None:
        """Возвращает (валидные гипотезы, невалидные с текстом ошибки) или None."""
        system = _tpl("researcher_system.md")
        user = render("researcher_user.md", **ctx["prompt"])
        out = self.client.call_json(self.role, system, user, ResearcherOut, context=ctx.get("mock", {}),
                                    retry_template=_tpl("json_retry.md"))
        return None if out is None else out.split()


class Engineer:
    role = "engineer"

    def __init__(self, client: LLMClient, max_cols: int):
        self.client = client
        self.max_cols = max_cols

    def start(self, feature_spec: str, rationale: str, columns: str, mock_ctx: dict) -> "EngineerTurn":
        call_id = self.client.new_call_id(self.role)
        messages = [{"role": "system", "content": render("engineer_system.md", max_cols=self.max_cols)},
                    {"role": "user", "content": render("engineer_user.md", feature_spec=feature_spec,
                                                       rationale=rationale or "-", columns=columns)}]
        return self._ask(messages, mock_ctx, 0, call_id)

    def fix(self, turn: "EngineerTurn", error: str, stage: str, mock_ctx: dict, attempt: int) -> "EngineerTurn":
        messages = turn.messages + [
            {"role": "assistant", "content": strip_think(turn.raw_answer)[:6000] or f"```python\n{turn.code}\n```"},
            {"role": "user", "content": render("engineer_fix.md", error=error[-2500:], stage=stage)}]
        return self._ask(messages, mock_ctx, attempt, turn.call_id)

    def _ask(self, messages, mock_ctx, attempt, call_id) -> "EngineerTurn":
        text = self.client.call_text(self.role, messages, context={**mock_ctx, "attempt": attempt},
                                     attempt=attempt, call_id=call_id)
        if text is None:
            return EngineerTurn(messages, None, "", call_id, "LLM не ответила")
        try:
            return EngineerTurn(messages, extract_code(text), text, call_id, None)
        except ValueError as e:
            return EngineerTurn(messages, None, text, call_id, str(e))


class EngineerTurn:
    def __init__(self, messages, code, raw_answer, call_id, error):
        self.messages, self.code, self.raw_answer, self.call_id, self.error = messages, code, raw_answer, call_id, error


class Critic:
    role = "critic"

    def __init__(self, client: LLMClient):
        self.client = client

    def judge_jump(self, prompt_kw: dict, mock_ctx: dict) -> JumpVerdict | None:
        return self.client.call_json(self.role, _tpl("critic_system.md"), render("critic_jump.md", **prompt_kw),
                                     JumpVerdict, context={**mock_ctx, "task": "jump"},
                                     retry_template=_tpl("json_retry.md"))

    def choose_final(self, prompt_kw: dict, mock_ctx: dict) -> FinalChoice | None:
        return self.client.call_json(self.role, _tpl("critic_system.md"), render("critic_final.md", **prompt_kw),
                                     FinalChoice, context={**mock_ctx, "task": "final"},
                                     retry_template=_tpl("json_retry.md"))


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None, default=str)
