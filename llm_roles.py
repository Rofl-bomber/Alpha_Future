"""Роли LLM поверх одной модели: Аналитик, Исследователь, Критик (Инженер — в llm_engineer.py).

У каждой роли короткий контекст и строгий JSON на выходе, проверяемый pydantic-схемой.
Сырые данные в промпт не попадают — только сводки.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

import llm_prompts
import task_metrics

log = logging.getLogger("agent")


class AnalystOut(BaseModel):
    task_type: str | None = None
    target: Any = None
    id_col: str | None = None
    metric: str | None = None
    greater_is_better: bool | None = None
    positive_class: Any = None
    output_file: str | None = None
    group_hint: str | None = None
    time_hint: str | None = None
    out_of_library: bool = False
    out_of_library_reason: str | None = None
    custom_metric_code: str | None = None
    notes: list = Field(default_factory=list)

    @field_validator("out_of_library", mode="before")
    @classmethod
    def _ool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "да")
        return bool(v)

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [str(v)]


class Hypothesis(BaseModel):
    kind: str = "config"
    block: str = "model"
    rationale: str = ""
    patch: dict = Field(default_factory=dict)
    feature_spec: str = ""
    expected: str = ""
    cost: str = "medium"

    @field_validator("kind", "block", "cost", mode="before")
    @classmethod
    def _low(cls, v):
        return str(v or "").strip().lower()

    @field_validator("patch", mode="before")
    @classmethod
    def _patch(cls, v):
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v

    @field_validator("feature_spec", "rationale", "expected", mode="before")
    @classmethod
    def _text(cls, v):
        if v is None:
            return ""
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


class ResearcherOut(BaseModel):
    hypotheses: list[Hypothesis]
    lessons: list[str] = Field(default_factory=list)

    @field_validator("lessons", mode="before")
    @classmethod
    def _lessons(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        return [str(x).strip() for x in items if str(x).strip()]


class CriticOut(BaseModel):
    verdict: str = "accept"
    reason: str = ""

    @field_validator("reason", mode="before")
    @classmethod
    def _r(cls, v):
        return "" if v is None else str(v)

    @field_validator("verdict", mode="before")
    @classmethod
    def _v(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("accept", "reject", "recheck") else "accept"


class FinalCriticOut(BaseModel):
    exclude: list = Field(default_factory=list)
    reason: str = ""

    @field_validator("reason", mode="before")
    @classmethod
    def _r(cls, v):
        return "" if v is None else str(v)

    @field_validator("exclude", mode="before")
    @classmethod
    def _ex(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]


def analyst(client, task_md: str, rules: str, columns: str, sample: str, timeout=None) -> dict:
    system = llm_prompts.ANALYST_SYSTEM.format(metrics=", ".join(task_metrics.REGISTRY))
    user = llm_prompts.ANALYST_USER.format(task_md=task_md[:14000], rules=rules, columns=columns[:12000],
                                           sample=sample[:1500])
    out = client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           role="analyst", schema=AnalystOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return out.model_dump()


def researcher(client, schema: str, n: int, context: dict, timeout=None) -> dict:
    """Гипотезы и обновлённые «уроки» Исследователя: {"hypotheses": [...], "lessons": [...]}."""
    system = llm_prompts.RESEARCHER_SYSTEM.format(schema=schema, n=n)
    user = llm_prompts.RESEARCHER_USER.format(**context)
    out = client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           role="researcher", schema=ResearcherOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return {"hypotheses": [h.model_dump() for h in out.hypotheses], "lessons": list(out.lessons)}


def critic_jump(client, evidence: dict, timeout=None) -> dict:
    user = llm_prompts.CRITIC_JUMP.format(**evidence)
    out = client.chat_json([{"role": "system", "content": llm_prompts.CRITIC_SYSTEM}, {"role": "user", "content": user}],
                           role="critic", schema=CriticOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return out.model_dump()


def critic_final(client, task: str, finalists: str, timeout=None) -> dict:
    user = llm_prompts.CRITIC_FINAL.format(task=task, finalists=finalists)
    out = client.chat_json([{"role": "system", "content": llm_prompts.CRITIC_SYSTEM}, {"role": "user", "content": user}],
                           role="critic", schema=FinalCriticOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return out.model_dump()
