"""Простые роли LLM с ответом в JSON: сборка TaskSpec, Критик и Критик кода.

Аналитик — в llm_analyst.py, Исследователь — в llm_researcher.py, Инженер — в llm_engineer.py.
Критик скачков и финалистов по умолчанию — заглушка (cfg.critic_enabled=False): всегда «принять»
и «никого не исключать», LLM не вызывается. Критик кода (cfg.code_critic) сверяет просьбу
Исследователя, код Инженера и сводку того, что код выдал на настоящем фолде; права вето у него нет.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

import llm_prompts
import task_metrics

log = logging.getLogger("agent")


class SpecOut(BaseModel):
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


def spec_reader(client, task_md: str, rules: str, columns: str, sample: str, timeout=None) -> dict:
    """Технический шаг: уточнить TaskSpec (тип, таргет, метрика, файл сабмита) по task.md и сводке файлов."""
    system = llm_prompts.SPEC_SYSTEM.format(metrics=", ".join(task_metrics.REGISTRY))
    user = llm_prompts.SPEC_USER.format(task_md=task_md[:14000], rules=rules, columns=columns[:12000],
                                        sample=sample[:1500])
    out = client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           role="spec", schema=SpecOut, max_tokens=client.cfg.researcher_max_tokens, timeout=timeout)
    return out.model_dump()


class CodeCriticOut(BaseModel):
    ok: bool = True
    issues: list = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def _issues(cls, v):
        if v is None:
            return []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def code_critic(client, hypothesis: str, contract: str, code: str, summary: str, timeout=None) -> dict:
    """Сверка просьбы, кода и результата. Возвращает {"ok", "issues"}; LLMError — наружу."""
    user = llm_prompts.CODE_CRITIC_USER.format(hypothesis=hypothesis, contract=contract, code=code[:20000],
                                               summary=summary[:8000])
    out = client.chat_json([{"role": "system", "content": llm_prompts.CODE_CRITIC_SYSTEM},
                            {"role": "user", "content": user}],
                           role="critic", schema=CodeCriticOut, max_tokens=client.cfg.critic_max_tokens,
                           timeout=timeout)
    d = out.model_dump()
    d["issues"] = [x.strip() for x in d["issues"] if x and x.strip()]
    d["ok"] = bool(d["ok"]) and not d["issues"]
    return d


def critic_jump(client, evidence: dict, timeout=None) -> dict:
    if not client.cfg.critic_enabled:
        return {"verdict": "accept", "reason": "Критик выключен (заглушка)"}
    user = llm_prompts.CRITIC_JUMP.format(**evidence)
    out = client.chat_json([{"role": "system", "content": llm_prompts.CRITIC_SYSTEM}, {"role": "user", "content": user}],
                           role="critic", schema=CriticOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return out.model_dump()


def critic_final(client, task: str, finalists: str, timeout=None) -> dict:
    if not client.cfg.critic_enabled:
        return {"exclude": [], "reason": "Критик выключен (заглушка)"}
    user = llm_prompts.CRITIC_FINAL.format(task=task, finalists=finalists)
    out = client.chat_json([{"role": "system", "content": llm_prompts.CRITIC_SYSTEM}, {"role": "user", "content": user}],
                           role="critic", schema=FinalCriticOut, max_tokens=client.cfg.researcher_max_tokens,
                           timeout=timeout)
    return out.model_dump()
