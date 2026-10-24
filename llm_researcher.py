"""Исследователь: цикл с инструментами, на выходе — гипотезы текущего подэтапа.

Каждый ответ модели — одно действие:
- python — код в Python-сессии (exp_kernel) по полным данным: сырые train/test, текущие данные
  решения cur_train/cur_test, y_search, OOF текущего решения; здесь же анализ ошибок;
- read_report — полный отчёт Аналитика с обновлениями и реестром;
- propose — итоговый JSON {lessons, obsolete, hypotheses}.
Инструментальных действий не больше cfg.researcher_tool_steps, затем модель обязана ответить propose.
Харнесс принимает только гипотезы текущего подэтапа (поле stage), остальные отбрасываются.
"""

import json
import logging
import re
import time

from pydantic import BaseModel, Field, field_validator

import llm_client
import llm_prompts

log = logging.getLogger("agent")

ACTION_RE = re.compile(r"Action\s*:\s*\**\s*([a-z_]+)", re.I)
PY_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
JSON_RE = re.compile(r"```(?:json|JSON)\s*\n(.*?)```", re.S)
KINDS = ("code", "config", "custom_model", "fit_predict")


class Hypothesis(BaseModel):
    stage: str = ""
    kind: str = "config"
    ref: str | None = None
    block: str = ""
    rationale: str = ""
    spec: str = ""
    patch: dict = Field(default_factory=dict)
    expected: str = ""
    cost: str = "medium"

    @field_validator("stage", "kind", "cost", "block", mode="before")
    @classmethod
    def _low(cls, v):
        return str(v or "").strip().lower()

    @field_validator("ref", mode="before")
    @classmethod
    def _ref(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in ("", "null", "none", "-") else v

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
        return v if isinstance(v, dict) else {}

    @field_validator("rationale", "spec", "expected", mode="before")
    @classmethod
    def _text(cls, v):
        if v is None:
            return ""
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


class ResearcherOut(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    obsolete: list = Field(default_factory=list)

    @field_validator("lessons", mode="before")
    @classmethod
    def _lessons(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        return [str(x).strip() for x in items if str(x).strip()]

    @field_validator("obsolete", mode="before")
    @classmethod
    def _obs(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]


def parse_action(text: str) -> dict:
    m = ACTION_RE.search(text or "")
    action = m.group(1).lower() if m else ""
    after = text[m.end():] if m else (text or "")
    py = PY_RE.findall(after)
    js = JSON_RE.findall(after) or JSON_RE.findall(text or "")
    if not action:
        action = "propose" if js else ("python" if py else "")
    return {"action": action, "code": py[0].strip() if py else "", "json": js[-1] if js else after}


def run(client, kernel, cfg, stage: str, schema: str, context: dict, full_report, timeout=None,
        log_path=None, contract: str = "") -> dict:
    """Один вызов Исследователя. Возвращает {hypotheses, lessons, obsolete, stage}; LLMError — наружу."""
    system = llm_prompts.RESEARCHER_SYSTEM.format(stage_text=llm_prompts.STAGE_TEXT[stage],
                                                  max_out=cfg.analysis_max_out, tool_steps=cfg.researcher_tool_steps,
                                                  schema=schema, stage=stage, n=cfg.researcher_hypotheses,
                                                  contract=_indent(contract))
    user = llm_prompts.RESEARCHER_USER.format(stage=stage, **context)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t_end = time.time() + timeout if timeout else None
    tools = 0
    bad_json = 0
    for turn in range(cfg.researcher_tool_steps + 4):
        left = (t_end - time.time()) if t_end else None
        if left is not None and left < 10:
            break
        text = client.chat(messages, role="researcher", max_tokens=cfg.researcher_max_tokens,
                           timeout=left, strip=False)
        act = parse_action(text)
        _write(log_path, {"t": round(time.time(), 1), "stage": stage, "turn": turn, "action": act["action"],
                          "content": text[-6000:]})
        if act["action"] == "propose":
            try:
                obj = llm_client.extract_json("```json\n" + act["json"] + "\n```")
                out = ResearcherOut.model_validate(obj)
            except Exception as e:  # noqa: BLE001
                bad_json += 1
                if bad_json > 2:
                    raise llm_client.LLMError(f"Исследователь: JSON не разобран: {e}") from e
                messages += [{"role": "assistant", "content": text[-6000:]},
                             {"role": "user", "content": llm_prompts.RESEARCHER_BAD_JSON.format(error=str(e)[:400])}]
                continue
            hyps = []
            for h in out.hypotheses:
                d = h.model_dump()
                d["stage"] = d["stage"] or stage
                if d["kind"] not in KINDS:
                    d["kind"] = "code" if d.get("spec") and not d.get("patch") else "config"
                hyps.append(d)
            return {"hypotheses": hyps, "lessons": list(out.lessons), "obsolete": list(out.obsolete),
                    "stage": stage, "tools": tools}
        if act["action"] in ("python", "read_report") and tools >= cfg.researcher_tool_steps:
            obs = llm_prompts.RESEARCHER_LIMIT
        elif act["action"] == "python":
            if not act["code"]:
                obs = "No ```python``` block found after 'Action: python'."
            elif kernel is None:
                obs = "The Python session is not available in this run; propose based on the request."
            else:
                tools += 1
                obs = llm_prompts.RESEARCHER_TOOL_OUT.format(
                    output=kernel.exec(act["code"], timeout=cfg.analysis_step_timeout_s, label=f"researcher:{stage}"))
        elif act["action"] == "read_report":
            tools += 1
            obs = llm_prompts.RESEARCHER_TOOL_OUT.format(output=full_report())
        else:
            obs = llm_prompts.RESEARCHER_BAD
        if tools >= cfg.researcher_tool_steps and act["action"] in ("python", "read_report"):
            obs += "\n\n" + llm_prompts.RESEARCHER_LIMIT
        messages += [{"role": "assistant", "content": text[-6000:]}, {"role": "user", "content": obs}]
    raise llm_client.LLMError("Исследователь не прислал propose")


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line if line.strip() else line for line in (text or "").splitlines())


def _write(path, rec):
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
