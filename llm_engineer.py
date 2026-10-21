"""Инженер: короткий ReAct-цикл (до cfg.react_max_steps шагов) с инструментами харнесса.

Инструменты: inspect_columns (статистика колонок), run_on_sample (код слота на подвыборке
в песочнице), submit (код на полных данных; при ошибке — traceback обратно, не больше
cfg.engineer_submit_attempts неудачных попыток). Инженер пишет только в слоты:
make_features, CustomModel, fit_predict.

Формат ответа модели — текстовый: Thought / Action / Columns + блок ```python```.
Так код не нужно экранировать внутри JSON, и разбор надёжнее.
"""

import logging
import re
import time

import pandas as pd

import feat_sandbox
import llm_prompts

log = logging.getLogger("agent")

KIND_BY_SLOT = {"features": "features", "model": "model", "escape": "escape"}


def parse_action(text: str) -> dict:
    act = re.search(r"Action\s*:\s*([a-z_]+)", text, re.I)
    code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    cols = re.search(r"Columns\s*:\s*(.+)", text)
    action = act.group(1).lower() if act else ("submit" if code_blocks else "")
    return {"action": action, "code": code_blocks[-1].strip() if code_blocks else "",
            "columns": [c.strip(" `'\"") for c in cols.group(1).split(",")] if cols else []}


def inspect_columns(X: pd.DataFrame, spec, names: list) -> str:
    lines = []
    for c in names[:15]:
        if c not in X.columns:
            lines.append(f"{c}: нет такой колонки")
            continue
        s = X[c]
        miss = s.isna().mean()
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).round(4).tolist()
            lines.append(f"{c} [число] NaN={miss:.1%}, уник.={s.nunique()}, квантили 0/1/25/50/75/99/100%: {q}, "
                         f"доля нулей={float((s == 0).mean()):.1%}")
        else:
            vc = s.astype(str).value_counts()
            lines.append(f"{c} [{s.dtype}] NaN={miss:.1%}, уник.={s.nunique()}, частые: "
                         + ", ".join(f"{k[:25]}={v}" for k, v in vc.head(8).items()))
        desc = spec.descriptions.get(c, "")
        if desc:
            lines.append(f"   описание: {desc[:200]}")
    return "\n".join(lines) or "колонки не указаны"


class Engineer:
    def __init__(self, client, ctx, spec, X_all: pd.DataFrame, columns_text: str, task_text: str):
        self.client = client
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.spec = spec
        self.X = X_all
        self.columns_text = columns_text
        self.task_text = task_text

    def _contract(self, slot: str) -> str:
        n_rows = len(self.X)
        if slot == "features":
            return llm_prompts.CONTRACT_FEATURES.format(n_rows=n_rows, max_new=self.cfg.max_new_features)
        if slot == "model":
            return llm_prompts.CONTRACT_MODEL.format(task_type=self.spec.task_type, n_jobs=self.cfg.n_jobs,
                                                     fit_budget=60.0, n_rows=n_rows)
        return llm_prompts.CONTRACT_ESCAPE.format(n_jobs=self.cfg.n_jobs)

    def implement(self, slot: str, slot_id: str, hypothesis: dict, deadline: float) -> dict:
        system = llm_prompts.ENGINEER_SYSTEM.format(sample_rows=self.cfg.sample_rows,
                                                    attempts=self.cfg.engineer_submit_attempts,
                                                    max_steps=self.cfg.react_max_steps,
                                                    contract=self._contract(slot))
        hyp_text = (f"{hypothesis.get('rationale', '')}\n\nСпецификация: {hypothesis.get('feature_spec', '')}"
                    f"\nОжидаемый эффект: {hypothesis.get('expected', '')}")
        user = llm_prompts.ENGINEER_USER.format(task=self.task_text, hypothesis=hyp_text,
                                                columns=self.columns_text[:9000])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        fails, steps = 0, []
        thinking = None if self.cfg.engineer_thinking else False
        for step in range(self.cfg.react_max_steps):
            if time.time() > deadline:
                return {"ok": False, "error": "время на реализацию вышло", "steps": steps}
            text = self.client.chat(messages, role="engineer", max_tokens=self.cfg.engineer_max_tokens,
                                    thinking=thinking)
            act = parse_action(text)
            a = act["action"]
            if a == "inspect_columns":
                obs = inspect_columns(self.X, self.spec, act["columns"])
            elif a in ("run_on_sample", "submit") and not act["code"]:
                obs = "Нет блока ```python``` с кодом. Повтори действие с полным кодом."
            elif a == "run_on_sample":
                res = feat_sandbox.run(KIND_BY_SLOT[slot], act["code"], self.ctx, sample=True, slot_id=slot_id)
                obs = _format_result(res)
            elif a == "submit":
                timeout = self.cfg.sandbox_timeout_s * (1 if slot == "features" else 2)
                res = feat_sandbox.run(KIND_BY_SLOT[slot], act["code"], self.ctx, sample=(slot != "features"),
                                       slot_id=slot_id, timeout=timeout)
                steps.append({"step": step, "action": a, "ok": res.get("ok")})
                if res.get("ok"):
                    code_path = self.ctx.p("codes", f"{slot_id}.py")
                    code_path.write_text(act["code"], encoding="utf-8")
                    return {"ok": True, "code": act["code"], "result": res, "steps": steps}
                fails += 1
                if fails >= self.cfg.engineer_submit_attempts:
                    return {"ok": False, "error": res.get("error", ""), "code": act["code"], "steps": steps}
                obs = "Submit не прошёл:\n" + _format_result(res)
            else:
                obs = ("Не понял действие. Ответ должен содержать строку 'Action: inspect_columns | run_on_sample | "
                       "submit' и, для кода, блок ```python```.")
            steps.append({"step": step, "action": a or "?"})
            messages += [{"role": "assistant", "content": text[-6000:]}, {"role": "user", "content": obs[-4000:]}]
            if len(messages) > 12:  # держим контекст коротким: система, задача и 4 последних обмена
                messages = messages[:2] + messages[-8:]
        return {"ok": False, "error": "исчерпан лимит шагов", "steps": steps}


def _format_result(res: dict) -> str:
    if res.get("ok"):
        parts = ["OK"]
        for k in ("n_new", "time_s", "score_on_sample", "shape"):
            if k in res:
                parts.append(f"{k}={res[k]}")
        out = ", ".join(parts)
        if res.get("notes"):
            out += "\nЗамечания: " + "; ".join(res["notes"])
        if res.get("describe"):
            out += "\n" + res["describe"][:2500]
        if res.get("head"):
            out += "\nПервые строки:\n" + res["head"][:2500]
        return out
    return f"ОШИБКА: {res.get('error', '')[:1500]}\n{res.get('traceback', '')[-2000:]}"


def columns_text(spec, X: pd.DataFrame, limit: int = 150) -> str:
    lines = []
    for c in spec.feature_cols[:limit]:
        if c not in X.columns:
            continue
        s = X[c]
        kind = ("cat" if c in spec.cat_cols else "num" if c in spec.num_cols else
                "text" if c in spec.text_cols else "date")
        if kind == "num":
            ex = f"медиана {s.median():.4g}, диапазон [{s.min():.4g}; {s.max():.4g}]" if s.notna().any() else "пусто"
        else:
            ex = ", ".join(str(v)[:20] for v in s.dropna().astype(str).value_counts().head(3).index)
        desc = spec.descriptions.get(c, "")
        lines.append(f"- {c} [{kind}] NaN={s.isna().mean():.0%}; {ex}" + (f"; {desc[:110]}" if desc else ""))
    return "\n".join(lines)
