"""Инженер: пишет код одного слота харнесса по гипотезе Исследователя.

Слоты: transform (Preprocessing, Features), CustomModel, fit_predict (весь пайплайн целиком).
Ответ Инженера — код. Харнесс запускает его в песочнице на подвыборке поверх текущего решения
(feat_sandbox): OK — код уходит в эксперимент; ошибка — traceback возвращается Инженеру на
исправление, не больше cfg.engineer_attempts попыток. Внутри слота свобода полная; харнесс
проверяет только форму выхода.

Любая неудача — ошибка запуска, ответ без кода, ответ, обрезанный по длине, ошибка или таймаут вызова
LLM — это одна неудачная попытка внутри цикла; implement не бросает исключений. Если попытки кончились,
возвращается {"ok": False, "error": ...}: оркестратор пишет в журнал «не реализовано» с причиной,
и Исследователь решает, предлагать ли гипотезу снова.
"""

import logging
import re
import time

import pandas as pd

import feat_sandbox
import llm_client
import llm_prompts

log = logging.getLogger("agent")

CODE_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
SANDBOX_KIND = {"transform": "transform", "model": "model", "escape": "escape"}


SIGNATURE = {"transform": "def transform", "model": "class CustomModel", "escape": "def fit_predict"}


def extract_code(text: str, slot: str = "transform") -> str:
    """Код слота из ответа: первый блок с сигнатурой слота, иначе самый длинный блок."""
    blocks = [b.strip() for b in CODE_RE.findall(text or "") if b.strip()]
    if blocks:
        sig = SIGNATURE.get(slot, "")
        with_sig = [b for b in blocks if sig and sig in b]
        return with_sig[0] if with_sig else max(blocks, key=len)
    t = (text or "").strip()
    if t.startswith(("import ", "from ", "def ", "class ")):
        return t
    return ""


def target_doc(spec) -> str:
    if spec.task_type == "regression":
        return "float values"
    if spec.task_type in ("binary", "multiclass"):
        return f"class indices 0..{spec.n_classes - 1} in the order {list(spec.classes)[:30]}"
    if spec.task_type == "multilabel":
        return f"matrix (n, {len(spec.targets)}) of 0/1 for {spec.targets}"
    return f"matrix (n, {len(spec.targets)}) of floats for {spec.targets}"


def columns_text(df: pd.DataFrame, spec, limit: int = 150) -> str:
    """Колонки текущих данных решения для Инженера: имя [вид] пропуски; значения; описание."""
    lines = []
    for c in list(df.columns)[:limit]:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
            kind = "cat" if isinstance(s.dtype, pd.CategoricalDtype) else ("text" if c in spec.text_cols else "date"
                                                                         if c in spec.datetime_cols else "str")
            ex = ", ".join(str(v)[:20] for v in s.dropna().astype(str).value_counts().head(3).index)
            ex += f" (уник. {s.nunique()})"
        else:
            kind = "num"
            ex = f"медиана {s.median():.4g}, диапазон [{s.min():.4g}; {s.max():.4g}]" if s.notna().any() else "пусто"
        desc = spec.descriptions.get(c, "")
        lines.append(f"- {c} [{kind}] NaN={s.isna().mean():.0%}; {ex}" + (f"; {desc[:110]}" if desc else ""))
    if df.shape[1] > limit:
        lines.append(f"... ещё {df.shape[1] - limit} колонок")
    return "\n".join(lines)


class Engineer:
    def __init__(self, client, ctx, spec, task_text: str):
        self.client = client
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.spec = spec
        self.task_text = task_text

    def _contract(self, slot: str, n_rows: int) -> str:
        if slot == "transform":
            return llm_prompts.CONTRACT_TRANSFORM.format(target_doc=target_doc(self.spec), n_rows=n_rows)
        if slot == "model":
            return llm_prompts.CONTRACT_MODEL.format(task_type=self.spec.task_type, n_jobs=self.cfg.n_jobs,
                                                     fit_budget=60.0, n_rows=n_rows)
        return llm_prompts.CONTRACT_ESCAPE.format(n_jobs=self.cfg.n_jobs)

    def implement(self, slot: str, slot_id: str, hypothesis: dict, config: dict, columns: str, analyst: str,
                  deadline: float, n_rows: int) -> dict:
        system = llm_prompts.ENGINEER_SYSTEM.format(attempts=self.cfg.engineer_attempts,
                                                    contract=self._contract(slot, n_rows))
        hyp = (f"{hypothesis.get('rationale', '')}\n\nСпецификация: {hypothesis.get('spec', '')}"
               f"\nОжидаемый эффект: {hypothesis.get('expected', '')}")
        user = llm_prompts.ENGINEER_USER.format(task=self.task_text, hypothesis=hyp, analyst=analyst or "нет",
                                                columns=columns[:12000])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        thinking = None if self.cfg.engineer_thinking else False
        err = "нет попыток"
        attempt = 0
        for attempt in range(1, self.cfg.engineer_attempts + 1):
            if time.time() > deadline:
                return {"ok": False, "error": f"время на реализацию вышло; последняя ошибка: {err}",
                        "attempts": attempt - 1}
            try:
                r = self.client.chat_full(messages, role="engineer", max_tokens=self.cfg.engineer_max_tokens,
                                          thinking=thinking, timeout=max(5.0, deadline - time.time()))
            except llm_client.LLMBudgetExceeded as e:
                return {"ok": False, "error": f"LLM: {e}", "attempts": attempt}
            except llm_client.LLMError as e:
                err = f"вызов LLM не удался: {e}"
                log.info("Инженер, попытка %d: %s", attempt, str(e)[:200])
                if "нет времени" in str(e):
                    return {"ok": False, "error": err, "attempts": attempt}
                continue  # сообщения не меняются: следующая попытка — тот же запрос
            text = r["text"]
            code = extract_code(text, slot)
            if not code:
                if r["finish"] == "length":
                    err = f"ответ оборвался по лимиту длины ({r['out']} токенов) до блока кода"
                    note = llm_prompts.ENGINEER_CUT
                else:
                    err = "в ответе нет блока ```python``` с кодом"
                    note = "No ```python``` block found. Reply with the full code."
                messages += [{"role": "assistant", "content": text[-4000:] or "(no answer text)"},
                             {"role": "user", "content": note}]
                continue
            timeout = self.cfg.sandbox_timeout_s if not self.cfg.unlimited else max(5.0, deadline - time.time())
            res = feat_sandbox.run(SANDBOX_KIND[slot], code, self.ctx, slot_id, config, timeout=timeout)
            if res.get("ok"):
                self.ctx.p("codes", f"{slot_id}.py").write_text(code, encoding="utf-8")
                return {"ok": True, "code": code, "result": res, "attempts": attempt}
            err = _format_error(res)
            messages += [{"role": "assistant", "content": text[-6000:]},
                         {"role": "user", "content": llm_prompts.ENGINEER_FIX.format(error=err[-3500:])}]
            if len(messages) > 8:  # система, задача и 3 последних обмена
                messages = messages[:2] + messages[-6:]
        return {"ok": False, "error": err, "attempts": attempt}


def format_result(res: dict) -> str:
    if res.get("ok"):
        parts = ["OK"] + [f"{k}={res[k]}" for k in ("n_columns", "time_s", "score_on_sample", "shape") if k in res]
        out = ", ".join(parts)
        if res.get("describe"):
            out += "\n" + res["describe"][:2500]
        return out
    return _format_error(res)


def _format_error(res: dict) -> str:
    return f"ОШИБКА: {str(res.get('error', ''))[:1500]}\n{str(res.get('traceback', ''))[-2000:]}"
