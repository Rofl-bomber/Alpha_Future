"""Инженер: пишет код одного слота харнесса по гипотезе Исследователя.

Слоты: transform (Preprocessing, Features), CustomModel, fit_predict (весь пайплайн целиком).
Ответ Инженера — код. Харнесс запускает его в песочнице на подвыборке поверх текущего решения
(feat_sandbox): OK — код уходит в эксперимент; ошибка — traceback возвращается Инженеру на
исправление, не больше cfg.engineer_attempts попыток. Внутри слота свобода полная; харнесс
проверяет только форму выхода.

После успешного запуска transform Критик кода (cfg.code_critic) сверяет просьбу Исследователя, код и сводку
того, что код выдал на настоящем фолде. Если есть замечания, Инженер получает их вместе со сводкой и одну
попытку исправить или объяснить, что код верен. Права вето у Критика нет: дальше код идёт в эксперимент,
замечания и итог проверки записываются в журнал (check).

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
import llm_roles

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


def sample_text(df: pd.DataFrame, n: int = 5, limit: int = 6000) -> str:
    """Несколько строк данных, транспонированно (колонка — строка таблицы), чтобы влезли все колонки."""
    if df is None or not len(df):
        return "нет данных"
    rows = df.sample(min(n, len(df)), random_state=0)
    with pd.option_context("display.max_rows", 500, "display.max_columns", 20, "display.width", 200,
                           "display.max_colwidth", 30):
        text = rows.T.to_string()
    return text if len(text) <= limit else text[:limit] + "\n... (сокращено)"


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
                  deadline: float, n_rows: int, context: dict | None = None) -> dict:
        """context: folds — как формируются train_df и others; chain — код принятых transform; sample — строки данных."""
        system = llm_prompts.ENGINEER_SYSTEM.format(attempts=self.cfg.engineer_attempts,
                                                    contract=self._contract(slot, n_rows))
        hyp = (f"{hypothesis.get('rationale', '')}\n\nСпецификация: {hypothesis.get('spec', '')}"
               f"\nОжидаемый эффект: {hypothesis.get('expected', '')}")
        context = context or {}
        user = llm_prompts.ENGINEER_USER.format(task=self.task_text, hypothesis=hyp, analyst=analyst or "нет",
                                                columns=columns[:12000], folds=context.get("folds") or "нет данных",
                                                chain=context.get("chain") or "нет",
                                                sample=context.get("sample") or "нет данных")
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
                code, res, check = self._review(slot, slot_id, hyp, config, code, res, messages, text, deadline,
                                                n_rows)
                self.ctx.p("codes", f"{slot_id}.py").write_text(code, encoding="utf-8")
                return {"ok": True, "code": code, "result": res, "attempts": attempt, "check": check}
            err = _format_error(res)
            messages += [{"role": "assistant", "content": text[-6000:]},
                         {"role": "user", "content": llm_prompts.ENGINEER_FIX.format(error=err[-3500:])}]
            if len(messages) > 8:  # система, задача и 3 последних обмена
                messages = messages[:2] + messages[-6:]
        return {"ok": False, "error": err, "attempts": attempt}

    def _review(self, slot, slot_id, hyp, config, code, res, messages, text, deadline, n_rows):
        """Критик кода и одна попытка исправления. Возвращает (код, результат песочницы, check)."""
        summary = res.get("summary") or ""
        check = {"summary": summary, "issues": [], "verdict": "не проверялся", "fixed": False}
        if slot != "transform" or not self.cfg.code_critic or not summary:
            return code, res, check
        left = deadline - time.time()
        if left < 30:
            check["verdict"] = "не проверялся: не хватило времени"
            return code, res, check
        try:
            v = llm_roles.code_critic(self.client, hyp, self._contract(slot, n_rows), code, summary, timeout=left)
        except llm_client.LLMError as e:
            check["verdict"] = f"Критик не ответил: {str(e)[:150]}"
            return code, res, check
        if v["ok"]:
            check["verdict"] = "замечаний нет"
            return code, res, check
        check["issues"] = v["issues"][:8]
        issues = "\n".join(f"- {x}" for x in check["issues"])
        log.info("Критик кода %s: %s", slot_id, " | ".join(check["issues"])[:300])
        msgs = messages + [{"role": "assistant", "content": text[-6000:]},
                           {"role": "user", "content": llm_prompts.ENGINEER_REVIEW.format(summary=summary[:6000],
                                                                                       issues=issues)}]
        try:
            r = self.client.chat_full(msgs, role="engineer", max_tokens=self.cfg.engineer_max_tokens,
                                      thinking=None if self.cfg.engineer_thinking else False,
                                      timeout=max(5.0, deadline - time.time()))
        except llm_client.LLMError as e:
            check["verdict"] = f"есть замечания; Инженер не ответил на них ({str(e)[:100]}) — оставлен исходный код"
            return code, res, check
        new = extract_code(r["text"], slot)
        if not new:
            why = "ответ оборвался" if r["finish"] == "length" else "в ответе нет кода"
            check["verdict"] = f"есть замечания; {why} — оставлен исходный код"
            return code, res, check
        if new.strip() == code.strip():
            note = " ".join(CODE_RE.sub("", r["text"]).split())[:300]
            check["verdict"] = "есть замечания; Инженер считает код верным" + (f": {note}" if note else "")
            return code, res, check
        timeout = self.cfg.sandbox_timeout_s if not self.cfg.unlimited else max(5.0, deadline - time.time())
        res2 = feat_sandbox.run(SANDBOX_KIND[slot], new, self.ctx, slot_id, config, timeout=timeout)
        if not res2.get("ok"):
            check["verdict"] = ("есть замечания; исправленный код не запустился — оставлен исходный: "
                                + str(res2.get("error"))[:150])
            return code, res, check
        check.update(fixed=True, verdict="есть замечания; Инженер исправил код",
                     summary=res2.get("summary") or summary)
        return new, res2, check


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
