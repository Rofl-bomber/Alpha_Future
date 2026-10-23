"""Подставной OpenAI-совместимый сервер для проверки харнесса без настоящей LLM (в архив не входит).

Запуск: python tests/fake_llm_server.py 8765
Затем: OPENAI_API_KEY=x OPENAI_MODEL=fake OPENAI_BASE_URL=http://127.0.0.1:8765/v1 uv run solution.py task.md

Отвечает по роли из системного промпта и числу ходов в диалоге: TaskSpec, Аналитик (шаги кода,
submit_split, отчёт), сжатие отчёта, повторный вызов Аналитика (дельта), Исследователь (python,
read_report, propose по подэтапу), Инженер (transform с ошибкой на первой попытке, CustomModel,
fit_predict), Критик.
"""

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_MODE", "")  # nosplit — Аналитик не вызывает submit_split; badjson — ломает часть B
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PY = "```python\n{}\n```"

ANALYST_STEPS = [
    "Thought: смотрю размеры и типы\n" + PY.format("print(train.shape, test.shape)\ntrain.dtypes.value_counts()")
    + "\n</think>\nThought: второй блок не должен исполниться\n" + PY.format("print('SECOND BLOCK RAN')"),
    "Thought: проверяю защиту данных\n" + PY.format("train['_key'] = 1\nprint(train.shape)"),
    "Thought: фиксирую разбиение\n" + PY.format(
        "import numpy as np\n"
        "if 'year' in train.columns:\n"
        "    ho = (train['year'] == train['year'].max()).to_numpy()\n"
        "    submit_split(ho, type='ordered', columns=['year'])\n"
        "else:\n"
        "    rng = np.random.default_rng(0)\n"
        "    ho = rng.random(len(train)) < 0.15\n"
        "    submit_split(ho, type='group', columns=[c for c in train.columns if c not in ('id', 'label')][:4])"),
]

REPORT = """Final:
Part A
1. Строка — объект в году; test — отдельный период (проверено: годы не пересекаются, 100% строк).
2. Повторы объекта по годам: 35.2% строк test имеют историю в train.
3. Валидация: по годам, holdout — последний год (17.4% строк).
4. Таргет скошен (skew 2.1); колонка wind в test только 1.0 или NaN.
5. Гипотезы: H1 средний таргет по типу объекта; H2 история объекта. Отклонено: R1 погода (corr 0.02).
6. Не проверено: стабильность ключа объекта.

Part B
```json
{"validation": {"type": "ordered", "columns": ["year"], "holdout": "последний год", "rejected_schemes": [{"scheme": "random", "reason": "утечка погоды"}]},
 "dependencies": [{"kind": "repeated_object", "rule": ["btype", "floor_area"], "alternatives": [["btype", "floor_area", "rating"]], "evidence": "35% строк test"}],
 "traps": [{"columns": ["wind"], "issue": "в test только 1.0/NaN", "evidence": "100% строк test"}],
 "hypotheses": [{"id": "H1", "what": "среднее таргета по типу", "evidence": "79% дисперсии", "preliminary_estimate": "RMSE -5"},
                {"id": "H2", "what": "история объекта по ключу", "evidence": "corr 0.9", "preliminary_estimate": null}],
 "rejected": [{"id": "R1", "what": "погода", "reason": "слабая связь", "evidence": "corr 0.02"}],
 "open_questions": ["стабильность ключа"]}
```"""

DELTA_STEPS = [
    "Thought: смотрю текущие данные\n" + PY.format("print(cur_train.shape, cur_test.shape, y_search[:3], oof[:2])"),
    'Final:\n```json\n{"retire": [{"id": "H1", "reason": "уже в решении", "evidence": "узел принят"}], '
    '"update": [{"id": "H2", "evidence": "покрытие 30%"}], '
    '"add": {"hypotheses": [{"id": "H9", "what": "взаимодействие типа и площади", "evidence": "остатки", '
    '"preliminary_estimate": null}]}, "notes": "после подэтапа данные стали чище"}\n```',
]

PRE_TRANSFORM = """import numpy as np
import pandas as pd

def transform(train_df, y_train, others):
    num = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])][:3]
    def f(d):
        d = d.copy()
        for c in num:
            d[c + "_isna"] = d[c].isna().astype(float)
        if "wind" in d.columns:
            d = d.drop(columns=["wind"])
        return d
    return f(train_df), [f(o) for o in others]"""

FEAT_TRANSFORM = """import numpy as np
import pandas as pd

def transform(train_df, y_train, others):
    cats = [c for c in train_df.columns if str(train_df[c].dtype) == "category"]
    if not cats:
        return train_df, others
    key = cats[0]
    y = np.asarray(y_train, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]
    k = train_df[key].astype(str).to_numpy()
    prior = float(y.mean())
    fold = np.arange(len(train_df)) % 5
    te = np.zeros(len(train_df))
    for f in range(5):
        m = fold == f
        st = pd.DataFrame({"k": k[~m], "y": y[~m]}).groupby("k")["y"].agg(["sum", "count"])
        mp = (st["sum"] + prior * 10) / (st["count"] + 10)
        te[m] = pd.Series(k[m]).map(mp).fillna(prior).to_numpy()
    st = pd.DataFrame({"k": k, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    mp = (st["sum"] + prior * 10) / (st["count"] + 10)
    tr = train_df.copy()
    tr[key + "_te"] = te
    outs = []
    for o in others:
        o = o.copy()
        o[key + "_te"] = o[key].astype(str).map(mp).fillna(prior).to_numpy()
        outs.append(o)
    return tr, outs"""

CUSTOM_MODEL = """import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class CustomModel:
    def __init__(self, task_type, n_classes, seed):
        self.t = task_type
        self.m = (RandomForestRegressor if task_type == "regression" else RandomForestClassifier)(
            n_estimators=60, random_state=seed, n_jobs=2, min_samples_leaf=3)
    def _x(self, X):
        X = X.copy()
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].cat.codes
        return X.fillna(-999)
    def fit(self, X, y, X_val=None, y_val=None):
        self.m.fit(self._x(X), y)
        self.classes_ = getattr(self.m, "classes_", None)
        return self
    def predict(self, X):
        return self.m.predict(self._x(X))
    def predict_proba(self, X):
        return self.m.predict_proba(self._x(X))"""

STATE = {"engineer_transform": 0}


def researcher_reply(stage: str, turns: int) -> str:
    if turns == 0:
        return "Thought: анализ ошибок\nAction: python\n" + PY.format(
            "import numpy as np\nprint(cur_train.shape, np.asarray(oof).shape, len(search_rows), len(holdout_rows))")
    if turns == 1:
        return "Thought: читаю полный отчёт\nAction: read_report"
    if stage == "preprocessing":
        hyps = [{"stage": "preprocessing", "kind": "code", "ref": "T1", "rationale": "убрать сломанную колонку wind",
                 "spec": "флаги пропусков трёх числовых колонок; удалить wind", "patch": {}, "expected": "чище данные"},
                {"stage": "preprocessing", "kind": "config", "rationale": "флаги пропусков",
                 "patch": {"preprocess.missing_flags": True}, "expected": "учёт пропусков"},
                {"stage": "features", "kind": "config", "rationale": "гипотеза не того подэтапа",
                 "patch": {"features.builtin": "+freq_enc"}, "expected": "должна быть отброшена"}]
    elif stage == "features":
        hyps = [{"stage": "features", "kind": "code", "ref": "H1", "rationale": "среднее таргета по первой категории",
                 "spec": "OOF target mean по первой категориальной колонке, сглаживание 10", "patch": {},
                 "expected": "прирост"},
                {"stage": "features", "kind": "config", "ref": "R1", "rationale": "частоты категорий",
                 "patch": {"features.builtin": "+freq_enc"}, "expected": "небольшой прирост"},
                {"stage": "features", "kind": "config", "rationale": "ключ не своего подэтапа",
                 "patch": {"model.name": "xgb"}, "expected": "должен быть отброшен валидатором"}]
    else:
        hyps = [{"stage": "model", "kind": "config", "rationale": "XGBoost", "patch": {"model.name": "xgb"},
                 "expected": "разнообразие"},
                {"stage": "model", "kind": "custom_model", "rationale": "случайный лес", "spec": "RandomForest",
                 "patch": {}, "expected": "разнообразие бленда"}]
    out = {"lessons": [f"урок подэтапа {stage}"], "obsolete": [{"item": "R1", "reason": "проверка"}],
           "hypotheses": hyps}
    return "Thought: предлагаю\nAction: propose\n```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"


def engineer_reply(system: str, turns: int) -> str:
    if "def transform" in system:
        STATE["engineer_transform"] += 1
        if turns == 0 and STATE["engineer_transform"] % 2 == 1:
            return "Thought: первая попытка с ошибкой\n" + PY.format(
                "def transform(train_df, y_train, others):\n    raise ValueError('намеренная ошибка')")
        return "Thought: готово\n" + PY.format(STATE.get("next_transform", FEAT_TRANSFORM))
    if "class CustomModel" in system:
        return "Thought: модель\n" + PY.format(CUSTOM_MODEL)
    return "Thought: пайплайн\n" + PY.format(
        "import numpy as np\n\ndef fit_predict(train, y, predict_on, spec, time_budget_s):\n"
        "    return [np.full(len(d), float(np.mean(y))) for d in predict_on]")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages", [])
        system = msgs[0]["content"] if msgs else ""
        last = msgs[-1]["content"] if msgs else ""
        turns = sum(1 for m in msgs if m["role"] == "assistant")
        if "You are the Task Spec reader" in system:
            text = json.dumps({"notes": ["проверка: подставной TaskSpec"], "out_of_library": False})
        elif "called again during the search" in system:
            text = DELTA_STEPS[min(turns, len(DELTA_STEPS) - 1)]
        elif "You are the Analyst" in system:
            if "could not be read" in last:
                text = "```json\n" + REPORT.split("```json\n")[1].split("```")[0] + "```"
            elif "Time for the analysis is over" in last or turns >= len(ANALYST_STEPS):
                text = REPORT.replace('"open_questions"', '"open_questions" ::') if "badjson" in MODE else REPORT
            elif "nosplit" in MODE and turns == 2:
                text = "Thought: без разбиения\n" + PY.format("print('no split')")
            else:
                text = ANALYST_STEPS[turns]
        elif "You compress a data analysis report" in system:
            text = last[:600]
        elif "You are the Researcher" in system:
            stage = "preprocessing" if "Stage PREPROCESSING" in system else (
                "features" if "Stage FEATURES" in system else "model")
            if "could not be read" in last:
                turns = 9
            text = researcher_reply(stage, turns)
        elif "You are the Engineer" in system:
            STATE["next_transform"] = PRE_TRANSFORM if "флаги пропусков трёх" in msgs[1]["content"] else FEAT_TRANSFORM
            text = engineer_reply(system, turns)
        elif "You are the Critic" in system:
            text = json.dumps({"exclude": [], "reason": "нет замечаний"} if "Finalists" in last
                              else {"verdict": "accept", "reason": "механизм правдоподобен"}, ensure_ascii=False)
        else:
            text = "{}"
        time.sleep(0.1)
        resp = {"id": "fake", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": sum(len(m.get("content", "")) for m in msgs) // 4,
                          "completion_tokens": len(text) // 4, "total_tokens": 0}}
        data = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
