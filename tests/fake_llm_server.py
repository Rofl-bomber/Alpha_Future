"""Подставной OpenAI-совместимый сервер для проверки харнесса без настоящей LLM (только для стенда).

Запуск: python tests/fake_llm_server.py 8765
Затем: OPENAI_API_KEY=x OPENAI_MODEL=fake OPENAI_BASE_URL=http://127.0.0.1:8765/v1 uv run solution.py task.md

Отвечает по роли из системного промпта: Аналитик, Исследователь (конфиг + признаки по коду),
Инженер (run_on_sample, затем submit простой функции признаков), Критик.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"researcher": 0, "engineer": 0}

FEATURE_CODE = '''```python
import numpy as np
import pandas as pd

def make_features(df):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = pd.DataFrame(index=df.index)
    if len(num) >= 2:
        a, b = df[num[0]], df[num[1]]
        out["ratio_ab"] = a / (b.abs() + 1.0)
        out["diff_ab"] = a - b
    return out
```'''

RESEARCH_SETS = [
    {"hypotheses": [
        {"kind": "config", "block": "features", "rationale": "частоты категорий", "patch": {"features.builtin": "+freq_enc"},
         "expected": "небольшой прирост", "cost": "low"},
        {"kind": "feature_code", "block": "features", "rationale": "отношение и разность двух первых числовых признаков",
         "feature_spec": "ratio_ab = a/(|b|+1), diff_ab = a-b для первых двух числовых колонок", "patch": {},
         "expected": "прирост на нелинейностях", "cost": "medium"},
        {"kind": "config", "block": "model", "rationale": "больше листьев", "patch": {"model.params": {"num_leaves": 31}},
         "expected": "меньше переобучения", "cost": "low"},
    ]},
    {"hypotheses": [
        {"kind": "config", "block": "preprocess", "rationale": "флаги пропусков", "patch": {"preprocess.missing_flags": True},
         "expected": "учёт пропусков", "cost": "low"},
        {"kind": "restart", "block": "model", "rationale": "XGBoost как новая ветка", "patch": {"model.name": "xgb"},
         "expected": "разнообразие", "cost": "medium"},
        {"kind": "config", "block": "model", "rationale": "неверный ключ для проверки валидатора",
         "patch": {"model.params.unknown_param": 3}, "expected": "", "cost": "low"},
    ]},
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages", [])
        system = msgs[0]["content"] if msgs else ""
        if "You are the Analyst" in system:
            text = json.dumps({"notes": ["проверка: Аналитик подставной"], "out_of_library": False})
        elif "You are the Researcher" in system:
            text = json.dumps(RESEARCH_SETS[STATE["researcher"] % len(RESEARCH_SETS)], ensure_ascii=False)
            STATE["researcher"] += 1
        elif "You are the Engineer" in system:
            turns = sum(1 for m in msgs if m["role"] == "assistant")
            action = "run_on_sample" if turns == 0 else "submit"
            text = f"Thought: проверяю признаки\nAction: {action}\n{FEATURE_CODE}"
        elif "You are the Critic" in system:
            last = msgs[-1]["content"]
            text = json.dumps({"exclude": [], "reason": "нет замечаний"} if "Finalists" in last
                              else {"verdict": "accept", "reason": "механизм правдоподобен"}, ensure_ascii=False)
        else:
            text = "{}"
        time.sleep(0.2)
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
