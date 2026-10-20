"""Локальный OpenAI-совместимый эндпоинт для smoke-теста живого пути (SDK, usage, <think>, кэш) без токенов.

    python bench/fake_llm_server.py 8765 &
    OPENAI_API_KEY=x OPENAI_MODEL=fake OPENAI_BASE_URL=http://127.0.0.1:8765/v1 NO_PROXY=127.0.0.1 \
        AGENT_ITERS=1 python solution.py путь/к/task.md

Роль определяется по системному промпту; ответы простые, но валидные и обёрнуты в <think>…</think>, как у Qwen3.
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def answer(messages: list[dict]) -> str:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    think = "<think>Рассуждаю о задаче...</think>\n"
    if "Исследователь" in system:
        nums = re.findall(r"^- (\w+) \| num \|", user, re.M)
        a, b = (nums + ["x", "y"])[:2]
        hyps = [
            {"id": "h1", "kind": "model", "rationale": "CatBoost", "patch": {"model.name": "catboost"},
             "feature_spec": None, "expected": "рост", "cost": "medium"},
            {"id": "h2", "kind": "feature_code", "rationale": f"отношение {a}/{b}",
             "patch": {"features.custom": ["+fe_ab"]}, "feature_spec": f"ab = {a} / ({b} + 1)", "expected": "рост",
             "cost": "low"},
            {"id": "h3", "kind": "config", "rationale": "флаги пропусков", "patch": {"preprocess.missing_flags": True},
             "feature_spec": None, "expected": "небольшой рост", "cost": "low"},
        ]
        return think + "```json\n" + json.dumps({"hypotheses": hyps}, ensure_ascii=False) + "\n```"
    if "Инженер" in system:
        m = re.search(r"Спецификация признака:\n(\w+) = (\w+) / \((\w+) \+ 1\)", user)
        name, a, b = m.groups() if m else ("f", None, None)
        if a is None:
            cols = re.findall(r"^- (\w+) \| числовая", user, re.M)
            a, b = cols[0], cols[1 if len(cols) > 1 else 0]
        code = (f"import numpy as np\nimport pandas as pd\n\ndef make_features(df):\n"
                f"    out = pd.DataFrame(index=df.index)\n    out['{name}'] = df['{a}'] / (df['{b}'].abs() + 1)\n"
                f"    return out\n")
        return think + "```python\n" + code + "```"
    if "Финальный выбор" in user:
        return think + json.dumps({"choice": "blend", "reason": "бленд стабильнее"}, ensure_ascii=False)
    return think + json.dumps({"verdict": "accept", "reason": "прирост объясним"}, ensure_ascii=False)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = answer(body["messages"])
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "fake"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": sum(len(m["content"]) for m in body["messages"]) // 3,
                          "completion_tokens": len(text) // 3, "total_tokens": 0}}
        data = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
