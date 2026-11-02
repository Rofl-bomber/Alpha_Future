"""Deterministic stand-in for the LLM, used for offline testing of every agent code path
(including malformed JSON, broken code and the debugger loop)."""
from __future__ import annotations

import json
import re
import time

from mla_llm import LLMResponse, extract_code, extract_json


class MockLLM:
    def __init__(self, cfg, deadline=None, latency: float = 1.0):
        self.cfg = cfg
        self.available = True
        self.calls = 0
        self.latency = latency
        self.research_round = 0
        self.model = "mock"

    def budget_left(self) -> bool:
        return self.calls < self.cfg.llm_max_calls

    def health_check(self, attempts: int = 1) -> bool:
        return True

    def usage(self) -> dict:
        return {"calls": self.calls, "model": "mock"}

    # ------------------------------------------------------------------
    def chat(self, messages, purpose="chat", mock_ctx=None, **kw) -> LLMResponse:
        time.sleep(self.latency)
        self.calls += 1
        user = messages[-1]["content"]
        if purpose == "analyst":
            text = self._analyst(user)
        elif purpose == "researcher":
            text = self._researcher(mock_ctx or {})
        elif purpose == "developer":
            text = self._developer(mock_ctx or {})
        elif purpose == "debugger":
            text = self._debugger(mock_ctx or {})
        elif purpose == "repair":
            text = '{"experiments": []}'
        else:
            text = "OK"
        return LLMResponse(text, "", "stop", 100, 100, self.latency, False)

    def chat_json(self, messages, purpose, expect=dict, **kw):
        r = self.chat(messages, purpose=purpose, **kw)
        return extract_json(r.text, expect), r

    def chat_code(self, messages, purpose, func="transform", **kw):
        r = self.chat(messages, purpose=purpose, **kw)
        return extract_code(r.text, func), r

    # ------------------------------------------------------------------
    def _analyst(self, user: str) -> str:
        m = re.search(r"## Preliminary automatic analysis.*?\n(\{.*?\n\})", user, re.DOTALL)
        h = json.loads(m.group(1)) if m else {}
        out = {
            "target_columns": h.get("target_columns"),
            "id_column": h.get("id_column"),
            "task_type": h.get("task_type"),
            "metric": h.get("metric"),
            "metric_reasoning": "mock",
            "positive_label": h.get("positive_label"),
            "prediction_format": "label",
            "cv_strategy": {"type": "random", "group_column": None, "time_column": None, "reasoning": "mock"},
            "categorical_columns": [],
            "drop_columns": [],
            "domain_summary": "Mock analysis: the target depends on the most informative columns.",
            "feature_ideas": ["ratio of the two most important numeric columns", "group mean of a numeric column by the top categorical"],
            "potential_leaks": "none",
        }
        # wrap in prose + code fence to exercise the JSON extractor
        return "Here is the analysis:\n```json\n" + json.dumps(out) + "\n```\nDone."

    def _researcher(self, ctx: dict) -> str:
        self.research_round += 1
        cols = ctx.get("columns", [])
        cats = ctx.get("cats", [])
        imp = [c for c in ctx.get("importance", []) if c in cols]
        nums = [c for c in imp if c not in cats] or [c for c in cols if c not in cats]
        top_cats = [c for c in imp if c in cats] or cats
        r = self.research_round
        exps = []
        if len(nums) >= 2:
            a, b = nums[(r - 1) % len(nums)], nums[r % len(nums)]
            exps.append({"type": "feature_code", "title": f"ratio {a} over {b} (round {r})", "rationale": "mock ratio",
                         "spec": f"MOCK:ratio:{a}:{b}", "expected_gain": "medium"})
        if top_cats and nums:
            exps.append({"type": "feature_code", "title": f"deviation of {nums[0]} from its mean by {top_cats[0]} (round {r})",
                         "rationale": "mock group deviation", "spec": f"MOCK:groupdev:{top_cats[0]}:{nums[0]}", "expected_gain": "high"})
        exps.append({"type": "feature_code", "title": f"buggy idea round {r}", "rationale": "tests the debugger",
                     "spec": f"MOCK:buggy:{nums[0] if nums else cols[0]}", "expected_gain": "low"})
        if len(top_cats) >= 2:
            exps.append({"type": "target_encoding", "title": f"TE of {top_cats[0]} x {top_cats[1]} (round {r})", "rationale": "mock",
                         "columns": [[top_cats[0], top_cats[1]]], "expected_gain": "medium"})
        elif top_cats:
            exps.append({"type": "target_encoding", "title": f"TE of {top_cats[0]} (round {r})", "rationale": "mock",
                         "columns": [[top_cats[0]]], "expected_gain": "medium"})
        body = {"analysis": f"mock analysis round {r}", "insights": [f"mock insight {r}"], "experiments": exps}
        txt = json.dumps(body)
        if r == 1:
            txt = txt[:-1] + ",}"  # trailing comma -> exercises JSON repair
        return "<think>pretend reasoning</think>\n" + txt

    def _developer(self, ctx: dict) -> str:
        spec = (ctx.get("exp") or {}).get("spec", "")
        parts = spec.split(":")
        if spec.startswith("MOCK:ratio"):
            a, b = parts[2], parts[3]
            body = f"    df['{a}_per_{b}'] = df['{a}'] / df['{b}'].replace(0, np.nan)\n"
        elif spec.startswith("MOCK:groupdev"):
            g, v = parts[2], parts[3]
            body = (f"    m = df.groupby('{g}', dropna=False)['{v}'].transform('mean')\n"
                    f"    df['{v}_dev_{g}'] = df['{v}'] - m\n")
        elif spec.startswith("MOCK:buggy"):
            v = parts[2]
            body = f"    df['{v}_x2'] = df['{v}_does_not_exist'] * 2\n"
        else:
            body = "    df['mock_const_feature'] = 1.0\n"
        return "```python\nimport numpy as np\nimport pandas as pd\n\ndef transform(df):\n    df = df.copy()\n" + body + "    return df\n```"

    def _debugger(self, ctx: dict) -> str:
        spec = (ctx.get("exp") or {}).get("spec", "")
        if spec.startswith("MOCK:buggy"):
            v = spec.split(":")[2]
            return ("```python\nimport numpy as np\nimport pandas as pd\n\ndef transform(df):\n    df = df.copy()\n"
                    f"    df['{v}_sq'] = df['{v}'].astype(float) ** 2\n    return df\n```")
        return self._developer(ctx)
