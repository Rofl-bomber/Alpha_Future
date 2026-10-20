"""Шаг 1 проверки: mock-LLM без эндпоинта, все ветки.

    python bench/check_mock.py [--work DIR] [--iters-reg 8] [--iters-clf 3]

Генерирует уменьшенные синтетические задачи, запускает solution.py с AGENT_MOCK_LLM=1
(как на платформе — отдельным процессом с путём к task.md) и проверяет по журналам, что пройдены все ветки.
Код возврата 0 — все ветки пройдены.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import make_synth  # noqa: E402


def run_task(task_md: Path, iters: int) -> Path:
    env = {**os.environ, "AGENT_MOCK_LLM": "1", "AGENT_ITERS": str(iters)}
    t = task_md.parent
    before = set((t / "agent_runs").glob("*")) if (t / "agent_runs").exists() else set()
    proc = subprocess.run([sys.executable, str(ROOT / "solution.py"), str(task_md)], env=env, cwd=str(t),
                          capture_output=True, text=True)
    (t / "solution_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        raise SystemExit(f"solution.py упал на {task_md}")
    after = set((t / "agent_runs").glob("*")) - before
    return sorted(after)[-1]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def checks_for(run_dir: Path, task_dir: Path) -> dict[str, bool]:
    J, L = load(run_dir / "journal.jsonl"), load(run_dir / "llm_log.jsonl")
    final = json.loads((run_dir / "final.json").read_text())
    hyps = [e for e in J if e.get("hypothesis")]
    by_call: dict[str, list] = {}
    for r in L:
        by_call.setdefault(r["call_id"], []).append(r)

    def retry_ok(pred):
        return any(len(v) >= 2 and v[0]["valid"] is False and pred(v[0].get("error") or "") and v[1]["valid"]
                   for v in by_call.values() if v[0]["role"] in ("researcher", "critic"))

    eng = [e for e in J if e.get("engineer")]
    statuses = [e["status"] for e in hyps if e["status"] != "filtered"]
    two_fail = any(a in ("discard", "error", "engineer_failed", "critic_reject")
                   and b in ("discard", "error", "engineer_failed", "critic_reject")
                   for a, b in zip(statuses, statuses[1:]))
    executed_kinds = {e["kind"] for e in hyps if e["status"] in ("keep", "discard", "critic_reject")}
    verdicts = {e["critic"]["verdict"] for e in J if e.get("critic")}
    sub = pd.read_csv(task_dir / "submission.csv")
    sample = next(task_dir.rglob("sample_submission.csv"))
    sample_df = pd.read_csv(sample)
    test_df = pd.read_csv(next(task_dir.rglob("test.csv")))
    id_col = sample_df.columns[0]
    return {
        "гипотезы всех четырёх типов исполнены": executed_kinds >= {"config", "feature_code", "model", "postprocess"},
        "невалидный JSON -> повтор -> валидный": retry_ok(lambda e: "JSON" in e),
        "невалидная схема (pydantic) -> повтор -> валидный": retry_ok(lambda e: "validation" in e.lower()),
        "код Инженера упал и исправлен": any(not x["engineer"]["attempts"][0]["ok"] and x["engineer"]["attempts"][-1]["ok"]
                                             for x in eng),
        "Инженер не справился за 2 исправления": any(e["status"] == "engineer_failed" and len(e["engineer"]["attempts"]) == 3
                                                    for e in eng),
        "фильтр: использование таргета/id": any("использует" in (e.get("reason") or "") for e in hyps),
        "фильтр: повтор": any("повтор" in (e.get("reason") or "") for e in hyps),
        "фильтр: невалидная отдельная гипотеза": any("невалидная гипотеза" in (e.get("reason") or "") for e in hyps),
        "смена направления (счётчик + фильтр)": any(e.get("event") == "direction_switch" for e in J)
                                                 and any("смена направления" in (e.get("reason") or "") for e in hyps),
        "две неудачные попытки подряд": two_fail,
        "скачок -> Критик: accept": "accept" in verdicts,
        "скачок -> Критик: reject": "reject" in verdicts,
        "postprocess без переобучения (reuse OOF)": any(e.get("reused_from") for e in J),
        "бленд посчитан": final.get("blend_score") is not None,
        "финальный выбор Критика": any(r["role"] == "critic" and "Финальный выбор" in r["messages"][-1]["content"]
                                       for r in L if r.get("valid")),
        "submission.csv: колонки и порядок строк": list(sub.columns) == list(sample_df.columns)
                                                   and len(sub) == len(test_df)
                                                   and (sub[id_col].astype(str).values == test_df[id_col].astype(str).values).all(),
        "OOF и test всех экспериментов сохранены": all((run_dir / "preds" / f"{e['exp_id']}.npz").exists()
                                                      for e in J if e.get("exp_id") and e.get("score") is not None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "bench" / "mock_tasks"))
    ap.add_argument("--iters-reg", type=int, default=8)
    ap.add_argument("--iters-clf", type=int, default=3)
    args = ap.parse_args()
    work = Path(args.work)
    make_synth.make_reg(work, n_tr=3000, n_te=1000)
    make_synth.make_clf(work, n_tr=4000, n_te=1500)
    make_synth.make_bin(work, n_tr=1500, n_te=500)
    runs = {}
    for name, iters in (("synth_reg", args.iters_reg), ("synth_clf", args.iters_clf), ("synth_bin", 1)):
        print(f"== {name}: mock-прогон, итераций {iters}", flush=True)
        runs[name] = run_task(work / name / "task.md", iters)
    table: dict[str, dict[str, bool]] = {name: checks_for(rd, work / name) for name, rd in runs.items()}
    names = list(next(iter(table.values())).keys())
    print("\n| Ветка | " + " | ".join(table) + " | итог |")
    print("|---|" + "---|" * (len(table) + 1))
    all_ok = True
    for n in names:
        vals = [table[t][n] for t in table]
        ok = any(vals)
        all_ok &= ok
        print(f"| {n} | " + " | ".join("✓" if v else "·" for v in vals) + f" | {'OK' if ok else 'НЕТ'} |")
    print("\nПрогоны:", {k: str(v) for k, v in runs.items()})
    print("ИТОГ:", "все ветки пройдены" if all_ok else "есть непройденные ветки")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
