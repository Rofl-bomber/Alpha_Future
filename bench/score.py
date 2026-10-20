"""Оценка на отложенной выборке: submission.csv против answers.csv, плюс L1 и все эксперименты прогона.

    python bench/score.py путь/к/task.md [--run agent_runs/<время>]

Без --run берётся последний прогон в agent_runs/ рядом с task.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.metrics import improvement, score  # noqa: E402
from agent.spec import build_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_md")
    ap.add_argument("--run")
    a = ap.parse_args()
    spec = build_spec(a.task_md)
    ans = pd.read_csv(spec.task_dir / "answers.csv")
    test = pd.read_csv(spec.test_path, usecols=[spec.id_col] if spec.id_col else None)
    id_col = spec.id_col or ans.columns[0]
    y_df = test[[id_col]].merge(ans, on=id_col, how="left")
    if spec.is_clf:
        y = y_df[spec.target].map({c: i for i, c in enumerate(spec.classes)}).values
        known = ~pd.isna(y)
        y = y[known].astype(int)
    else:
        y = y_df[spec.target].values.astype(float)
        known = np.ones(len(y), bool)

    def sc(pred):
        return score(spec.metric, y, pred[known], spec.n_classes or None)

    sub = pd.read_csv(spec.output_path)
    if spec.is_clf:
        lab = sub[spec.target].map({c: i for i, c in enumerate(spec.classes)}).fillna(0).astype(int).values
        pred = np.eye(spec.n_classes)[lab]
    else:
        pred = sub[spec.target].values.astype(float)
    final_h = sc(pred)

    runs = sorted((spec.task_dir / "agent_runs").glob("*"))
    run = Path(a.run) if a.run else (runs[-1] if runs else None)
    rows = []
    if run is not None:
        J = [json.loads(l) for l in (run / "journal.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for e in J:
            p = run / "preds" / f"{e.get('exp_id')}.npz"
            if e.get("exp_id") and e.get("score") is not None and p.exists():
                rows.append({"exp_id": e["exp_id"], "status": e["status"], "model": e.get("model"),
                             "cv": e["score"], "holdout": sc(np.load(p)["test"])})
        bp = run / "preds" / "blend.npz"
        if bp.exists():
            fin = json.loads((run / "final.json").read_text())
            rows.append({"exp_id": "blend", "status": "-", "model": "-", "cv": fin.get("blend_score"),
                         "holdout": sc(np.load(bp)["test"])})
    df = pd.DataFrame(rows)
    if len(df):
        print(df.to_string(index=False))
    l1 = df[df.status == "baseline"].holdout.iloc[0] if len(df) and (df.status == "baseline").any() else None
    print(f"\nМетрика {spec.metric}; submission.csv на отложенных: {final_h:.6g}")
    if l1 is not None:
        print(f"L1 на отложенных: {l1:.6g}; прирост над L1: {improvement(spec.metric, final_h, l1):+.6g}")


if __name__ == "__main__":
    main()
