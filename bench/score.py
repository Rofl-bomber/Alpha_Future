"""Стенд: посчитать метрику сабмита по answers.csv и балл по формуле кейса (в архив не входит).

python bench/score.py <task_dir> [--baseline VALUE]

Метрика определяется так же, как в агенте (task_spec + task_metrics). Балл:
регрессия S = 100·max(0, 1 − RMSE/B), классификация S = 100·max(0, (F1 − B)/(1 − B)).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import task_metrics  # noqa: E402
import task_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--baseline", type=float, default=None)
    a = ap.parse_args()
    d = Path(a.task_dir)
    spec, train, test, _, _, _ = task_spec.build_rules_spec(d / "task.md")
    ans = pd.read_csv(d / "answers.csv")
    sub = pd.read_csv(spec.output_path)
    m = ans.merge(sub, on=spec.id_col, suffixes=("_true", "_pred"))
    t = spec.targets[0]
    metric = task_metrics.REGISTRY[spec.metric]
    if spec.task_type == "regression":
        value = metric.fn(m[f"{t}_true"].to_numpy(float), m[f"{t}_pred"].to_numpy(float))
    else:
        index = {str(c): i for i, c in enumerate(spec.classes)}
        yt = m[f"{t}_true"].astype(str).map(index).to_numpy()
        if metric.need == "proba":
            yp = m[f"{t}_pred"].to_numpy(float)
        else:
            yp = m[f"{t}_pred"].astype(str).map(index).to_numpy()
        value = metric.fn(yt, yp)
    print(f"{spec.metric} = {value:.6g} (строк {len(m)} из {len(ans)})")
    if a.baseline is not None:
        b = a.baseline
        if metric.greater_is_better:
            s = 100 * max(0.0, (value - b) / (1 - b))
        else:
            s = 100 * max(0.0, 1 - value / b)
        print(f"балл по формуле кейса относительно {b}: {s:.2f}")


if __name__ == "__main__":
    main()
