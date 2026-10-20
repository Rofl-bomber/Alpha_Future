"""Задача-двойник с отложенными 30% train вместо test (шаг 4: прирост над L1 на отложенной выборке).

    python bench/make_holdout.py путь/к/task.md out_dir [--frac 0.3] [--seed 0] [--subsample 0]

Копирует task.md и data_dictionary.csv в ту же относительную структуру, train делится на
train (1-frac) и test (frac, без таргета); ответы — в out_dir/answers.csv.
Для мультикласса оценка оптимистична: случайный сплит не воспроизводит сдвиг классов в настоящем test.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.spec import build_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_md")
    ap.add_argument("out_dir")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=0, help="взять столько строк train до деления")
    a = ap.parse_args()
    spec = build_spec(a.task_md)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(spec.task_dir)  # noqa: E731
    train = pd.read_csv(spec.train_path, low_memory=False)
    if a.subsample and len(train) > a.subsample:
        train = train.sample(n=a.subsample, random_state=a.seed)
    strat = train[spec.target] if spec.is_clf else None
    if strat is not None:
        vc = strat.value_counts()
        strat = strat.where(~strat.isin(vc.index[vc < 2]), "__rare__")
    tr, te = train_test_split(train, test_size=a.frac, random_state=a.seed, stratify=strat)
    id_col = spec.id_col or "id"
    if id_col not in train.columns:
        tr, te = tr.copy(), te.copy()
        tr[id_col], te[id_col] = range(len(tr)), range(len(tr), len(tr) + len(te))
    for src, df in ((spec.train_path, tr), (spec.test_path, te.drop(columns=[spec.target]))):
        dst = out / rel(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
    if spec.dict_path:
        (out / rel(spec.dict_path)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(spec.dict_path, out / rel(spec.dict_path))
    sample_cols = spec.submission_columns or [id_col, spec.target]
    sample = pd.DataFrame({c: (te[id_col].values if c == id_col else (spec.classes[0] if spec.is_clf else 0.0))
                           for c in sample_cols})
    sp = out / (rel(spec.sample_sub_path) if spec.sample_sub_path else Path("sample_submission.csv"))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sp, index=False)
    te[[id_col, spec.target]].to_csv(out / "answers.csv", index=False)
    shutil.copy(spec.task_md, out / spec.task_md.name)
    print(f"Готово: {out / spec.task_md.name}; train {len(tr)}, отложено {len(te)}")


if __name__ == "__main__":
    main()
