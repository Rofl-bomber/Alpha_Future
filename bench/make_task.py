"""Стенд: собрать задачу с известными ответами из train исходной задачи (в архив не входит).

python bench/make_task.py <task.md> <out_dir> [--group COL] [--frac 0.3] [--seed 0]

Делит train на новый train/test (случайно или по группам колонки), копирует task.md и
data_dictionary, пишет sample_submission и answers.csv (ответы test для bench/score.py).
Пути внутри task.md сохраняются: файлы кладутся в те же относительные каталоги.
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_md")
    ap.add_argument("out_dir")
    ap.add_argument("--group", default=None)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.task_md).resolve()
    text = src.read_text(encoding="utf-8")
    rel = {}
    for raw in re.findall(r"`([\w\-./]+\.csv)`", text):
        name = Path(raw).name.lower()
        for role in ("train", "test", "sample", "dict"):
            if role in name and role not in rel and (src.parent / raw).exists():
                rel[role] = raw
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out / "task.md")
    train = pd.read_csv(src.parent / rel["train"], low_memory=False)
    sample = pd.read_csv(src.parent / rel["sample"])
    id_col = sample.columns[0]
    targets = [c for c in sample.columns if c != id_col]
    rng = np.random.default_rng(a.seed)
    if a.group:
        groups = train[a.group].astype(str).unique()
        rng.shuffle(groups)
        test_groups, tot = [], 0
        for g in groups:
            test_groups.append(g)
            tot += (train[a.group].astype(str) == g).sum()
            if tot >= a.frac * len(train):
                break
        mask = train[a.group].astype(str).isin(test_groups).to_numpy()
    else:
        mask = rng.random(len(train)) < a.frac
    new_train, new_test = train[~mask], train[mask]
    for role, df in (("train", new_train), ("test", new_test.drop(columns=targets))):
        p = out / rel[role]
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    sub = new_test[[id_col]].copy()
    for t in targets:
        sub[t] = sample[t].iloc[0]
    sub.to_csv(out / rel["sample"], index=False)
    if "dict" in rel:
        shutil.copy(src.parent / rel["dict"], out / rel["dict"])
    new_test[[id_col] + targets].to_csv(out / "answers.csv", index=False)
    print(f"train={len(new_train)}, test={len(new_test)} → {out}")


if __name__ == "__main__":
    main()
