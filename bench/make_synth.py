"""Синтетические задачи в формате платформы для проверки харнесса (mock и живые прогоны).

    python bench/make_synth.py [out_dir]

Создаёт synth_reg (регрессия, RMSE), synth_clf (мультикласс, macro F1) и synth_bin (бинарная, метрика
не названа явно). В обеих основных задачах таргет сильно зависит от отношения двух признаков a/(b+1),
которое деревьям трудно выучить из сырых колонок, — так feature_code даёт скачок и зовёт Критика.
Ответы test лежат в answers.csv в корне задачи (не в participants/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK_MD_REG = """# Определение годового удельного энергопотребления зданий (синтетика)

## Описание задачи
Нужно предсказать удельное энергопотребление здания `site_eui` по его характеристикам и погоде.

## Данные
- `regression/participants/train.csv` — обучающая выборка ({n_tr} строк)
- `regression/participants/test.csv` — тестовая выборка ({n_te} строк, без таргета)
- `regression/participants/data_dictionary.csv` — описание столбцов
- `regression/participants/sample_submission.csv` — пример сабмита

## Метрика
RMSE (root mean squared error), чем меньше, тем лучше.

## Формат ответа
CSV-файл `submission.csv` с колонками `id`, `site_eui`, по одной строке на каждую строку test.csv.
"""

TASK_MD_CLF = """# Определение типа кибер-атаки (синтетика)

## Описание задачи
По характеристикам сетевого соединения нужно определить категорию атаки `attack_cat`.

## Данные
- `classification/participants/train.csv` — обучающая выборка ({n_tr} строк)
- `classification/participants/test.csv` — тестовая выборка ({n_te} строк, без таргета)
- `classification/participants/data_dictionary.csv` — описание столбцов
- `classification/participants/sample_submission.csv` — пример сабмита

## Метрика
Основная метрика — Macro F1:

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average="macro")
```

Accuracy и Weighted F1 — справочные.

## Формат ответа
CSV-файл `submission.csv` с колонками `id`, `attack_cat`.
"""

TASK_MD_BIN = """# Отток клиентов (синтетика)

Предскажите, уйдёт ли клиент (`churn`: yes/no). Файлы: train.csv, test.csv, data_dictionary.csv,
sample_submission.csv в папке data. Сабмит — submission.csv с колонками client_id, churn.
"""


def _dict_rows(cols: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(cols, columns=["feature", "role", "dtype", "unit", "description_ru"])


def _write(task_dir: Path, sub: str, md: str, train, test, dd, sample, answers):
    p = task_dir / sub
    p.mkdir(parents=True, exist_ok=True)
    train.to_csv(p / "train.csv", index=False)
    test.to_csv(p / "test.csv", index=False)
    dd.to_csv(p / "data_dictionary.csv", index=False)
    sample.to_csv(p / "sample_submission.csv", index=False)
    answers.to_csv(task_dir / "answers.csv", index=False)
    (task_dir / "task.md").write_text(md, encoding="utf-8")


def make_reg(out: Path, n_tr=6000, n_te=2500, seed=0):
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    a = rng.uniform(0, 10, n)                       # electricity_kwh_per_day
    b = rng.uniform(0, 5, n)
    b[rng.random(n) < 0.03] = 0.0                    # нули в знаменателе
    facility = rng.choice([f"type_{i}" for i in range(20)], n)
    fac_eff = {f"type_{i}": e for i, e in enumerate(rng.normal(0, 15, 20))}
    state = rng.choice([f"State_{i}" for i in range(6)], n, p=[.3, .25, .2, .1, .1, .05])
    heating = rng.gamma(3, 800, n)
    year = rng.integers(1900, 2016, n).astype(float)
    y = (60 * a / (b + 1) + np.vectorize(fac_eff.get)(facility) + 0.004 * heating
         + 0.05 * (year - 1950) + rng.normal(0, 2, n))
    df = pd.DataFrame({
        "electricity_load": a, "occupant_density": b, "facility_type": facility, "state_factor": state,
        "heating_degree_days": heating, "year_built": year,
        "floor_area": rng.lognormal(10, 1, n), "elevation": rng.normal(200, 150, n),
        "energy_star_rating": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(1, 100, n)),
        "cooling_degree_days": rng.gamma(2, 500, n),
        "wind_speed": np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0, 20, n)),
        "fog_days": rng.poisson(5, n).astype(float),
    })
    df["id"] = np.arange(n) + 100000
    df["site_eui"] = y
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    feats = [c for c in df.columns if c not in ("id", "site_eui")]
    dd = _dict_rows(
        [(c, "feature", "category" if df[c].dtype == object else "numeric", "", f"признак {c}") for c in feats]
        + [("site_eui", "target", "numeric", "kBtu/ft2", "удельное энергопотребление"), ("id", "id", "numeric", "", "идентификатор")])
    sample = pd.DataFrame({"id": te["id"].values, "site_eui": 0.0})
    _write(out / "synth_reg", "regression/participants", TASK_MD_REG.format(n_tr=n_tr, n_te=n_te),
           tr, te.drop(columns=["site_eui"]), dd, sample, te[["id", "site_eui"]])


def make_clf(out: Path, n_tr=8000, n_te=3000, seed=1):
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    classes = ["Normal", "DoS", "Exploits", "Fuzzers", "Reconnaissance", "Backdoor"]
    probs = [.45, .2, .15, .1, .07, .03]
    yi = rng.choice(len(classes), n, p=probs)
    bpp_mean = np.array([600, 90, 350, 1200, 60, 220])      # байт на пакет по классам
    spkts = rng.lognormal(3, 1.0, n).round() + 1
    bpp = bpp_mean[yi] * rng.lognormal(0, 0.25, n)
    sbytes = spkts * bpp
    dpkts = (spkts * rng.uniform(0.3, 1.2, n)).round()
    dbytes = dpkts * rng.lognormal(6, 0.8, n)
    proto_p = {0: [.7, .25, .05], 1: [.4, .4, .2], 2: [.8, .15, .05], 3: [.5, .45, .05], 4: [.3, .3, .4], 5: [.6, .3, .1]}
    proto = np.array([rng.choice(["tcp", "udp", "arp"], p=proto_p[k]) for k in yi])
    service = rng.choice(["-", "http", "dns", "ftp", "smtp", "ssh"], n)
    dur = rng.exponential(1.0, n) * (1 + (yi == 1) * 0.5)
    df = pd.DataFrame({
        "dur": dur, "proto": proto, "service": service, "state": rng.choice(["FIN", "INT", "CON", "REQ"], n),
        "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
        "rate": spkts / (dur + 0.01), "sttl": rng.choice([31, 62, 254], n).astype(float),
        "ct_srv_src": rng.poisson(4, n).astype(float),
        "is_ftp_login": (rng.random(n) < 0.02).astype(int),
        "smean": np.where(rng.random(n) < 0.1, np.nan, rng.normal(100, 30, n)),
    })
    df["id"] = np.arange(n) + 1
    df["attack_cat"] = np.array(classes)[yi]
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    feats = [c for c in df.columns if c not in ("id", "attack_cat")]
    dt = {c: ("category" if df[c].dtype == object else ("binary" if c == "is_ftp_login" else "numeric")) for c in feats}
    dd = _dict_rows([("id", "id", "numeric", "", "идентификатор")]
                    + [(c, "feature", dt[c], "", f"признак соединения {c}") for c in feats]
                    + [("attack_cat", "target", "category", "", "категория атаки")])
    sample = pd.DataFrame({"id": te["id"].values, "attack_cat": "Normal"})
    _write(out / "synth_clf", "classification/participants", TASK_MD_CLF.format(n_tr=n_tr, n_te=n_te),
           tr, te.drop(columns=["attack_cat"]), dd, sample, te[["id", "attack_cat"]])


def make_bin(out: Path, n_tr=3000, n_te=1000, seed=2):
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    x1, x2 = rng.normal(0, 1, n), rng.uniform(1, 5, n)
    plan = rng.choice(["basic", "pro", "max"], n)
    logit = 1.5 * x1 / x2 + (plan == "basic") * 0.8 - 1.0 + rng.normal(0, 0.5, n)
    y = np.where(logit > 0, "yes", "no")
    df = pd.DataFrame({"client_id": [f"c{i:05d}" for i in range(n)], "tenure": x2, "usage_delta": x1,
                       "plan": plan, "calls": rng.poisson(3, n), "churn": y})
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    dd = _dict_rows([("client_id", "id", "category", "", "клиент"), ("tenure", "feature", "numeric", "", "стаж"),
                     ("usage_delta", "feature", "numeric", "", "изменение использования"),
                     ("plan", "feature", "category", "", "тариф"), ("calls", "feature", "numeric", "", "звонки"),
                     ("churn", "target", "category", "", "ушёл ли клиент")])
    sample = pd.DataFrame({"client_id": te["client_id"].values, "churn": "no"})
    _write(out / "synth_bin", "data", TASK_MD_BIN, tr, te.drop(columns=["churn"]), dd, sample,
           te[["client_id", "churn"]])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "tasks")
    make_reg(out)
    make_clf(out)
    make_bin(out)
    print("Задачи созданы в", out)
