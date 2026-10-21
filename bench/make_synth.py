"""Стенд: синтетическая бинарная задача с ROC-AUC, текстом, датой, пропусками и строковыми id.

python bench/make_synth.py <out_dir> [--n 8000]

Проверяет универсальность агента на формате, которого нет в публичных задачах.
Ответы test пишутся в answers.csv для bench/score.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS = ["отлично", "быстро", "удобно", "рекомендую", "доволен"]
NEG = ["плохо", "долго", "сломалось", "возврат", "ужасно"]
NEUTRAL = ["заказ", "доставка", "товар", "магазин", "курьер", "упаковка", "цена", "сайт"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n = int(a.n * 1.4)
    num = rng.normal(size=(n, 5))
    cat_a = rng.choice([f"A{i}" for i in range(10)], n)
    cat_b = rng.choice([f"B{i}" for i in range(200)], n)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    sent = rng.normal(size=n)
    texts = []
    for s in sent:
        words = list(rng.choice(NEUTRAL, 5))
        words += list(rng.choice(POS if s > 0 else NEG, 1 + int(abs(s) * 2)))
        rng.shuffle(words)
        texts.append(" ".join(words))
    logit = 1.2 * num[:, 0] - 0.8 * num[:, 1] ** 2 + 0.9 * sent + (cat_a == "A3") * 1.5 \
        + 0.5 * np.sin(dates.month.to_numpy() / 12 * 2 * np.pi)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"row_id": [f"row_{i:06d}" for i in range(n)]})
    for j in range(5):
        col = num[:, j].copy()
        col[rng.random(n) < 0.1] = np.nan
        df[f"x{j}"] = col
    df["segment"] = cat_a
    df["shop"] = cat_b
    df["order_date"] = dates.strftime("%Y-%m-%d")
    df["review"] = texts
    df["churn"] = np.where(y == 1, "yes", "no")
    out = Path(a.out_dir)
    data = out / "synth" / "participants"
    data.mkdir(parents=True, exist_ok=True)
    cut = a.n
    train, test = df.iloc[:cut], df.iloc[cut:]
    train.to_csv(data / "train.csv", index=False)
    test.drop(columns=["churn"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"row_id": test["row_id"], "churn": 0.5}).to_csv(data / "sample_submission.csv", index=False)
    test[["row_id", "churn"]].to_csv(out / "answers.csv", index=False)
    dd = pd.DataFrame({
        "feature": ["row_id"] + [f"x{j}" for j in range(5)] + ["segment", "shop", "order_date", "review", "churn"],
        "role": ["id"] + ["feature"] * 9 + ["target"],
        "dtype": ["category"] + ["numeric"] * 5 + ["category", "category", "datetime", "text", "category"],
        "description_ru": ["Идентификатор"] + [f"Показатель {j}" for j in range(5)]
        + ["Сегмент клиента", "Магазин", "Дата заказа", "Текст отзыва", "Ушёл ли клиент (yes/no)"]})
    dd.to_csv(data / "data_dictionary.csv", index=False)
    (out / "task.md").write_text(f"""# Отток клиентов

## Задача
Предскажите вероятность того, что клиент уйдёт (`churn` = yes).

## Данные
Пути указаны относительно каталога этого Markdown-файла:
- `synth/participants/train.csv` — {len(train)} строк;
- `synth/participants/test.csv` — {len(test)} строк без таргета;
- `synth/participants/data_dictionary.csv` — описание полей;
- `synth/participants/sample_submission.csv` — шаблон ответа.

## Что предсказать
Основная метрика — **ROC-AUC** по вероятности класса `yes`. Больше — лучше.

## Формат результата
Файл `submission.csv` в каталоге этого Markdown-файла: колонки `row_id,churn`, где `churn` — вероятность ухода.
""", encoding="utf-8")
    print(f"train={len(train)}, test={len(test)} → {out}")


if __name__ == "__main__":
    main()
