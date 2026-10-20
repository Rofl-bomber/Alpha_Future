"""Запись submission.csv в каталог task.md: колонки из sample_submission, порядок строк как в test."""

from __future__ import annotations

import numpy as np
import pandas as pd


def write_submission(preds: np.ndarray, spec, data) -> str:
    cols = list(spec.submission_columns)
    out = pd.DataFrame(index=range(data.n_test))
    if spec.id_col and data.test_ids is not None:
        out[spec.id_col] = data.test_ids.values
    if spec.prediction_kind == "proba":
        for j, c in enumerate(spec.classes):
            out[str(c)] = preds[:, j]
    elif spec.is_clf:
        labels = np.asarray(spec.classes, dtype=object)[np.asarray(preds).argmax(axis=1)]
        out[spec.target] = labels
    else:
        out[spec.target] = np.asarray(preds, dtype=float)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan  # колонка из sample_submission, которую мы не умеем заполнять
    out = out[cols] if cols else out
    out.to_csv(spec.output_path, index=False)
    return str(spec.output_path)
