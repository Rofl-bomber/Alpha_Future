"""TaskSpec по правилам: пути из task.md, роли колонок из data_dictionary.csv,
колонки сабмита из sample_submission.csv, метрика регуляркой, тип задачи по таргету."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")


@dataclass
class TaskSpec:
    task_md: Path
    task_dir: Path
    task_text: str
    train_path: Path
    test_path: Path
    dict_path: Path | None
    sample_sub_path: Path | None
    output_path: Path

    id_col: str | None
    target: str
    feature_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    descriptions: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    task_type: str = "regression"        # regression | binary | multiclass
    metric: str = "rmse"                 # rmse | f1_macro
    greater_is_better: bool = False
    classes: list = field(default_factory=list)
    submission_columns: list[str] = field(default_factory=list)
    prediction_kind: str = "value"       # value | label | proba
    notes: list[str] = field(default_factory=list)

    @property
    def is_clf(self) -> bool:
        return self.task_type in ("binary", "multiclass")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def short(self) -> dict:
        d = asdict(self)
        for k in ("task_text", "descriptions", "units"):
            d.pop(k, None)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        d["classes"] = [str(c) for c in self.classes][:50]
        return d

    def brief(self) -> str:
        """Короткое описание задачи для промптов."""
        direction = "больше — лучше" if self.greater_is_better else "меньше — лучше"
        s = [f"Тип задачи: {self.task_type}; таргет: {self.target}; метрика: {self.metric} ({direction})."]
        if self.is_clf:
            s.append(f"Классы ({self.n_classes}): " + ", ".join(map(str, self.classes[:30])))
        s.append(f"Признаков: {len(self.feature_cols)} (категориальных {len(self.cat_cols)}, числовых {len(self.num_cols)}).")
        return " ".join(s)


# ---------------------------------------------------------------- пути

def _find_paths(task_md: Path, text: str) -> dict[str, Path | None]:
    base = task_md.parent
    found: dict[str, Path | None] = {k: None for k in DATA_FILES}
    for m in re.findall(r"[\w./\\-]+\.csv", text):
        rel = m.strip("`'\"()[] ").replace("\\", "/")
        name = Path(rel).name
        if name in found and found[name] is None:
            p = (base / rel).resolve()
            if p.exists():
                found[name] = p
    for name in DATA_FILES:
        if found[name] is None:
            cands = sorted(base.rglob(name), key=lambda p: (len(p.parts), str(p)))
            cands = [c for c in cands if "agent_runs" not in c.parts and ".agent_cache" not in c.parts]
            if cands:
                found[name] = cands[0].resolve()
    return found


# ---------------------------------------------------------------- метрика

def _detect_metric(text: str) -> str | None:
    t = text.lower()
    has_f1_macro = bool(re.search(r"macro[\s_-]*f1|f1[\s_-]*macro|average\s*=\s*['\"]macro['\"]", t))
    has_rmse = bool(re.search(r"\brmse\b|root[\s_-]+mean[\s_-]+squared", t))
    if has_f1_macro and not has_rmse:
        return "f1_macro"
    if has_rmse and not has_f1_macro:
        return "rmse"
    if has_rmse and has_f1_macro:
        # Берём ту, что ближе к слову «метрика»/«основная»
        m = re.search(r"(основн\w*\s+метрик\w*|метрик\w*)[^\n]{0,80}", t)
        if m:
            frag = m.group(0)
            if re.search(r"rmse", frag):
                return "rmse"
            if re.search(r"f1", frag):
                return "f1_macro"
    return None


# ---------------------------------------------------------------- сборка

def build_spec(task_md_path: str | Path) -> TaskSpec:
    task_md = Path(task_md_path).resolve()
    text = task_md.read_text(encoding="utf-8")
    paths = _find_paths(task_md, text)
    if paths["train.csv"] is None or paths["test.csv"] is None:
        raise FileNotFoundError(f"Не найдены train.csv / test.csv рядом с {task_md}")
    notes: list[str] = []

    train_head = pd.read_csv(paths["train.csv"], nrows=2000, low_memory=False)
    test_head = pd.read_csv(paths["test.csv"], nrows=2000, low_memory=False)
    sample = pd.read_csv(paths["sample_submission.csv"], nrows=50) if paths["sample_submission.csv"] else None

    id_col = target = None
    feature_cols: list[str] = []
    dtypes: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    units: dict[str, str] = {}

    if paths["data_dictionary.csv"] is not None:
        dd = pd.read_csv(paths["data_dictionary.csv"])
        dd.columns = [c.strip().lower() for c in dd.columns]
        name_col = "feature" if "feature" in dd.columns else dd.columns[0]
        for _, r in dd.iterrows():
            name = str(r[name_col]).strip()
            role = str(r.get("role", "feature")).strip().lower()
            dt = str(r.get("dtype", "")).strip().lower()
            dtypes[name] = dt
            if isinstance(r.get("description_ru"), str):
                descriptions[name] = r["description_ru"]
            if isinstance(r.get("unit"), str):
                units[name] = r["unit"]
            if role == "id" and id_col is None:
                id_col = name
            elif role == "target" and target is None:
                target = name
            elif role == "feature":
                feature_cols.append(name)

    # Фолбэк без словаря: колонки сабмита и разница train/test.
    if target is None:
        diff = [c for c in train_head.columns if c not in test_head.columns]
        if sample is not None:
            cand = [c for c in sample.columns if c in diff]
            target = cand[0] if cand else (diff[0] if diff else None)
        elif diff:
            target = diff[0]
        notes.append("таргет определён без словаря")
    if target is None:
        raise ValueError("Не удалось определить таргет")
    if id_col is None and sample is not None:
        c0 = sample.columns[0]
        if c0 in test_head.columns and c0 != target:
            id_col = c0
    if not feature_cols:
        feature_cols = [c for c in test_head.columns if c not in (id_col, target)]
    feature_cols = [c for c in feature_cols if c in train_head.columns and c in test_head.columns]

    cat_cols, num_cols = [], []
    for c in feature_cols:
        dt = dtypes.get(c, "")
        is_num_data = pd.api.types.is_numeric_dtype(train_head[c])
        if dt in ("category", "categorical", "string", "object", "text"):
            cat_cols.append(c)
        elif dt == "binary":
            (num_cols if is_num_data else cat_cols).append(c)
        elif dt == "numeric":
            if is_num_data:
                num_cols.append(c)
            else:
                share = pd.to_numeric(train_head[c], errors="coerce").notna().mean()
                (num_cols if share > 0.9 else cat_cols).append(c)
        else:
            (num_cols if is_num_data else cat_cols).append(c)

    # Тип задачи по таргету и метрика.
    y = pd.read_csv(paths["train.csv"], usecols=[target], low_memory=False)[target].dropna()
    metric = _detect_metric(text)
    numeric_y = pd.api.types.is_numeric_dtype(y)
    nunique = y.nunique()
    if not numeric_y:
        task_type = "binary" if nunique == 2 else "multiclass"
    elif metric == "f1_macro" or (nunique <= 2) or (
        nunique <= 30 and np.allclose(y.values, np.round(y.values)) and metric != "rmse"
    ):
        task_type = "binary" if nunique == 2 else "multiclass"
    else:
        task_type = "regression"
    if metric is None:
        metric = "rmse" if task_type == "regression" else "f1_macro"
        notes.append(f"метрика не найдена в task.md, взята по типу задачи: {metric}")
    if task_type == "regression" and metric != "rmse":
        notes.append("конфликт: метрика классификации при числовом таргете, оставлена rmse")
        metric = "rmse"
    if task_type != "regression" and metric == "rmse":
        notes.append("конфликт: rmse при классификации, взята f1_macro")
        metric = "f1_macro"

    classes = sorted(y.unique().tolist()) if task_type != "regression" else []

    sub_cols = list(sample.columns) if sample is not None else [c for c in (id_col, target) if c]
    pred_kind = "value" if task_type == "regression" else "label"
    if task_type != "regression" and sample is not None and target not in sample.columns:
        pred_kind = "proba"

    return TaskSpec(
        task_md=task_md, task_dir=task_md.parent, task_text=text,
        train_path=paths["train.csv"], test_path=paths["test.csv"],
        dict_path=paths["data_dictionary.csv"], sample_sub_path=paths["sample_submission.csv"],
        output_path=task_md.parent / "submission.csv",
        id_col=id_col, target=target, feature_cols=feature_cols,
        cat_cols=cat_cols, num_cols=num_cols, descriptions=descriptions, units=units,
        task_type=task_type, metric=metric, greater_is_better=(metric != "rmse"),
        classes=classes, submission_columns=sub_cols, prediction_kind=pred_kind, notes=notes,
    )


# ---------------------------------------------------------------- данные

@dataclass
class Data:
    X: pd.DataFrame            # train+test признаки, индекс 0..n_all-1
    n_train: int
    y: np.ndarray              # для классификации — индексы классов 0..K-1
    y_raw: pd.Series
    test_ids: pd.Series | None
    test_index: pd.Index       # исходный порядок строк test

    @property
    def n_test(self) -> int:
        return len(self.X) - self.n_train


def load_data(spec: TaskSpec, subsample: int = 0, seed: int = 42) -> Data:
    train = pd.read_csv(spec.train_path, low_memory=False)
    test = pd.read_csv(spec.test_path, low_memory=False)
    train = train[train[spec.target].notna()].reset_index(drop=True)
    if subsample and len(train) > subsample:
        train = train.sample(n=subsample, random_state=seed).reset_index(drop=True)
    X = pd.concat([train[spec.feature_cols], test[spec.feature_cols]], axis=0, ignore_index=True)
    for c in spec.num_cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].astype("float64")
    for c in spec.cat_cols:
        X[c] = X[c].astype("string").astype(object).where(X[c].notna(), np.nan)
    y_raw = train[spec.target]
    if spec.is_clf:
        mapping = {c: i for i, c in enumerate(spec.classes)}
        y = y_raw.map(mapping).astype(int).values
    else:
        y = y_raw.astype(float).values
    ids = test[spec.id_col] if spec.id_col and spec.id_col in test.columns else None
    return Data(X=X, n_train=len(train), y=y, y_raw=y_raw, test_ids=ids, test_index=test.index)
