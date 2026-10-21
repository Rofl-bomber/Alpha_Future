"""TaskSpec: строгое описание задачи, которым пользуются все остальные модули.

Порядок доверия: data_dictionary.csv → sample_submission.csv → train.csv → task.md
(правила и Аналитик-LLM) → сверка. Если LLM противоречит данным, побеждают данные,
а конфликт пишется в spec.conflicts.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import task_metrics

log = logging.getLogger("agent")

DATA_EXT = (".csv", ".tsv", ".parquet")


@dataclass
class TaskSpec:
    task_md: str
    task_dir: str
    train_path: str
    test_path: str
    dict_path: str | None = None
    sample_path: str | None = None
    output_path: str = ""
    title: str = ""
    task_type: str = "regression"
    targets: list = field(default_factory=list)
    id_col: str | None = None
    metric: str = "rmse"
    greater_is_better: bool = False
    metric_need: str = "value"
    metric_source: str = "default"
    prediction_kind: str = "value"  # value | label | proba | proba_columns
    submission_columns: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    feature_cols: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)
    num_cols: list = field(default_factory=list)
    text_cols: list = field(default_factory=list)
    datetime_cols: list = field(default_factory=list)
    descriptions: dict = field(default_factory=dict)
    group_hint: str | None = None
    time_hint: str | None = None
    out_of_library: bool = False
    out_of_library_reason: str = ""
    custom_metric_code: str | None = None
    notes: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    dropped_target_nan: int = 0

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.task_type in ("binary", "multiclass", "multilabel") else 0

    @property
    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass", "multilabel")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1, default=_json_default)

    def save(self, path):
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "TaskSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def short(self) -> str:
        return (f"тип={self.task_type}, таргет={self.targets}, метрика={self.metric} "
                f"({'больше' if self.greater_is_better else 'меньше'} — лучше), id={self.id_col}, "
                f"признаков={len(self.feature_cols)} (кат. {len(self.cat_cols)}, числ. {len(self.num_cols)}, "
                f"текст {len(self.text_cols)}, даты {len(self.datetime_cols)}), классов={self.n_classes}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _py(v):
    """Значение метки класса в JSON-совместимом виде."""
    if isinstance(v, np.generic):
        return v.item()
    return v


@dataclass
class TaskData:
    train: pd.DataFrame
    test: pd.DataFrame
    y: np.ndarray  # закодированный таргет по строкам train
    X_all: pd.DataFrame  # признаки train + test (в этом порядке), типизированные
    n_train: int
    test_ids: pd.Series | None
    train_ids: pd.Series | None


# ---------------------------------------------------------------- поиск файлов


def _role_by_name(name: str):
    n = name.lower()
    if "sample" in n or "example" in n:
        return "sample"
    if "dict" in n or "description" in n or "schema" in n or "columns" in n:
        return "dict"
    if "train" in n:
        return "train"
    if "test" in n:
        return "test"
    return None


def find_files(task_md: Path, text: str) -> dict:
    base = task_md.parent
    found = {}
    for raw in re.findall(r"[`\"'\(\s]([\w\-./\\]+\.(?:csv|tsv|parquet))", " " + text):
        p = (base / raw.replace("\\", "/")).resolve()
        role = _role_by_name(p.name)
        if role and p.exists() and role not in found:
            found[role] = p
    if "train" not in found or "test" not in found:
        cands = sorted(list(base.glob("*")) + list(base.glob("*/*")) + list(base.glob("*/*/*"))
                       + list(base.glob("*/*/*/*")), key=lambda q: len(q.parts))
        for p in cands:
            if p.is_file() and p.suffix.lower() in DATA_EXT:
                role = _role_by_name(p.name)
                if role and role not in found and "submission" != p.stem.lower():
                    found[role] = p.resolve()
    # dict/sample ищем рядом с train, если в тексте их не нашли
    if "train" in found:
        tdir = found["train"].parent
        for p in tdir.iterdir():
            role = _role_by_name(p.name)
            if role in ("dict", "sample", "test") and role not in found and p.suffix.lower() in DATA_EXT:
                found[role] = p.resolve()
    return found


def _output_path(task_md: Path, text: str, inputs: set) -> Path:
    for raw in re.findall(r"`([^`\s]+\.csv)`", text):
        name = Path(raw).name.lower()
        if "sample" in name or name in {Path(i).name.lower() for i in inputs}:
            continue
        if "submission" in name or "prediction" in name or "answer" in name:
            p = (task_md.parent / raw).resolve()
            return p
    return (task_md.parent / "submission.csv").resolve()


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if df.shape[1] == 1 and sep == ",":
        df2 = pd.read_csv(path, sep=None, engine="python")
        if df2.shape[1] > 1:
            df = df2
    return df


# ---------------------------------------------------------------- правила


def _read_dict(path):
    if not path:
        return None
    try:
        d = read_table(path)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in d.columns}
    name_col = next((cols[k] for k in ("feature", "column", "name", "field", "признак", "колонка") if k in cols), d.columns[0])
    out = pd.DataFrame({"feature": d[name_col].astype(str).str.strip()})
    for key, variants in (("role", ("role", "роль", "type_role")), ("dtype", ("dtype", "type", "тип")),
                          ("unit", ("unit", "units", "единицы", "формат")),
                          ("description", ("description_ru", "description", "описание", "desc"))):
        col = next((cols[v] for v in variants if v in cols), None)
        out[key] = d[col].astype(str).str.strip().str.lower() if col and key in ("role", "dtype") else (
            d[col].astype(str) if col else "")
    return out


def _is_text(s: pd.Series) -> bool:
    v = s.dropna().astype(str)
    if len(v) == 0:
        return False
    v = v.sample(min(len(v), 2000), random_state=0)
    mean_len = v.str.len().mean()
    words = v.str.split().str.len().mean()
    uniq = s.nunique(dropna=True) / max(1, s.notna().sum())
    return bool((mean_len > 30 and words > 4 and uniq > 0.2) or (words > 8 and uniq > 0.1))


def _is_datetime(s: pd.Series) -> bool:
    v = s.dropna().astype(str)
    if len(v) == 0 or not v.str.contains(r"\d").mean() > 0.9:
        return False
    v = v.sample(min(len(v), 500), random_state=0)
    if not v.str.contains(r"[-/.:T ]").mean() > 0.9 or v.str.len().median() < 6:
        return False
    try:
        parsed = pd.to_datetime(v, errors="coerce", format="mixed")
    except Exception:
        return False
    return bool(parsed.notna().mean() > 0.9)


def _guess_type(y: pd.Series | pd.DataFrame, metric_name: str | None):
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        vals = pd.unique(y.values.ravel())
        vals = [v for v in vals if not pd.isna(v)]
        if set(map(float, vals)) <= {0.0, 1.0} if all(isinstance(v, (int, float, np.number, bool)) for v in vals) else False:
            return "multilabel"
        if all(pd.api.types.is_numeric_dtype(y[c]) for c in y.columns):
            return "multi_regression"
        return "unsupported"
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    yv = y.dropna()
    nun = yv.nunique()
    metric = task_metrics.REGISTRY.get(metric_name) if metric_name else None
    cls_metric = metric is not None and metric.need in ("label", "proba") or metric_name == "f1"
    reg_metric = metric is not None and metric.need == "value"
    if not pd.api.types.is_numeric_dtype(yv) or pd.api.types.is_bool_dtype(yv):
        return "binary" if nun == 2 else "multiclass"
    if nun <= 2:
        return "regression" if reg_metric else "binary"
    integer_like = bool(np.all(np.isclose(yv.values, np.round(yv.values))))
    if cls_metric and integer_like and nun <= 200:
        return "multiclass"
    if not reg_metric and integer_like and nun <= 10:
        return "multiclass"
    return "regression"


def build_rules_spec(task_md: Path) -> tuple:
    """Спецификация по правилам и данным, без LLM. Возвращает (spec, train, test, dict_df, sample, text)."""
    task_md = Path(task_md).resolve()
    text = task_md.read_text(encoding="utf-8", errors="replace")
    files = find_files(task_md, text)
    if "train" not in files or "test" not in files:
        raise FileNotFoundError(f"Не найдены train/test рядом с {task_md}: {files}")
    train = read_table(files["train"])
    test = read_table(files["test"])
    dd = _read_dict(files.get("dict"))
    sample = None
    if files.get("sample"):
        try:
            sample = read_table(files["sample"])
        except Exception:
            sample = None
    title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip().startswith("#")), task_md.stem)

    roles = {}
    descriptions = {}
    dtypes_dd = {}
    if dd is not None:
        for _, r in dd.iterrows():
            roles[r["feature"]] = r["role"]
            dtypes_dd[r["feature"]] = r["dtype"]
            desc = str(r.get("description", "") or "")
            unit = str(r.get("unit", "") or "")
            descriptions[r["feature"]] = (desc + (f" [{unit}]" if unit and unit != "nan" else "")).strip()

    # id
    id_col = next((c for c, r in roles.items() if r in ("id", "identifier", "идентификатор") and c in test.columns), None)
    if id_col is None and sample is not None and sample.columns[0] in test.columns:
        id_col = sample.columns[0]
    if id_col is None:
        id_col = next((c for c in test.columns if c.lower() in ("id", "row_id", "index")), None)

    # таргеты
    targets = [c for c, r in roles.items() if r in ("target", "label", "таргет", "целевая") and c in train.columns]
    if not targets and sample is not None:
        targets = [c for c in sample.columns if c != id_col and c in train.columns]
    if not targets:
        targets = [c for c in train.columns if c not in test.columns and c != id_col]
    if not targets:
        raise ValueError("Не удалось определить целевую переменную")

    metric_name, conf = task_metrics.detect_metric(text)
    ytr = train[targets] if len(targets) > 1 else train[targets[0]]
    task_type = _guess_type(ytr, metric_name)

    ignore_roles = ("id", "target", "ignore", "drop", "leak", "identifier", "таргет")
    feature_cols = [c for c in train.columns if c in test.columns and c != id_col and c not in targets
                    and roles.get(c, "feature") not in ignore_roles]

    spec = TaskSpec(
        task_md=str(task_md), task_dir=str(task_md.parent), train_path=str(files["train"]),
        test_path=str(files["test"]), dict_path=str(files["dict"]) if files.get("dict") else None,
        sample_path=str(files["sample"]) if files.get("sample") else None,
        output_path=str(_output_path(task_md, text, {str(p) for p in files.values()})),
        title=title, task_type=task_type, targets=targets, id_col=id_col, feature_cols=feature_cols,
        descriptions=descriptions,
    )
    if metric_name:
        spec.metric = task_metrics.resolve(metric_name, task_type)
        spec.metric_source = f"текст ({conf})"
    else:
        spec.metric = task_metrics.default_metric(task_type)
        spec.metric_source = "по умолчанию"
    _finish_metric(spec)
    _type_columns(spec, train, test, dtypes_dd)
    _classes_and_kind(spec, train, sample)
    if task_type == "unsupported":
        spec.out_of_library = True
        spec.out_of_library_reason = "несколько нечисловых таргетов"
    return spec, train, test, dd, sample, text


def _finish_metric(spec: TaskSpec):
    if spec.metric not in task_metrics.REGISTRY or spec.task_type not in task_metrics.REGISTRY[spec.metric].kinds:
        if spec.metric in task_metrics.REGISTRY:
            spec.conflicts.append(f"метрика {spec.metric} не подходит к типу {spec.task_type}")
        spec.metric = task_metrics.default_metric(spec.task_type)
        spec.metric_source += " → замена по типу задачи"
    m = task_metrics.REGISTRY[spec.metric]
    spec.greater_is_better = m.greater_is_better
    spec.metric_need = m.need


def _type_columns(spec: TaskSpec, train, test, dtypes_dd):
    both = pd.concat([train[spec.feature_cols], test[spec.feature_cols]], axis=0, ignore_index=True)
    cat, num, text_cols, dt = [], [], [], []
    for c in spec.feature_cols:
        s = both[c]
        ddt = dtypes_dd.get(c, "")
        if pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if any(k in ddt for k in ("categ", "категор")) and s.nunique() <= 1000:
                cat.append(c)
            else:
                num.append(c)
        elif any(k in ddt for k in ("date", "time", "дата")) or _is_datetime(s):
            dt.append(c)
        elif "text" in ddt or "текст" in ddt or _is_text(s):
            text_cols.append(c)
        else:
            cat.append(c)
    spec.cat_cols, spec.num_cols, spec.text_cols, spec.datetime_cols = cat, num, text_cols, dt


def _classes_and_kind(spec: TaskSpec, train, sample):
    if spec.task_type in ("binary", "multiclass"):
        y = train[spec.targets[0]].dropna()
        classes = sorted(pd.unique(y).tolist(), key=lambda v: (str(type(v)), v) if not isinstance(v, str) else ("str", v))
        try:
            classes = sorted(classes)
        except TypeError:
            classes = sorted(classes, key=str)
        if spec.task_type == "binary" and len(classes) == 2:
            pos = _positive_class(classes)
            classes = [c for c in classes if c != pos] + [pos]
        spec.classes = [_py(c) for c in classes]
    elif spec.task_type == "multilabel":
        spec.classes = list(spec.targets)

    sub_cols = list(sample.columns) if sample is not None else ([spec.id_col] if spec.id_col else []) + spec.targets
    spec.submission_columns = sub_cols
    if spec.task_type in ("regression", "multi_regression"):
        spec.prediction_kind = "value"
        return
    pred_cols = [c for c in sub_cols if c != spec.id_col]
    class_names = {str(c) for c in spec.classes}
    if spec.task_type == "multiclass" and len(pred_cols) > 1 and {str(c) for c in pred_cols} <= class_names | {""}:
        spec.prediction_kind = "proba_columns"
        if spec.metric_need == "label":
            spec.conflicts.append("шаблон ждёт вероятности по классам, метрика — метки; пишем вероятности")
        return
    if spec.metric_need == "proba":
        spec.prediction_kind = "proba"
    else:
        spec.prediction_kind = "label"
    if sample is not None and spec.task_type == "binary" and len(pred_cols) == 1:
        vals = pd.to_numeric(sample[pred_cols[0]], errors="coerce")
        if vals.notna().all() and ((vals > 0) & (vals < 1)).any():
            spec.prediction_kind = "proba"


def _positive_class(classes):
    lowered = {str(c).lower(): c for c in classes}
    for key in ("1", "1.0", "true", "yes", "да", "positive", "pos", "attack", "fraud", "default", "churn"):
        if key in lowered:
            return lowered[key]
    return classes[-1]


# ---------------------------------------------------------------- Аналитик


def apply_analyst(spec: TaskSpec, out: dict, train: pd.DataFrame) -> TaskSpec:
    """Слить ответ Аналитика с правилами. Данные важнее LLM; конфликты — в spec.conflicts."""
    if not out:
        return spec
    m = str(out.get("metric") or "").strip().lower()
    alias = task_metrics._first_alias(m) if m and m not in task_metrics.REGISTRY else m
    tt = str(out.get("task_type") or "").strip().lower()
    if tt and tt != spec.task_type:
        ytr = train[spec.targets[0]]
        if tt in ("binary", "multiclass") and spec.task_type == "regression" and pd.api.types.is_numeric_dtype(ytr) \
                and ytr.dropna().nunique() <= 200 and np.allclose(ytr.dropna(), np.round(ytr.dropna())):
            spec.notes.append(f"Аналитик: целочисленный таргет трактуем как {tt}")
            spec.task_type = "binary" if ytr.dropna().nunique() == 2 else "multiclass"
        else:
            spec.conflicts.append(f"Аналитик считает тип {tt}, данные — {spec.task_type}; оставляем данные")
    if alias and alias in task_metrics.REGISTRY:
        alias = task_metrics.resolve(alias, spec.task_type)
        if alias != spec.metric:
            if spec.metric_source.startswith("текст (high"):
                spec.conflicts.append(f"Аналитик предлагает метрику {alias}, в тексте найдена {spec.metric}; оставляем текст")
            else:
                spec.metric = alias
                spec.metric_source = "Аналитик"
    elif out.get("custom_metric_code") and m:
        code = str(out["custom_metric_code"])
        need = "value" if spec.task_type in ("regression", "multi_regression") else "label"
        gib = bool(out.get("greater_is_better")) if out.get("greater_is_better") is not None else spec.greater_is_better
        if _custom_metric_works(code, spec.task_type):
            spec.custom_metric_code = code
            spec.metric_need, spec.greater_is_better = need, gib
            spec.metric_source = f"Аналитик (своя метрика «{m[:40]}»)"
        else:
            spec.conflicts.append(f"код метрики «{m[:40]}» от Аналитика не прошёл проверку; оставляем {spec.metric}")
    if spec.task_type in ("binary", "multiclass") and not spec.classes:
        _classes_and_kind(spec, train, None)
    pos = out.get("positive_class")
    if spec.task_type == "binary" and pos is not None and len(spec.classes) == 2:
        match = [c for c in spec.classes if str(c) == str(pos)]
        if match and spec.classes[1] != match[0]:
            spec.classes = [c for c in spec.classes if c != match[0]] + [match[0]]
            spec.notes.append(f"положительный класс по Аналитику: {pos}")
    if out.get("out_of_library"):
        spec.out_of_library = True
        spec.out_of_library_reason = str(out.get("out_of_library_reason") or "по оценке Аналитика")[:300]
    for key in ("group_hint", "time_hint"):
        v = out.get(key)
        if v and v in spec.feature_cols:
            setattr(spec, key, v)
    for n in (out.get("notes") or [])[:8]:
        spec.notes.append(str(n)[:300])
    if spec.custom_metric_code is None:
        _finish_metric(spec)
    return spec


def _custom_metric_works(code: str, task_type: str) -> bool:
    """Проверить код метрики от LLM на игрушечных примерах: число, конечное, без исключений."""
    try:
        m = task_metrics.custom_metric(code, "value", True)
        if task_type in ("regression", "multi_regression"):
            y = np.array([1.0, 2.0, 3.0, 4.0])
            vals = [m.fn(y, y), m.fn(y, y[::-1])]
        else:
            y = np.array([0, 1, 1, 0])
            vals = [m.fn(y, y), m.fn(y, 1 - y)]
        return all(np.isfinite(v) for v in vals) and vals[0] != vals[1]
    except Exception:
        return False


# ---------------------------------------------------------------- данные для моделей


def prepare_data(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> TaskData:
    """Закодировать таргет, удалить строки без таргета, типизировать признаки train+test."""
    if spec.task_type in ("regression", "multi_regression", "multilabel"):
        mask = train[spec.targets].notna().all(axis=1)
    else:
        mask = train[spec.targets[0]].notna()
    spec.dropped_target_nan = int((~mask).sum())
    train = train.loc[mask].reset_index(drop=True)

    if spec.task_type == "regression":
        y = pd.to_numeric(train[spec.targets[0]], errors="coerce").to_numpy(dtype=float)
    elif spec.task_type == "multi_regression":
        y = train[spec.targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    elif spec.task_type == "multilabel":
        y = train[spec.targets].astype(float).to_numpy().round().astype(int)
    else:
        index = {str(c): i for i, c in enumerate(spec.classes)}
        y = train[spec.targets[0]].map(lambda v: index.get(str(_py(v)), -1)).to_numpy(dtype=int)
        if (y < 0).any():
            raise ValueError("не все метки train попали в список классов")

    X_all = pd.concat([train[spec.feature_cols], test[spec.feature_cols]], axis=0, ignore_index=True)
    for c in spec.num_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce").astype("float64")
    for c in spec.cat_cols:
        s = X_all[c]
        s = s.where(s.isna(), s.astype(str))
        X_all[c] = s.astype("category")
    for c in spec.text_cols + spec.datetime_cols:
        X_all[c] = X_all[c].astype("object").where(X_all[c].notna(), None)

    test_ids = test[spec.id_col].reset_index(drop=True) if spec.id_col and spec.id_col in test.columns else None
    train_ids = train[spec.id_col].reset_index(drop=True) if spec.id_col and spec.id_col in train.columns else None
    return TaskData(train=train, test=test.reset_index(drop=True), y=y, X_all=X_all, n_train=len(train),
                    test_ids=test_ids, train_ids=train_ids)


def dictionary_summary(spec: TaskSpec, data: TaskData, limit: int = 120) -> str:
    """Описание колонок для LLM: имя, тип, примеры, пропуски, описание из словаря."""
    lines = []
    X = data.X_all
    for c in spec.feature_cols[:limit]:
        kind = ("cat" if c in spec.cat_cols else "num" if c in spec.num_cols else
                "text" if c in spec.text_cols else "date")
        s = X[c]
        miss = float(s.isna().mean())
        if kind == "num":
            ex = f"min={s.min():.4g}, med={s.median():.4g}, max={s.max():.4g}" if s.notna().any() else "пусто"
        else:
            vc = s.astype(str).value_counts().head(4)
            ex = ", ".join(f"{k[:20]}" for k in vc.index) + f" (уник. {s.nunique()})"
        desc = spec.descriptions.get(c, "")
        lines.append(f"- {c} [{kind}] пропуски={miss:.0%}; {ex}" + (f"; {desc[:120]}" if desc else ""))
    if len(spec.feature_cols) > limit:
        lines.append(f"... и ещё {len(spec.feature_cols) - limit} колонок")
    return "\n".join(lines)
