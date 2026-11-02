"""Task understanding: heuristic TaskSpec + reconciliation with the LLM Analyst."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import mla_metrics as MT
from mla_io_data import RawData, read_columns_as_str
from mla_utils import LOG

ID_NAME_RX = re.compile(r"^(id|index|idx|row_?id|record_?id|.*_id|id_.*|.*id)$", re.IGNORECASE)


@dataclass
class TargetSpec:
    name: str
    task_type: str  # regression | binary | multiclass
    pred_format: str  # value | label | proba | proba_matrix
    sub_columns: list
    classes: list | None = None  # canonical class keys in internal order (binary: [neg, pos])
    out_tokens: list | None = None  # raw output token per class index
    positive: str | None = None
    class_columns: dict | None = None  # proba_matrix: submission column -> class index

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.classes else 0

    @property
    def kind(self) -> str:
        return "regression" if self.task_type == "regression" else "classification"


@dataclass
class TaskSpec:
    id_col: str | None
    targets: list
    metric: str
    metric_source: str = "default"
    metric_text_priority: int = 9
    cv: dict = field(default_factory=lambda: {"type": "auto"})
    drop_hints: list = field(default_factory=list)
    cat_hints: list = field(default_factory=list)
    domain_notes: str = ""
    feature_ideas: list = field(default_factory=list)
    leak_notes: str = ""
    custom_metric_code: str | None = None
    notes: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d

    def summary(self) -> str:
        t = self.targets[0]
        s = f"target={t.name} type={t.task_type} format={t.pred_format} metric={self.metric}"
        if t.classes:
            s += f" classes={len(t.classes)}"
            if t.task_type == "binary":
                s += f" positive={t.positive!r}"
        if len(self.targets) > 1:
            s += f" (+{len(self.targets) - 1} more targets)"
        return s


# ---------------------------------------------------------------------------------------------
def canon_label(v: str) -> str:
    """Canonical class key: '1', '1.0', ' 1 ' -> '1'; 'True'/'true' -> 'True'."""
    s = str(v).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
        return repr(f)
    except Exception:
        pass
    if s.lower() in ("true", "false"):
        return s.lower().capitalize()
    return s


def _natural_key(s: str):
    try:
        return (0, float(s), s)
    except Exception:
        return (1, 0.0, s)


POS_PAIRS = [("0", "1"), ("False", "True"), ("no", "yes"), ("n", "y"), ("neg", "pos"), ("negative", "positive"),
             ("-1", "1"), ("нет", "да"), ("f", "t"), ("0", "2"), ("benign", "malignant"), ("normal", "abnormal"),
             ("normal", "attack"), ("legit", "fraud"), ("ham", "spam")]


def choose_positive(classes: list[str], counts: dict) -> str:
    low = {c.lower(): c for c in classes}
    for neg, pos in POS_PAIRS:
        if neg.lower() in low and pos.lower() in low:
            return low[pos.lower()]
    # minority class is usually the "event"
    return min(classes, key=lambda c: (counts.get(c, 0), c))


def _target_labels(raw: RawData, tname: str) -> tuple[pd.Series, pd.Series]:
    """Return (canonical labels, raw tokens) for a classification target (missing -> NaN)."""
    try:
        rs = read_columns_as_str(raw.train_path, [tname])[tname]
        if len(rs) != len(raw.train):
            raise ValueError
    except Exception:
        rs = raw.train[tname].astype(str).where(raw.train[tname].notna(), "")
    rs = rs.fillna("").astype(str)
    missing = rs.str.strip().isin(["", "nan", "NaN", "None", "null", "NULL"]) & raw.train[tname].isna()
    canon = rs.map(canon_label).where(~missing, np.nan)
    return canon, rs.where(~missing, np.nan)


def build_classes(raw: RawData, tname: str, positive_hint: str | None = None, task_type: str | None = None):
    canon, raw_tok = _target_labels(raw, tname)
    counts = canon.value_counts().to_dict()
    classes = sorted(counts.keys(), key=_natural_key)
    tokens = {}
    for c in classes:
        tokens[c] = raw_tok[canon == c].value_counts().index[0]
    positive = None
    if len(classes) == 2:
        positive = None
        if positive_hint is not None:
            ph = canon_label(positive_hint)
            if ph in classes:
                positive = ph
        if positive is None:
            positive = choose_positive(classes, counts)
        neg = [c for c in classes if c != positive][0]
        classes = [neg, positive]
    return classes, [tokens[c] for c in classes], positive, canon


def _is_classification_series(s: pd.Series) -> bool:
    if s.dtype == object or pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        x = pd.to_numeric(s.dropna(), errors="coerce")
        if x.notna().mean() > 0.98 and x.nunique() > 50:
            return False
        return True
    return False


def infer_task_type(s: pd.Series, metric_kind: str | None) -> str:
    x = s.dropna()
    nun = x.nunique()
    if _is_classification_series(s):
        return "binary" if nun <= 2 else "multiclass"
    xv = pd.to_numeric(x, errors="coerce").dropna()
    is_int = len(xv) > 0 and np.all(np.mod(xv.values[:20000], 1) == 0)
    if nun <= 2:
        return "binary" if metric_kind != "regression" else "regression"
    if metric_kind == "regression":
        return "regression"
    if metric_kind == "classification" and is_int and nun <= 100:
        return "multiclass"
    if is_int and nun <= 10 and metric_kind is None:
        return "multiclass"
    return "regression"


def heuristic_spec(raw: RawData, ddict_info: dict) -> TaskSpec:
    train, test, sample = raw.train, raw.test, raw.sample
    notes = []
    roles = {c: (d.get("role") or "").lower() for c, d in ddict_info.items()}
    dict_targets = [c for c, r in roles.items() if re.search(r"target|таргет|целев|^label$|^y$|^цель$", r.strip()) and c in train.columns]
    dict_ids = [c for c, r in roles.items() if re.search(r"^id$|\bid\b|идент|identifier|^key$|^index$", r.strip()) and c in test.columns]

    # ---- id column
    id_col = None
    if sample is not None and len(sample.columns):
        common = [c for c in sample.columns if c in test.columns]
        named = [c for c in common if ID_NAME_RX.match(c)]
        if dict_ids and dict_ids[0] in common:
            id_col = dict_ids[0]
        elif named:
            id_col = named[0]
        elif common:
            # the column whose values match test values one-to-one
            for c in common:
                try:
                    if set(sample[c].astype(str).head(200)) <= set(test[c].astype(str)):
                        id_col = c
                        break
                except Exception:
                    pass
            id_col = id_col or common[0]
    if id_col is None:
        cands = dict_ids + [c for c in test.columns if ID_NAME_RX.match(c) and test[c].is_unique]
        id_col = cands[0] if cands else None

    # ---- targets
    targets_cols, proba_matrix_cols = [], None
    if sample is not None:
        pred_cols = [c for c in sample.columns if c != id_col]
        in_train = [c for c in pred_cols if c in train.columns and c not in test.columns]
        if pred_cols and len(in_train) == len(pred_cols):
            targets_cols = in_train
        elif pred_cols:
            cand = [c for c in train.columns if c not in test.columns and c != id_col]
            best, best_ov = None, 0.0
            for c in cand:
                vals = set(train[c].dropna().astype(str).map(canon_label).unique()[:5000])
                ov = np.mean([canon_label(p) in vals or p in vals for p in pred_cols])
                if ov > best_ov:
                    best, best_ov = c, ov
            if best is not None and best_ov >= 0.5:
                targets_cols = [best]
                proba_matrix_cols = pred_cols
            elif in_train:
                targets_cols = in_train
    if not targets_cols:
        targets_cols = dict_targets or [c for c in train.columns if c not in test.columns and c != id_col]
        if len(targets_cols) > 1 and dict_targets:
            targets_cols = dict_targets
        notes.append("targets inferred without sample_submission")
    if not targets_cols:
        raise RuntimeError("Could not determine target column")

    # ---- metric (text)
    first = train[targets_cols[0]]
    cands = MT.find_metrics_in_text(raw.task_text)
    metric_kind = None
    if cands:
        top = cands[0][0]
        metric_kind = MT.REGISTRY[top].kind
    targets = []
    for tname in targets_cols:
        ttype = infer_task_type(train[tname], metric_kind)
        if proba_matrix_cols is not None and ttype == "regression":
            ttype = "multiclass"
        if ttype == "regression":
            targets.append(TargetSpec(tname, "regression", "value", [tname]))
            continue
        classes, tokens, positive, _ = build_classes(raw, tname)
        if len(classes) < 2:
            notes.append(f"target {tname} has <2 classes")
        ttype = "binary" if len(classes) == 2 else "multiclass"
        if proba_matrix_cols is not None:
            cc = {}
            for col in proba_matrix_cols:
                key = canon_label(col)
                if key in classes:
                    cc[col] = classes.index(key)
                else:  # e.g. 'class_3', 'prob_Normal'
                    m = [i for i, k in enumerate(classes) if col.lower().endswith(str(k).lower())]
                    if m:
                        cc[col] = m[0]
            targets.append(TargetSpec(tname, ttype, "proba_matrix", list(proba_matrix_cols), classes, tokens, positive, cc))
        else:
            targets.append(TargetSpec(tname, ttype, "label", [tname], classes, tokens, positive))

    if sample is not None and len(targets) == 1 and targets[0].pred_format != "proba_matrix":
        pred_cols = [c for c in sample.columns if c != id_col]
        if len(pred_cols) == 1 and pred_cols[0] != targets[0].name:
            notes.append(f"submission column '{pred_cols[0]}' differs from train target '{targets[0].name}'")
            targets[0].sub_columns = pred_cols
    t0 = targets[0]
    metric = MT.guess_metric_from_text(raw.task_text, t0.kind, t0.n_classes or None)
    prio = cands[0][1] if cands else 9
    source = "text" if metric else "default"
    # sample values hint for binary probability output
    proba_hint = False
    if sample is not None and t0.task_type == "binary" and t0.pred_format == "label":
        v = pd.to_numeric(sample[t0.sub_columns[0]], errors="coerce")
        if v.notna().mean() > 0.99 and ((v > 0) & (v < 1)).mean() > 0.5:
            proba_hint = True
    if t0.pred_format == "proba_matrix":
        proba_hint = True
    if metric is None:
        metric = MT.default_metric(t0.kind, t0.n_classes, proba_hint)
    spec = TaskSpec(id_col=id_col, targets=targets, metric=metric, metric_source=source, metric_text_priority=prio, notes=notes)
    finalize_formats(spec, proba_hint)
    return spec


def finalize_formats(spec: TaskSpec, proba_hint: bool = False) -> None:
    m = MT.REGISTRY.get(spec.metric)
    for t in spec.targets:
        if t.task_type == "regression":
            t.pred_format = "value"
            continue
        if t.pred_format == "proba_matrix":
            continue
        if m is not None and m.needs == "proba" and t.task_type == "binary":
            t.pred_format = "proba"
        elif proba_hint and t.task_type == "binary" and (m is None or m.needs != "label"):
            t.pred_format = "proba"
        else:
            t.pred_format = "label"
    # metric / task-kind consistency
    t0 = spec.targets[0]
    if m is None or m.kind != t0.kind:
        new = MT.default_metric(t0.kind, t0.n_classes, t0.pred_format in ("proba", "proba_matrix"))
        spec.notes.append(f"metric {spec.metric} inconsistent with {t0.kind}; using {new}")
        spec.metric = new
        spec.metric_source = "default"
    elif m.binary_only and t0.task_type != "binary":
        alt = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(spec.metric, "accuracy")
        spec.notes.append(f"metric {spec.metric} is binary-only; using {alt}")
        spec.metric = alt
    # a label-format submission with a probability metric (e.g. multiclass AUC in one column) cannot be honoured
    if t0.kind == "classification" and t0.pred_format == "label" and MT.REGISTRY[spec.metric].needs == "proba" and t0.task_type != "binary":
        spec.notes.append(f"metric {spec.metric} needs probabilities but submission takes labels; optimizing accuracy-like labels")


def reconcile_with_llm(spec: TaskSpec, llm: dict, raw: RawData) -> TaskSpec:
    """Merge LLM Analyst output into the heuristic spec. Facts are verified; heuristics from
    sample_submission are authoritative for output format."""
    if not isinstance(llm, dict):
        return spec
    notes = spec.notes
    train = raw.train
    t0 = spec.targets[0]
    # --- task type (only for ambiguous numeric targets)
    ltype = str(llm.get("task_type", "")).lower()
    if ltype in ("regression", "binary", "multiclass") and ltype != t0.task_type and t0.pred_format != "proba_matrix":
        s = train[t0.name]
        ambiguous = pd.api.types.is_numeric_dtype(s) and s.nunique() <= 100
        # an explicit metric in the task text decides the task kind; the LLM may not contradict it
        lkind = "regression" if ltype == "regression" else "classification"
        if spec.metric_source == "text" and MT.REGISTRY[spec.metric].kind != lkind:
            ambiguous = False
            notes.append(f"LLM task type {ltype} ignored: contradicts metric {spec.metric}")
        if ambiguous:
            notes.append(f"LLM changed task type {t0.task_type} -> {ltype}")
            if ltype == "regression":
                spec.targets[0] = TargetSpec(t0.name, "regression", "value", t0.sub_columns)
            else:
                classes, tokens, positive, _ = build_classes(raw, t0.name)
                spec.targets[0] = TargetSpec(t0.name, "binary" if len(classes) == 2 else "multiclass", "label",
                                             t0.sub_columns, classes, tokens, positive)
            t0 = spec.targets[0]
    # --- metric
    lm = str(llm.get("metric", "")).strip().lower()
    alias = {"f1-macro": "f1_macro", "macro_f1": "f1_macro", "auc": "roc_auc", "roc-auc": "roc_auc", "log_loss": "logloss",
             "cross_entropy": "logloss", "rmse_log": "rmsle", "weighted_f1": "f1_weighted", "micro_f1": "f1_micro",
             "quadratic_weighted_kappa": "qwk", "cohen_kappa": "kappa", "average_precision": "pr_auc", "r^2": "r2"}
    lm = alias.get(lm, lm)
    if lm in MT.REGISTRY and MT.REGISTRY[lm].kind == t0.kind:
        if lm != spec.metric:
            if spec.metric_source == "text" and spec.metric_text_priority <= 0:
                notes.append(f"LLM metric {lm} ignored; explicit metric line says {spec.metric}")
            else:
                notes.append(f"metric {spec.metric} -> {lm} (LLM)")
                spec.metric = lm
                spec.metric_source = "llm"
    elif lm == "custom" and llm.get("custom_metric_code") and not (spec.metric_source == "text" and spec.metric_text_priority <= 0):
        spec.custom_metric_code = str(llm.get("custom_metric_code"))
        name = MT.register_custom_metric(spec.custom_metric_code, t0.kind)
        if name:
            notes.append("custom metric registered from LLM code")
            spec.metric = name
            spec.metric_source = "llm"
        else:
            notes.append("LLM custom metric code failed validation; keeping " + spec.metric)
    # --- positive label
    pl = llm.get("positive_label")
    if t0.task_type == "binary" and pl not in (None, "", "null"):
        key = canon_label(pl)
        if key in t0.classes and key != t0.positive:
            neg = [c for c in t0.classes if c != key][0]
            tok = dict(zip(t0.classes, t0.out_tokens))
            t0.classes = [neg, key]
            t0.out_tokens = [tok[neg], tok[key]]
            t0.positive = key
            notes.append(f"positive label set to {key!r} (LLM)")
    # --- hints
    for k_src, k_dst in (("drop_columns", "drop_hints"), ("categorical_columns", "cat_hints")):
        v = llm.get(k_src) or []
        if isinstance(v, list):
            vals = []
            for item in v:
                name = item.get("column") if isinstance(item, dict) else item
                if isinstance(name, str) and name in train.columns and name not in [t.name for t in spec.targets] and name != spec.id_col:
                    vals.append(name)
            setattr(spec, k_dst, vals)
    cv = llm.get("cv_strategy") or llm.get("cv")
    if isinstance(cv, dict):
        spec.cv = {"type": str(cv.get("type", "auto")).lower(), "group_column": cv.get("group_column"),
                   "time_column": cv.get("time_column"), "reason": str(cv.get("reasoning", cv.get("reason", "")))[:500]}
    spec.domain_notes = str(llm.get("domain_summary", llm.get("domain_notes", "")))[:3000]
    ideas = llm.get("feature_ideas") or []
    if isinstance(ideas, list):
        spec.feature_ideas = [str(i)[:400] for i in ideas[:15]]
    spec.leak_notes = str(llm.get("potential_leaks", ""))[:1000]
    finalize_formats(spec, proba_hint=(t0.pred_format == "proba"))
    return spec
