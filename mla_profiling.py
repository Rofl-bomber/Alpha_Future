"""Dataset profiling and compact text summaries for LLM prompts."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mla_utils import fmt_num, truncate

DATE_RX = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?\s*$|^\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")


def _sample_values(s: pd.Series, k: int = 4) -> list:
    vals = s.dropna().unique()[:k]
    out = []
    for v in vals:
        if isinstance(v, float):
            out.append(fmt_num(v))
        else:
            out.append(truncate(str(v), 40))
    return out


def looks_like_date(s: pd.Series) -> bool:
    x = s.dropna().astype(str)
    if len(x) == 0:
        return False
    x = x.sample(min(len(x), 300), random_state=0)
    return x.map(lambda v: bool(DATE_RX.match(v))).mean() > 0.9


def looks_like_text(s: pd.Series) -> bool:
    x = s.dropna().astype(str)
    if len(x) < 20:
        return False
    x = x.sample(min(len(x), 500), random_state=0)
    avg_len = x.str.len().mean()
    avg_words = x.str.split().map(len).mean()
    uniq = s.nunique(dropna=True) / max(1, s.notna().sum())
    return avg_len > 25 and avg_words >= 3 and uniq > 0.2


def profile_column(tr: pd.Series, te: pd.Series | None) -> dict:
    p = {"dtype": str(tr.dtype)}
    n = len(tr)
    p["missing_train"] = float(tr.isna().mean()) if n else 0.0
    p["missing_test"] = float(te.isna().mean()) if te is not None and len(te) else None
    nun = int(tr.nunique(dropna=True))
    p["nunique"] = nun
    p["unique_ratio"] = nun / max(1, tr.notna().sum())
    p["examples"] = _sample_values(tr)
    if pd.api.types.is_bool_dtype(tr):
        p["kind"] = "bool"
    elif pd.api.types.is_numeric_dtype(tr):
        x = tr.dropna().astype(float)
        if len(x):
            q = np.quantile(x, [0, 0.01, 0.5, 0.99, 1.0])
            p.update({"min": float(q[0]), "p01": float(q[1]), "median": float(q[2]), "p99": float(q[3]), "max": float(q[4]),
                      "mean": float(x.mean()), "std": float(x.std()) if len(x) > 1 else 0.0,
                      "is_integer": bool(np.all(np.mod(x.values[:5000], 1) == 0))})
            p["zero_frac"] = float((x == 0).mean())
            neg_codes = [v for v in (-1, -9, -99, -999, -9999, 999, 9999) if (x == v).mean() > 0.01]
            if neg_codes:
                p["possible_sentinels"] = neg_codes
        p["kind"] = "numeric"
    else:
        s = tr.astype("object")
        if looks_like_date(s):
            p["kind"] = "datetime"
        elif looks_like_text(s):
            p["kind"] = "text"
            p["avg_len"] = float(s.dropna().astype(str).str.len().mean())
        else:
            p["kind"] = "categorical"
        vc = s.value_counts(dropna=True)
        p["top_values"] = {truncate(str(k), 30): int(v) for k, v in vc.head(6).items()}
    if te is not None and p["kind"] in ("categorical", "numeric") and nun > 0:
        try:
            tr_vals = set(tr.dropna().unique()[:200000])
            te_nonnull = te.dropna()
            if len(te_nonnull):
                p["test_in_train_frac"] = float(te_nonnull.isin(tr_vals).mean())
        except Exception:
            pass
    p["is_id_like"] = bool(p["unique_ratio"] > 0.999 and p["missing_train"] == 0 and n > 50 and
                           (p["kind"] in ("categorical",) or (p["kind"] == "numeric" and p.get("is_integer", False))))
    p["is_constant"] = nun <= 1
    return p


def profile_dataset(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        try:
            out[c] = profile_column(train[c], test[c] if c in test.columns else None)
        except Exception as e:  # pragma: no cover
            out[c] = {"dtype": str(train[c].dtype), "kind": "unknown", "error": str(e)}
    return out


def target_summary(y: pd.Series, task_type: str) -> str:
    if task_type == "regression":
        x = pd.to_numeric(y, errors="coerce").dropna()
        q = np.quantile(x, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
        skew = float(x.skew()) if len(x) > 2 else 0.0
        return (f"regression target: n={len(x)}, mean={fmt_num(x.mean())}, std={fmt_num(x.std())}, skew={fmt_num(skew, 3)}, "
                f"min={fmt_num(q[0])}, p01={fmt_num(q[1])}, p25={fmt_num(q[2])}, median={fmt_num(q[3])}, p75={fmt_num(q[4])}, "
                f"p99={fmt_num(q[5])}, max={fmt_num(q[6])}, negative_frac={fmt_num((x < 0).mean(), 3)}")
    vc = y.astype(str).value_counts()
    tot = vc.sum()
    parts = [f"{truncate(k, 30)}: {v} ({100 * v / tot:.2f}%)" for k, v in vc.head(30).items()]
    return f"classification target with {len(vc)} classes: " + "; ".join(parts)


def column_line(name: str, p: dict, desc: str | None = None, max_desc: int = 140) -> str:
    kind = p.get("kind", "?")
    bits = [f"{name} [{kind}]"]
    if p.get("missing_train", 0) > 0 or (p.get("missing_test") or 0) > 0:
        bits.append(f"miss={100 * p.get('missing_train', 0):.1f}%/{100 * (p.get('missing_test') or 0):.1f}%")
    bits.append(f"nuniq={p.get('nunique')}")
    if kind == "numeric" and "min" in p:
        bits.append(f"range=[{fmt_num(p['min'])}..{fmt_num(p['max'])}] med={fmt_num(p['median'])}")
        if p.get("possible_sentinels"):
            bits.append(f"sentinels?={p['possible_sentinels']}")
    elif "top_values" in p:
        tv = ", ".join(f"{k}:{v}" for k, v in list(p["top_values"].items())[:4])
        bits.append(f"top={{{tv}}}")
    if p.get("test_in_train_frac") is not None and p.get("nunique", 0) > 20 and kind == "categorical":
        bits.append(f"test_vals_seen_in_train={100 * p['test_in_train_frac']:.0f}%")
    if desc:
        bits.append(f"— {truncate(desc, max_desc)}")
    return " ".join(bits)


def dataset_overview_text(profile: dict, ddict_info: dict, max_chars: int = 9000, desc_len: int = 140) -> str:
    lines = []
    for c, p in profile.items():
        d = ddict_info.get(c, {})
        desc = d.get("description")
        if d.get("unit"):
            desc = f"{desc or ''} (unit: {d['unit']})"
        lines.append("- " + column_line(c, p, desc, desc_len))
    text = "\n".join(lines)
    if len(text) > max_chars:
        # shrink descriptions first
        lines = []
        for c, p in profile.items():
            d = ddict_info.get(c, {})
            lines.append("- " + column_line(c, p, d.get("description"), 50))
        text = "\n".join(lines)
    return truncate(text, max_chars, tail=500)
