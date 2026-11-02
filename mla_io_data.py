"""Locate and robustly read task files."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mla_utils import LOG

ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")


@dataclass
class RawData:
    task_text: str
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame | None  # read as strings
    ddict: pd.DataFrame | None
    train_path: Path
    test_path: Path
    sample_path: Path | None
    ddict_path: Path | None
    extra_files: list


def _sniff_sep(path: Path, encoding: str) -> str | None:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            head = f.read(64 * 1024)
        dialect = csv.Sniffer().sniff(head, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return None


def read_csv_robust(path: Path, **kw) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kw)
            if df.shape[1] == 1 and "sep" not in kw:
                sep = _sniff_sep(path, enc)
                if sep and sep != ",":
                    df = pd.read_csv(path, encoding=enc, low_memory=False, sep=sep, **kw)
            df.columns = [str(c).replace("﻿", "") for c in df.columns]
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:  # parser errors: try python engine
            last_err = e
            try:
                df = pd.read_csv(path, encoding=enc, engine="python", sep=None, **kw)
                df.columns = [str(c).replace("﻿", "") for c in df.columns]
                return df
            except Exception as e2:
                last_err = e2
                continue
    raise RuntimeError(f"Cannot read {path}: {last_err}")


def read_columns_as_str(path: Path, cols: list[str]) -> pd.DataFrame:
    """Re-read selected columns as raw strings (preserves '001', '1' vs '1.0', 'NA' labels)."""
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, low_memory=False)
            df.columns = [str(c).replace("﻿", "") for c in df.columns]
            keep = [c for c in cols if c in df.columns]
            return df[keep]
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    df = read_csv_robust(path, dtype=str, keep_default_na=False)
    return df[[c for c in cols if c in df.columns]]


DATA_EXT = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".csv.gz", ".zip")
SKIP_DIRS = {".agent_work", "__pycache__", ".git", ".venv", "venv", "node_modules", "catboost_info", "autogluon", "AutogluonModels"}


def _is_data(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and any(n.endswith(e) for e in DATA_EXT) and n != "requirements.txt" and n != "submission.csv"


def _walk(root: Path, max_depth: int) -> list:
    out = []
    try:
        root = Path(root)
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = sorted(d.iterdir(), key=lambda q: q.name)
            except Exception:
                continue
            for e in entries:
                if e.is_dir():
                    if depth < max_depth and e.name not in SKIP_DIRS and not e.name.startswith("."):
                        stack.append((e, depth + 1))
                elif _is_data(e):
                    out.append(e)
    except Exception:
        pass
    return out


OWN_OUTPUTS = {"submission.csv", "submission.tmp", "submission_fallback.csv"}


def _role(p: Path) -> str | None:
    n = p.name.lower()
    stem = n.split(".")[0]
    if n in OWN_OUTPUTS or n.startswith("submission.csv"):
        return None  # our own output (possibly from an earlier run), never an input
    if "sample" in n or "submission" in n:
        return "sample"
    if any(k in n for k in ("dictionary", "data_dict", "description", "schema")) or stem in ("dict", "columns", "metadata"):
        return "dict"
    if "train" in n:
        return "train"
    if "test" in n:
        return "test"
    return None


def _rank(p: Path, role: str, base: Path, mentioned: set) -> tuple:
    exact = {"train": "train.csv", "test": "test.csv", "sample": "sample_submission.csv", "dict": "data_dictionary.csv"}[role]
    try:
        rel_depth = len(p.resolve().relative_to(base.resolve()).parts)
    except Exception:
        rel_depth = 50
    return (0 if p.resolve() in mentioned else 1, 0 if p.name.lower() == exact else 1, rel_depth, len(p.name), str(p))


def _mentioned_paths(task_text: str, task_dir: Path) -> set:
    """Data paths written in task.md (e.g. `data/train.csv`), resolved relative to the task.md folder."""
    out = set()
    for m in re.finditer(r"[\w./\\-]+\.(?:csv|tsv|parquet|txt)", task_text or "", re.IGNORECASE):
        raw = m.group(0).strip("`'\"()[] ").replace("\\", "/")
        for cand in (task_dir / raw, task_dir / raw.lstrip("./")):
            try:
                if cand.is_file() and cand.name.lower() not in OWN_OUTPUTS:
                    out.add(cand.resolve())
            except Exception:
                pass
    return out


def locate_files(task_md: Path, task_dir: Path, task_text: str = "") -> dict:
    """Find train/test/sample/dict relative to task.md: mentioned paths, the task folder, its subfolders,
    then the parent folder tree. train and test are taken from the same folder whenever possible."""
    task_dir = Path(task_dir).resolve()
    mentioned = _mentioned_paths(task_text, task_dir)
    searches = [(task_dir, 0), (task_dir, 4), (task_dir.parent, 4), (task_dir.parent.parent, 3)]
    for base, depth in searches:
        files = list(mentioned) + _walk(base, depth)
        files = list(dict.fromkeys(Path(f).resolve() for f in files))
        by_role = {"train": [], "test": [], "sample": [], "dict": []}
        for f in files:
            r = _role(f)
            if r:
                by_role[r].append(f)
        if not by_role["train"] or not by_role["test"]:
            continue
        # best (train, test) pair: same directory first
        best = None
        for tr in by_role["train"]:
            for te in by_role["test"]:
                key = (0 if tr.parent == te.parent else 1,) + _rank(tr, "train", task_dir, mentioned)[:3] + _rank(te, "test", task_dir, mentioned)[:3]
                if best is None or key < best[0]:
                    best = (key, tr, te)
        _, tr, te = best
        data_dir = tr.parent

        def pick(role):
            c = by_role[role]
            if not c:
                return None
            return sorted(c, key=lambda p: (0 if p.parent == data_dir else 1,) + _rank(p, role, task_dir, mentioned))[0]
        res = {"train": tr, "test": te, "sample": pick("sample"), "dict": pick("dict"), "data_dir": data_dir,
               "all": [f for f in files if f.parent == data_dir]}
        if data_dir != task_dir:
            LOG.info(f"Data files found in {data_dir} (task.md is in {task_dir})")
        return res
    listing = [str(p.relative_to(task_dir)) if str(p).startswith(str(task_dir)) else str(p) for p in _walk(task_dir, 4)][:50]
    raise FileNotFoundError(f"train/test files not found near {task_dir}; data files seen: {listing}")


def load_task(task_md: Path, task_dir: Path) -> RawData:
    try:
        task_text = Path(task_md).read_text(encoding="utf-8", errors="replace")
    except Exception:
        task_text = ""
    loc = locate_files(task_md, task_dir, task_text)
    train_p, test_p, sample_p, dict_p = loc["train"], loc["test"], loc["sample"], loc["dict"]
    files = loc["all"]
    LOG.info(f"Files: train={train_p}, test={test_p}, sample={sample_p}, dict={dict_p}")

    def _read(p):
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        if p.name.lower().endswith(".tsv"):
            return read_csv_robust(p, sep="\t")
        return read_csv_robust(p)

    train = _read(train_p)
    test = _read(test_p)
    sample = None
    if sample_p is not None:
        try:
            sample = read_csv_robust(sample_p, dtype=str, keep_default_na=False)
        except Exception as e:
            LOG.warning(f"sample_submission unreadable: {e}")
    ddict = None
    if dict_p is not None:
        try:
            ddict = read_csv_robust(dict_p)
        except Exception as e:
            LOG.warning(f"data dictionary unreadable: {e}")
    used = {train_p, test_p, sample_p, dict_p}
    extra = [f.name for f in files if f not in used]
    LOG.info(f"Loaded train {train.shape}, test {test.shape}, sample {None if sample is None else sample.shape}, "
             f"dict {None if ddict is None else ddict.shape}, extra files: {extra}")
    return RawData(task_text, train, test, sample, ddict, train_p, test_p, sample_p, dict_p, extra)


def parse_dictionary(ddict: pd.DataFrame | None) -> dict:
    """Return {column_name: {role, dtype, unit, description}} from a data dictionary of unknown layout."""
    if ddict is None or ddict.empty:
        return {}
    cols = [str(c) for c in ddict.columns]
    low = [c.lower() for c in cols]

    def pick(pats):
        for p in pats:
            for c, l in zip(cols, low):
                if re.search(p, l):
                    return c
        return None

    name_c = pick([r"^feature$", r"^column", r"^name$", r"feature", r"col", r"name", r"field", r"признак", r"поле"]) or cols[0]
    role_c = pick([r"role", r"роль", r"type_of", r"kind"])
    dtype_c = pick([r"dtype", r"data_?type", r"^type$", r"тип"])
    unit_c = pick([r"unit", r"единиц", r"format", r"формат"])
    desc_c = pick([r"description_ru", r"desc", r"описан", r"meaning", r"comment"])
    out = {}
    for _, r in ddict.iterrows():
        name = str(r.get(name_c, "")).strip()
        if not name or name.lower() == "nan":
            continue
        rec = {}
        for key, c in (("role", role_c), ("dtype", dtype_c), ("unit", unit_c), ("description", desc_c)):
            if c is not None and c != name_c:
                v = r.get(c)
                if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
                    rec[key] = str(v).strip()
        out[name] = rec
    return out
