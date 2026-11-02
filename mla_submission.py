"""Build, validate and atomically write submission.csv in the exact sample_submission format."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from mla_io_data import RawData, read_columns_as_str
from mla_postprocess import Decision
from mla_task_spec import TaskSpec
from mla_utils import LOG


class SubmissionWriter:
    def __init__(self, raw: RawData, spec: TaskSpec, out_path: Path):
        self.raw = raw
        self.spec = spec
        self.out_path = Path(out_path)
        self.sample = raw.sample
        self.id_col = spec.id_col
        n_test = len(raw.test)
        # test ids as raw strings
        self.test_ids = None
        if self.id_col and self.id_col in raw.test.columns:
            try:
                s = read_columns_as_str(raw.test_path, [self.id_col])[self.id_col]
                if len(s) == n_test:
                    self.test_ids = s.astype(str).str.strip().values
            except Exception:
                pass
            if self.test_ids is None:
                self.test_ids = raw.test[self.id_col].astype(str).values
        # row mapping: output row -> test row
        if self.sample is not None and self.id_col and self.id_col in self.sample.columns and self.test_ids is not None:
            pos = {v: i for i, v in enumerate(self.test_ids)}
            sid = self.sample[self.id_col].astype(str).str.strip().values
            self.rows = np.array([pos.get(v, -1) for v in sid])
            if (self.rows < 0).mean() > 0.5:
                # ids do not match as strings (e.g. 1 vs 1.0) -> try numeric matching, else positional
                try:
                    tnum = pd.to_numeric(pd.Series(self.test_ids), errors="coerce").values
                    snum = pd.to_numeric(pd.Series(sid), errors="coerce").values
                    posn = {v: i for i, v in enumerate(tnum)}
                    rows = np.array([posn.get(v, -1) for v in snum])
                    if (rows >= 0).mean() > 0.9:
                        self.rows = rows
                    elif len(sid) == n_test:
                        self.rows = np.arange(n_test)
                except Exception:
                    if len(sid) == n_test:
                        self.rows = np.arange(n_test)
            if (self.rows < 0).any():
                LOG.warning(f"{(self.rows < 0).sum()} sample ids not found in test; they get fallback predictions")
        else:
            self.rows = np.arange(n_test)

    # -----------------------------------------------------------------------------------------
    def _frame(self) -> pd.DataFrame:
        if self.sample is not None:
            return self.sample[[c for c in self.sample.columns]].copy()
        cols = {}
        if self.id_col:
            cols[self.id_col] = self.test_ids if self.test_ids is not None else np.arange(len(self.raw.test))
        df = pd.DataFrame(cols)
        for t in self.spec.targets:
            for c in t.sub_columns:
                df[c] = np.nan
        return df

    def build(self, preds: dict, decisions: dict, fallback: dict | None = None) -> pd.DataFrame:
        """preds: target name -> internal test prediction (n_test,) or (n_test, K)."""
        df = self._frame()
        ok_rows = self.rows >= 0
        rows = np.where(ok_rows, self.rows, 0)
        for t in self.spec.targets:
            P = preds.get(t.name)
            dec = decisions.get(t.name) or Decision()
            if P is None:
                raise ValueError(f"no predictions for {t.name}")
            P = np.asarray(P)
            if t.pred_format == "value":
                v = dec.apply_regression(np.asarray(P, float))[rows]
                if not ok_rows.all():
                    v[~ok_rows] = np.nanmean(v[ok_rows]) if ok_rows.any() else 0.0
                v = np.where(np.isfinite(v), v, np.nanmean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
                df[t.sub_columns[0]] = v
            elif t.pred_format == "label":
                lab = dec.labels(P)[rows]
                if not ok_rows.all():
                    major = np.bincount(lab[ok_rows]).argmax() if ok_rows.any() else 0
                    lab[~ok_rows] = major
                df[t.sub_columns[0]] = [t.out_tokens[i] for i in lab]
            elif t.pred_format == "proba":
                Q = dec.transform_proba(P)
                v = Q[:, 1][rows]
                v = np.clip(np.nan_to_num(v, nan=0.5), 0.0, 1.0)
                df[t.sub_columns[0]] = v
            elif t.pred_format == "proba_matrix":
                Q = dec.transform_proba(P)[rows]
                for col in t.sub_columns:
                    k = (t.class_columns or {}).get(col)
                    df[col] = Q[:, k] if k is not None else 1.0 / max(1, t.n_classes)
            else:
                raise ValueError(t.pred_format)
        return df

    def validate(self, df: pd.DataFrame) -> list[str]:
        errs = []
        if self.sample is not None:
            if list(df.columns) != list(self.sample.columns):
                errs.append("columns differ from sample_submission")
            if len(df) != len(self.sample):
                errs.append("row count differs from sample_submission")
        elif len(df) != len(self.raw.test):
            errs.append("row count differs from test")
        if df.isna().any().any():
            errs.append("NaN values present")
        return errs

    def write(self, df: pd.DataFrame, path: Path | None = None) -> bool:
        path = Path(path or self.out_path)
        errs = self.validate(df)
        if errs:
            LOG.error(f"submission validation failed: {errs}")
            return False
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return True

    def fallback(self) -> pd.DataFrame:
        """Prior-based predictions in the right format (mean / majority class / class priors)."""
        preds, decs = {}, {}
        n = len(self.raw.test)
        for t in self.spec.targets:
            y = self.raw.train[t.name]
            if t.task_type == "regression":
                v = pd.to_numeric(y, errors="coerce")
                center = float(v.median()) if self.spec.metric in ("mae", "medae", "mape") else float(v.mean())
                preds[t.name] = np.full(n, center if np.isfinite(center) else 0.0)
            else:
                from mla_task_spec import canon_label
                canon = y.astype(str).map(canon_label)
                pri = np.array([(canon == c).mean() for c in t.classes], float)
                pri = pri / pri.sum() if pri.sum() > 0 else np.full(len(t.classes), 1.0 / len(t.classes))
                preds[t.name] = np.tile(pri, (n, 1))
            decs[t.name] = Decision()
        return self.build(preds, decs)


def emergency_submission(task_dir: Path) -> bool:
    """Last-resort: copy sample_submission (or create a minimal file) so that *something* valid exists."""
    try:
        task_dir = Path(task_dir)
        out = task_dir / "submission.csv"
        if out.exists() and out.stat().st_size > 0:
            return True
        from mla_io_data import _walk
        cands = [p for p in _walk(task_dir, 4) if "sample" in p.name.lower() and p.suffix.lower() == ".csv"]
        cands.sort(key=lambda p: (len(p.relative_to(task_dir).parts), p.name))
        if cands:
            df = pd.read_csv(cands[0], dtype=str, keep_default_na=False)
            df.to_csv(out, index=False)
            return True
    except Exception as e:
        LOG.error(f"emergency submission failed: {e}")
    return False
