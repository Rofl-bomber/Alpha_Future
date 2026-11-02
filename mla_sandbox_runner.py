"""Executed in a separate process: applies LLM-written `transform(df)` to the feature frame.

argv: in_parquet code_file out_parquet meta_json
Writes only new/modified columns to out_parquet and a JSON status to meta_json.
"""
import json
import sys
import traceback


def main():
    in_p, code_p, out_p, meta_p = sys.argv[1:5]
    meta = {"ok": False}
    try:
        import numpy as np
        import pandas as pd
        df = pd.read_parquet(in_p)
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object or str(df[c].dtype) in ("string", "large_string"):
                s = df[c].astype("object")
                df[c] = s.where(s.notna(), np.nan)
        n = len(df)
        orig_cols = list(df.columns)
        snapshot = {c: df[c] for c in orig_cols}
        code = open(code_p, encoding="utf-8").read()
        ns = {"np": np, "pd": pd, "__name__": "feature_block"}
        exec(compile(code, "<feature_block>", "exec"), ns)
        if "transform" not in ns or not callable(ns["transform"]):
            raise RuntimeError("the code must define a function `transform(df)`")
        res = ns["transform"](df.copy())
        if isinstance(res, pd.Series):
            res = res.to_frame()
        if not isinstance(res, pd.DataFrame):
            raise TypeError(f"transform must return a pandas DataFrame, got {type(res).__name__}")
        if len(res) != n:
            raise ValueError(f"transform changed the number of rows: {n} -> {len(res)} (rows must be kept as they are)")
        res = res.reset_index(drop=True)
        if res.columns.duplicated().any():
            res = res.loc[:, ~res.columns.duplicated()]
        res.columns = [str(c) for c in res.columns]
        removed = [c for c in orig_cols if c not in res.columns]
        changed = []
        for c in res.columns:
            s = res[c]
            if isinstance(s, pd.DataFrame):
                raise ValueError(f"column {c} is not 1-D")
            if c in snapshot:
                try:
                    if snapshot[c].reset_index(drop=True).equals(s):
                        continue
                except Exception:
                    pass
            changed.append(c)
        out = {}
        warnings = []
        for c in changed:
            s = res[c]
            if pd.api.types.is_bool_dtype(s):
                s = s.astype(float)
            if pd.api.types.is_numeric_dtype(s):
                s = s.astype(float).replace([np.inf, -np.inf], np.nan)
                if s.isna().all():
                    warnings.append(f"{c}: all values NaN")
            elif pd.api.types.is_datetime64_any_dtype(s):
                s = (s.astype("int64") / 86400e9).where(s.notna(), np.nan)
            else:
                s = s.astype("object")
                s = s.where(s.isna(), s.astype(str))
            out[c] = s.values
        out_df = pd.DataFrame(out)
        if out_df.shape[1] > 400:
            raise ValueError(f"too many new/modified columns: {out_df.shape[1]}")
        out_df.to_parquet(out_p, index=False)
        meta = {"ok": True, "changed": changed, "removed": removed, "warnings": warnings}
    except Exception as e:
        tb = traceback.format_exc()
        lines = [l for l in tb.splitlines()]
        meta = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(lines[-25:])}
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
