"""Run untrusted feature code in a subprocess with a timeout and static checks."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mla_features import FeatureState
from mla_utils import LOG, short_hash

FORBIDDEN = [
    (r"\b(exec|eval|compile|__import__|input|breakpoint|globals|vars)\s*\(", "dynamic code execution / input"),
    (r"\bopen\s*\(", "file access"),
    (r"^\s*(import|from)\s+(os|sys|subprocess|socket|shutil|requests|urllib|http|pickle|ctypes|multiprocessing|threading|pathlib|glob|signal|resource|importlib|builtins)\b", "forbidden module"),
    (r"\.to_(csv|parquet|pickle|excel|json|feather|sql|hdf)\s*\(", "writing files"),
    (r"\bread_(csv|parquet|pickle|excel|json|sql|feather|hdf|table)\s*\(", "reading files"),
    (r"\bos\.|\bsys\.|\bsubprocess\.", "system access"),
]

RUNNER = Path(__file__).with_name("mla_sandbox_runner.py")


def static_check(code: str) -> str | None:
    for rx, why in FORBIDDEN:
        if re.search(rx, code, re.MULTILINE):
            return f"Static check failed: {why} is not allowed (pattern {rx!r}). Remove it; use only numpy/pandas on `df`."
    if not re.search(r"def\s+transform\s*\(", code):
        return "Static check failed: the code must define `def transform(df):`."
    return None


class Sandbox:
    def __init__(self, work_dir: Path, n_jobs: int = 2):
        self.dir = Path(work_dir) / "sandbox"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self._frame_cache: dict = {}
        # run the runner from an isolated directory so that package modules cannot shadow library imports
        self.runner = self.dir / "fe_runner.py"
        self.runner.write_text(RUNNER.read_text(encoding="utf-8"), encoding="utf-8")

    def _input_path(self, state: FeatureState) -> Path:
        key = state.key()
        p = self.dir / f"frame_{key}.parquet"
        if not p.exists():
            df = state.X.copy()
            for c in state.cat_cols:
                df[c] = df[c].astype("object")
            df.to_parquet(p, index=False)
            # keep at most a few frames on disk
            frames = sorted(self.dir.glob("frame_*.parquet"), key=lambda q: q.stat().st_mtime)
            for old in frames[:-4]:
                try:
                    old.unlink()
                except Exception:
                    pass
        return p

    def run(self, state: FeatureState, code: str, timeout: float = 120.0) -> tuple[pd.DataFrame | None, dict]:
        """Returns (new full frame or None, meta). meta has ok/error/traceback/changed/removed."""
        err = static_check(code)
        if err:
            return None, {"ok": False, "error": err, "traceback": ""}
        tag = short_hash(code, time.time())
        code_p = self.dir / f"code_{tag}.py"
        out_p = self.dir / f"out_{tag}.parquet"
        meta_p = self.dir / f"meta_{tag}.json"
        code_p.write_text(code, encoding="utf-8")
        in_p = self._input_path(state)
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
            env[k] = str(max(1, self.n_jobs))
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, str(self.runner), str(in_p), str(code_p), str(out_p), str(meta_p)],
                                  capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return None, {"ok": False, "error": f"Timeout: the function did not finish in {timeout:.0f}s. Vectorise it (no Python loops over rows, no apply with lambdas over rows).", "traceback": ""}
        finally:
            try:
                code_p.unlink()
            except Exception:
                pass
        if not meta_p.exists():
            tail = (proc.stderr or "")[-2000:]
            return None, {"ok": False, "error": f"process crashed (exit {proc.returncode})", "traceback": tail}
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        meta_p.unlink(missing_ok=True)
        meta["seconds"] = round(time.time() - t0, 2)
        if not meta.get("ok"):
            return None, meta
        try:
            out = pd.read_parquet(out_p)
        finally:
            out_p.unlink(missing_ok=True)
        new_X = state.X.copy()
        for c in meta.get("removed", []):
            if c in new_X.columns:
                new_X = new_X.drop(columns=[c])
        for c in out.columns:
            s = out[c]
            if s.dtype == object:
                s = s.where(s.notna(), np.nan)
            new_X[c] = s.values
        return new_X, meta
