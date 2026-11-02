"""Small shared helpers: logging, JSON, atomic writes, timing."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LOG = logging.getLogger("mlagent")


def setup_logging(work_dir: Path, verbose: bool = True) -> logging.Logger:
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    try:
        fh = logging.FileHandler(Path(work_dir) / "agent.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    except Exception:
        pass
    LOG.propagate = False
    return LOG


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        try:
            return str(o)
        except Exception:
            return None


def dumps(obj, **kw) -> str:
    return json.dumps(obj, cls=NpEncoder, ensure_ascii=False, **kw)


def save_json(path: Path, obj) -> None:
    try:
        atomic_write_text(Path(path), dumps(obj, indent=2))
    except Exception as e:  # never crash on reporting
        LOG.debug(f"save_json failed for {path}: {e}")


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def short_hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]


def truncate(text: str, max_chars: int, tail: int = 0) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    if tail > 0 and tail < max_chars:
        head = max_chars - tail - 30
        return text[:head] + "\n...[truncated]...\n" + text[-tail:]
    return text[: max_chars - 20] + "\n...[truncated]"


def fmt_num(v, digits: int = 5) -> str:
    try:
        if v is None:
            return "NA"
        v = float(v)
        if math.isnan(v):
            return "nan"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e6 or a < 1e-3:
            return f"{v:.{max(1, digits - 2)}e}"
        if a >= 1000:
            return f"{v:.1f}"
        return f"{v:.{digits}g}"
    except Exception:
        return str(v)


@contextmanager
def timer(name: str, level=logging.INFO):
    t0 = time.time()
    LOG.log(level, f"[start] {name}")
    try:
        yield
    finally:
        LOG.log(level, f"[done ] {name} in {time.time() - t0:.1f}s")


def format_exc(limit_lines: int = 40) -> str:
    tb = traceback.format_exc()
    lines = tb.strip().splitlines()
    if len(lines) > limit_lines:
        lines = lines[:5] + ["..."] + lines[-(limit_lines - 6):]
    return "\n".join(lines)


def safe_call(fn, *args, default=None, what: str = "", **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa
        LOG.warning(f"{what or getattr(fn, '__name__', 'call')} failed: {type(e).__name__}: {e}")
        LOG.debug(format_exc())
        return default
