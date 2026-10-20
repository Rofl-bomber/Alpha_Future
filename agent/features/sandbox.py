"""Исполнение make_features(df) от Инженера в отдельном процессе с таймаутом.

Проверки: число строк не изменилось, нет inf, нет полностью пустых колонок,
не больше max_new_cols новых колонок. Ошибки возвращаются текстом (traceback) — их
оркестратор отправляет Инженеру на исправление.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_CHILD = r'''
import sys, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
code_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    df = pd.read_pickle(in_path)
    ns = {"np": np, "pd": pd, "__name__": "fe_module"}
    src = open(code_path, encoding="utf-8").read()
    exec(compile(src, "make_features.py", "exec"), ns)
    if "make_features" not in ns:
        raise NameError("в коде нет функции make_features(df)")
    res = ns["make_features"](df.copy())
    if isinstance(res, pd.Series):
        res = res.to_frame()
    if not isinstance(res, pd.DataFrame):
        raise TypeError(f"make_features должна вернуть pandas.DataFrame, получено {type(res).__name__}")
    res.to_pickle(out_path)
except BaseException:
    sys.stderr.write(traceback.format_exc(limit=8))
    sys.exit(3)
'''


@dataclass
class SandboxResult:
    ok: bool
    frame: pd.DataFrame | None = None   # только новые колонки, длина = len(df)
    error: str | None = None
    stage: str = ""                     # exec | timeout | check
    stdout: str = ""


def _clean_name(c: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(c)).strip("_")
    return s or "col"


def run_make_features(code: str, df: pd.DataFrame, work_dir: Path, tag: str, timeout: int = 120,
                      max_new_cols: int = 20) -> SandboxResult:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / f"{tag}.py"
    in_path = work_dir / "_input.pkl"
    out_path = work_dir / f"{tag}_out.pkl"
    child_path = work_dir / "_child.py"
    code_path.write_text(code, encoding="utf-8")
    if not in_path.exists():
        df.to_pickle(in_path)
    child_path.write_text(_CHILD, encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        proc = subprocess.run([sys.executable, "-I", str(child_path), str(code_path), str(in_path), str(out_path)],
                              capture_output=True, text=True, timeout=timeout, env=env, cwd=str(work_dir))
    except subprocess.TimeoutExpired:
        return SandboxResult(False, error=f"Превышен таймаут {timeout} с", stage="timeout")
    if proc.returncode != 0 or not out_path.exists():
        return SandboxResult(False, error=(proc.stderr or "процесс завершился без результата")[-3000:],
                             stage="exec", stdout=proc.stdout[-1000:])
    res = pd.read_pickle(out_path)
    out_path.unlink(missing_ok=True)

    # --- проверки
    if len(res) != len(df):
        return SandboxResult(False, error=f"Число строк изменилось: было {len(df)}, стало {len(res)}. "
                                          "Нельзя удалять, добавлять или переупорядочивать строки.", stage="check")
    res = res.reset_index(drop=True)
    new_cols = [c for c in res.columns if c not in df.columns]
    if not new_cols:
        # функция могла вернуть изменённые исходные колонки — считаем это ошибкой
        return SandboxResult(False, error="make_features не добавила ни одной новой колонки "
                                          "(верни df с новыми колонками; имена не должны совпадать с исходными).",
                             stage="check")
    if len(new_cols) > max_new_cols:
        return SandboxResult(False, error=f"Слишком много новых колонок: {len(new_cols)} > {max_new_cols}.",
                             stage="check")
    out = res[new_cols].copy()
    problems = []
    for c in new_cols:
        s = out[c]
        if isinstance(s, pd.DataFrame):
            problems.append(f"{c}: дублирующееся имя колонки")
            continue
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("int8")
            continue
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype(float).values
            if np.isinf(v).any():
                problems.append(f"{c}: есть inf ({int(np.isinf(v).sum())} значений) — обработай деление на ноль")
            if np.isnan(v).all():
                problems.append(f"{c}: колонка полностью пустая (все NaN)")
        else:
            if s.isna().all():
                problems.append(f"{c}: колонка полностью пустая")
            out[c] = s.astype("string").astype(object).where(s.notna(), np.nan)
    if problems:
        return SandboxResult(False, error="Проверки не пройдены:\n" + "\n".join(problems), stage="check")
    out.columns = [_clean_name(c) for c in out.columns]
    if out.columns.duplicated().any():
        out.columns = [f"{c}_{i}" for i, c in enumerate(out.columns)]
    return SandboxResult(True, frame=out, stdout=proc.stdout[-1000:])


class FeatureRegistry:
    """Зарегистрированные признаки fe_XXX: колонки на train+test и метаданные."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "features"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}
        self.meta: dict[str, dict] = {}
        self._n = 0

    def next_id(self) -> str:
        self._n += 1
        return f"fe_{self._n:03d}"

    def register(self, fe_id: str, frame: pd.DataFrame, code: str, spec_text: str, alias: str = "",
                 hypothesis_id: str = "") -> list[str]:
        frame = frame.copy()
        frame.columns = [f"{fe_id}__{c}" for c in frame.columns]
        self._frames[fe_id] = frame
        self.meta[fe_id] = {"columns": list(frame.columns), "feature_spec": spec_text, "alias": alias,
                            "hypothesis": hypothesis_id, "status": "registered"}
        (self.dir / f"{fe_id}.py").write_text(code, encoding="utf-8")
        frame.to_pickle(self.dir / f"{fe_id}.pkl")
        return list(frame.columns)

    def __contains__(self, fe_id: str) -> bool:
        return fe_id in self._frames

    def frame(self, fe_id: str) -> pd.DataFrame:
        return self._frames[fe_id]

    def code(self, fe_id: str) -> str:
        p = self.dir / f"{fe_id}.py"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def resolve(self, name: str) -> str | None:
        if name in self._frames:
            return name
        for k, m in self.meta.items():
            if m.get("alias") and m["alias"] == name:
                return k
        return None

    def menu(self) -> str:
        if not self.meta:
            return "пока нет"
        return "\n".join(f"- {k} ({m['alias'] or '-'}): {m['feature_spec'][:120]} | колонки: "
                         f"{', '.join(m['columns'])[:160]} | статус: {m['status']}" for k, m in self.meta.items())
