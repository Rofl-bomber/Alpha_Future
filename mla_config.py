"""Run configuration: environment variables, time budget, CPU detection."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on", "y")


def parse_time_budget(value: str | None, default_seconds: float = 3 * 3600) -> float:
    """Parse AGENT_TIME_BUDGET.

    Accepts: '10800', '10800s', '180m', '3h', '2.5h', '02:30:00', '150:00' (mm:ss is ambiguous,
    so 'a:b' is treated as hours:minutes).  A bare number <= 24 is interpreted as hours
    (the documented default is "3 hours"), a larger bare number as seconds.
    """
    if value is None:
        return default_seconds
    s = str(value).strip().lower().replace(",", ".")
    if not s:
        return default_seconds
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2:
                return parts[0] * 3600 + parts[1] * 60
        m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-zа-я]*)", s)
        if not m:
            return default_seconds
        num = float(m.group(1))
        unit = m.group(2)
        if unit.startswith("h") or unit.startswith("ч"):
            return num * 3600
        if unit.startswith("m") or unit.startswith("м"):
            return num * 60
        if unit.startswith("s") or unit.startswith("с"):
            return num
        if unit == "":
            return num * 3600 if num <= 24 else num
        return default_seconds
    except Exception:
        return default_seconds


def detect_cpus() -> int:
    """Real CPU budget of the container (cgroup quota / affinity), not the host core count."""
    candidates = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()[:2]
            if q != "max":
                candidates.append(max(1, int(round(float(q) / float(p)))))
    except Exception:
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            candidates.append(max(1, int(round(q / p))))
    except Exception:
        pass
    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def detect_memory_gb() -> float:
    vals = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                v = f.read().strip()
            if v != "max":
                b = float(v)
                if 0 < b < 1 << 50:
                    vals.append(b / 2**30)
        except Exception:
            pass
    try:
        import psutil
        vals.append(psutil.virtual_memory().total / 2**30)
    except Exception:
        pass
    return min(vals) if vals else 16.0


class Deadline:
    """Wall-clock budget helper. All phases ask it how much time is left."""

    def __init__(self, budget_seconds: float, start: float | None = None, reserve: float = 0.0):
        self.start = start if start is not None else time.time()
        self.budget = float(budget_seconds)
        self.reserve = float(reserve)  # time kept for final writing / safety

    @property
    def end(self) -> float:
        return self.start + self.budget - self.reserve

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.end - time.time()

    def hard_remaining(self) -> float:
        return self.start + self.budget - time.time()

    def frac_elapsed(self) -> float:
        return self.elapsed() / max(1.0, self.budget)

    def until(self, frac: float) -> float:
        """Seconds left until a fraction of the total budget is reached (can be <0)."""
        return self.start + self.budget * frac - time.time()


@dataclass
class Config:
    task_md: Path
    task_dir: Path
    work_dir: Path
    time_budget: float
    start_time: float
    n_jobs: int
    memory_gb: float
    seed: int = 42
    # LLM
    llm_enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    llm_max_calls: int = 200
    llm_max_total_tokens: int = 5_000_000
    llm_max_tokens_per_call: int = 54000
    llm_concurrency: int = 3
    llm_timeout: float = 600.0
    llm_thinking: str = "auto"  # auto|on|off
    llm_mock: bool = False
    # misc
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_env(task_md: str) -> "Config":
        start = time.time()
        task_md_p = Path(task_md).resolve()
        task_dir = task_md_p.parent if task_md_p.suffix else task_md_p
        work_dir = task_dir / ".agent_work"
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            test = work_dir / ".write_test"
            test.write_text("ok")
            test.unlink()
        except Exception:
            work_dir = Path("/tmp") / "agent_work" / task_dir.name
            work_dir.mkdir(parents=True, exist_ok=True)
        budget = parse_time_budget(_env("AGENT_TIME_BUDGET"))
        n_jobs = _env_int("AGENT_N_JOBS", 0) or detect_cpus()
        api_key = _env("OPENAI_API_KEY")
        base_url = _env("OPENAI_BASE_URL") or _env("OPENAI_API_BASE")
        model = _env("OPENAI_MODEL") or _env("OPENAI_MODEL_NAME")
        mock = _env_bool("AGENT_LLM_MOCK", False)
        llm_enabled = (not _env_bool("AGENT_DISABLE_LLM", False)) and (mock or bool(model or base_url or api_key))
        return Config(
            task_md=task_md_p,
            task_dir=task_dir,
            work_dir=work_dir,
            time_budget=budget,
            start_time=start,
            n_jobs=max(1, n_jobs),
            memory_gb=detect_memory_gb(),
            seed=_env_int("AGENT_SEED", 42),
            llm_enabled=llm_enabled,
            api_key=api_key,
            base_url=base_url,
            model=model,
            llm_max_calls=_env_int("AGENT_LLM_MAX_CALLS", 200),
            llm_max_total_tokens=_env_int("AGENT_LLM_MAX_TOTAL_TOKENS", 5_000_000),
            llm_max_tokens_per_call=_env_int("AGENT_LLM_MAX_TOKENS_PER_CALL", 54000),
            llm_concurrency=max(1, _env_int("AGENT_LLM_CONCURRENCY", 3)),
            llm_timeout=_env_float("AGENT_LLM_TIMEOUT", 600.0),
            llm_thinking=(_env("AGENT_LLM_THINKING", "auto") or "auto").lower(),
            llm_mock=mock,
            verbose=_env_bool("AGENT_VERBOSE", True),
            extra={"autogluon": _env_bool("AGENT_USE_AUTOGLUON", True),
                   "role_tokens": {"analyst": _env_int("AGENT_LLM_TOKENS_ANALYST", 54000),
                                   "researcher": _env_int("AGENT_LLM_TOKENS_RESEARCHER", 54000),
                                   "developer": _env_int("AGENT_LLM_TOKENS_DEVELOPER", 54000),
                                   "debugger": _env_int("AGENT_LLM_TOKENS_DEBUGGER", 54000),
                                   "repair": _env_int("AGENT_LLM_TOKENS_REPAIR", 54000)},
                   "researcher_calls": _env_int("AGENT_RESEARCHER_CALLS", 14),
                   "ideas_per_call": _env_int("AGENT_IDEAS_PER_CALL", 5)},
        )
