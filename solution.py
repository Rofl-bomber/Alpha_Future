"""Entry point.

    python3 solution.py task.md

Env: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL (Qwen endpoint), AGENT_TIME_BUDGET (default 3h),
optional AGENT_* knobs (see PLAN.md / README.md). Writes submission.csv next to task.md.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_T0 = time.time()
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _set_thread_env():
    try:
        from mla_config import detect_cpus
        n = int(os.environ.get("AGENT_N_JOBS") or detect_cpus())
    except Exception:
        n = os.cpu_count() or 1
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(k, str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def main(argv: list[str]) -> int:
    _set_thread_env()
    task_md = argv[1] if len(argv) > 1 else "task.md"
    if not os.path.exists(task_md):
        cand = Path.cwd() / "task.md"
        task_md = str(cand) if cand.exists() else task_md
    # make the path absolute BEFORE changing directory, otherwise a relative path gets applied twice
    task_md = str(Path(task_md).resolve())
    task_dir = Path(task_md).parent
    try:
        os.chdir(task_dir)
    except Exception:
        pass
    try:
        from mla_config import Config
        from mla_orchestrator import Orchestrator
        from mla_utils import LOG, setup_logging
        cfg = Config.from_env(task_md)
        cfg.start_time = _T0
        setup_logging(cfg.work_dir, cfg.verbose)
        Orchestrator(cfg).run()
    except BaseException as e:  # noqa: BLE001 - never leave without a submission
        print(f"[solution] fatal: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            import traceback
            traceback.print_exc()
            from mla_submission import emergency_submission
            emergency_submission(task_dir)
        except Exception:
            pass
        if isinstance(e, KeyboardInterrupt):
            return 130
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    # Background LLM threads may still be waiting on HTTP responses; do not let interpreter shutdown
    # join them (that could overrun the time budget). The submission is already on disk.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        import logging
        logging.shutdown()
    finally:
        os._exit(code)
