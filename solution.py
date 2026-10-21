"""Точка входа AI-агента для ML-задач. Запуск как в baseline.py:

    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py task.md

Результат — submission.csv (или файл из условия) в каталоге task.md. Все модули агента
лежат рядом с этим файлом в корне архива; рабочие файлы прогона пишутся во временный
каталог (переопределяется AGENT_WORKDIR). Отсчёт времени — от старта этого файла.
"""

import time

T0 = time.time()

import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if sys.argv[1:] == ["--mcp-server"]:
        return  # интерфейс baseline.py сохранён; MCP-сервер агенту не нужен
    task = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    warnings.filterwarnings("ignore")
    log = logging.getLogger("agent")
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.propagate = False

    import core_config
    import core_timer

    cfg = core_config.load_config()
    timer = core_timer.Timer(T0, cfg.time_budget, cfg.phases)
    registry = core_timer.ProcessRegistry()
    log.info("старт: задача %s, бюджет %.0f с, CPU=%d, память %.1f GB, рабочий каталог %s", task, cfg.time_budget,
             cfg.n_jobs, cfg.mem_limit_gb, cfg.workdir)

    def on_deadline():
        print("Агент остановлен по дедлайну; на диске лучший записанный сабмит.", flush=True)

    watchdog = core_timer.Watchdog(timer, registry, on_deadline)
    watchdog.start()
    result = {}
    try:
        from core_orchestrator import Orchestrator

        result = Orchestrator(task, cfg, timer, registry).run()
    except Exception as e:  # noqa: BLE001
        log.exception("агент упал: %s", e)
        result = {"error": str(e)}
    finally:
        watchdog.stop()
        registry.kill_all()
    short = {k: result.get(k) for k in ("choice", "l1_cv", "best_cv", "submission", "elapsed_s", "workdir", "error")
             if result.get(k) is not None}
    print("Агент завершил работу:", json.dumps(short, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
