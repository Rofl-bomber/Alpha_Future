"""Запуск (формат хакатона, как в baseline.py):

    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py путь/к/task.md

Остальные настройки — переменные окружения AGENT_* (см. agent/config.py), например:
    AGENT_ITERS=3        число итераций цикла (по умолчанию 10)
    AGENT_MOCK_LLM=1     подставная LLM без эндпоинта и токенов
    AGENT_SUBSAMPLE=10000  подвыборка train для быстрых прогонов
Результат: submission.csv в каталоге task.md; журнал и лог LLM — в agent_runs/<время>/ рядом.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from agent.config import load_config  # noqa: E402
from agent.orchestrator import Orchestrator  # noqa: E402


def main():
    task = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    cfg = load_config()
    result = Orchestrator(task, cfg).run()
    short = {k: result.get(k) for k in ("choice", "final_score", "l1_score", "best_single", "best_single_score",
                                         "blend_score", "submission", "run_dir", "total_time_s")}
    print("Агент завершил работу:", json.dumps(short, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
