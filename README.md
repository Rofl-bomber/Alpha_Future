# AI-агент для табличных ML-задач

Запуск (как на платформе):

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... AGENT_TIME_BUDGET=3h python3 solution.py task.md
```

Скрипт читает `task.md`, `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv` из папки задачи и пишет
`submission.csv` туда же. Все модули лежат в корне рядом с `solution.py` (префикс `mla_`). Рабочие файлы, лог и отчёт — в `<папка задачи>/.agent_work/`
(`agent.log`, `agent_report.json`, `journal.json`, `llm_calls.jsonl`).

Подробное описание архитектуры и обоснование решений — в `PLAN.md`.

## Структура

| Модуль | Назначение |
|---|---|
| `solution.py` | точка входа, гарантия сабмита при любых сбоях |
| `mla_orchestrator.py` | фазы, бюджет времени, anytime-сабмит, отчёт |
| `mla_task_spec.py` | понимание задачи: эвристики + согласование с LLM Analyst |
| `mla_metrics.py` | реестр 27 метрик и распознавание метрики в тексте (рус/англ) |
| `mla_preprocess.py`, `mla_profiling.py`, `mla_io_data.py` | чтение, профиль, типизация, даты, текст, категории |
| `mla_cv.py` | выбор схемы валидации (random / group / time), adversarial validation |
| `mla_features.py` | состояние признаков, target encoding внутри фолдов, auto-гипотезы |
| `mla_research.py` | цикл AutoResearch: Researcher → Developer → Sandbox → оценка → журнал |
| `mla_llm.py`, `mla_prompts.py` | клиент Qwen (thinking, ретраи, бюджеты) и промпты агентов |
| `mla_sandbox.py` | выполнение кода от LLM в отдельном процессе с таймаутом |
| `mla_models.py`, `mla_models_nn.py`, `mla_models_ag.py` | LightGBM, XGBoost, CatBoost, MLP, ExtraTrees, AutoGluon + CV-раннер |
| `mla_hpo.py`, `mla_modeling.py` | Optuna и очередь моделей с распределением времени |
| `mla_ensemble.py`, `mla_postprocess.py` | hill-climbing, стекинг, пороги/веса классов/калибровка |
| `mla_submission.py` | сборка и проверка сабмита в формате sample_submission |
| `mla_mock_llm.py` | имитация LLM для офлайн-тестов (`AGENT_LLM_MOCK=1`) |

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `AGENT_TIME_BUDGET` | `3h` | `10800`, `10800s`, `180m`, `3h`, `02:30:00`; голое число ≤ 24 — часы, больше — секунды |
| `AGENT_LLM_MAX_CALLS` | 200 | максимум вызовов LLM на задачу |
| `AGENT_LLM_MAX_TOTAL_TOKENS` | 5000000 | лимит токенов (prompt + completion) на задачу |
| `AGENT_LLM_MAX_TOKENS_PER_CALL` | 54000 | лимит ответа одного вызова, включая рассуждение |
| `AGENT_LLM_TOKENS_ANALYST` | 54000 | лимит ответа Analyst (включая рассуждение) |
| `AGENT_LLM_TOKENS_RESEARCHER` | 54000 | лимит ответа Researcher |
| `AGENT_LLM_TOKENS_DEVELOPER` | 54000 | лимит ответа Developer |
| `AGENT_LLM_TOKENS_DEBUGGER` | 54000 | лимит ответа Debugger |
| `AGENT_LLM_TOKENS_REPAIR` | 54000 | лимит повторного запроса при битом JSON |
| `AGENT_RESEARCHER_CALLS` | 14 | максимум вызовов Researcher за задачу |
| `AGENT_IDEAS_PER_CALL` | 5 | гипотез за один вызов Researcher |
| `AGENT_LLM_CONCURRENCY` | 3 | одновременных запросов к LLM |
| `AGENT_LLM_TIMEOUT` | 600 | таймаут одного запроса, с |
| `AGENT_LLM_THINKING` | `auto` | `on` / `off` / `auto` (рассуждение для Analyst/Researcher/Developer) |
| `AGENT_DISABLE_LLM` | 0 | `1` — полностью детерминированный режим |
| `AGENT_LLM_MOCK` | 0 | `1` — имитация LLM (для тестов без API) |
| `AGENT_N_JOBS` | авто | число потоков (по умолчанию — лимит CPU контейнера из cgroup) |
| `AGENT_SEED` | 42 | сид |
| `AGENT_USE_AUTOGLUON` | 1 | `0` — не использовать AutoGluon в ансамбле |

Без доступной LLM система работает в детерминированном режиме: auto-гипотезы признаков, модели, ансамбль.
