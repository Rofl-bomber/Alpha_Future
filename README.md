# AI-агент для ML-задач — архитектура v2

Реализация документа «AI-агент для ML-задач: архитектура v2». Детерминированный харнесс считает
метрику и решает «оставить / откатить»; LLM (Qwen3.8-27B) в четырёх ролях только предлагает.

## Запуск

Как на платформе (интерфейс `baseline.py`):

```bash
OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py path/to/task.md
```

Результат — `submission.csv` в каталоге `task.md`. Рабочие файлы прогона (журнал, логи, OOF,
код Инженера) пишутся во временный каталог; путь печатается в начале и в итоговой строке.

## Архив для платформы

```bash
python bench/pack.py submission.zip
```

В архив попадают только `*.py` из корня: `solution.py` и 30 модулей агента, без папок.
`bench/`, `tests/` и файлы прогона в архив не входят.

| Префикс | Файлы | Назначение |
| --- | --- | --- |
| — | `solution.py` | точка входа, watchdog |
| `core_` | `config`, `timer`, `orchestrator` | бюджеты и фазы, часы и дочерние процессы, конечный автомат |
| `task_` | `spec`, `profiler`, `metrics` | TaskSpec, профиль и детектор сплита, реестр метрик |
| `eval_` | `protocol`, `acceptance` | фолды, selection-holdout, сейф меток; критерий принятия и проверки |
| `search_` | `population`, `scheduler`, `operators` | популяция веток, выбор оператора, патчи/Merge/Restart/библиотека |
| `diag_` | `ablation`, `errors` | абляции блоков, анализ ошибок |
| `llm_` | `client`, `roles`, `engineer`, `prompts` | клиент, Аналитик/Исследователь/Критик, ReAct-Инженер, промпты |
| `feat_` | `builtin`, `sandbox` | операции библиотеки и кодирования; песочница для кода LLM |
| `model_` | `lgbm`, `catboost`, `xgb`, `linear`, `extratrees`, `mlp` | обёртки моделей с единым fit / predict |
| `escape_` | `contract` | аварийный оператор `fit_predict` |
| `exp_` | `runner`, `journal` | эксперимент в отдельном процессе, journal.jsonl |
| `final_` | `ensemble`, `submission` | бленд, постобработка, выбор на holdout; валидатор и атомарная запись |

## Настройки

Любое поле `core_config.Config` переопределяется переменной `AGENT_<ПОЛЕ>`:

- `AGENT_TIME_BUDGET` — бюджет в секундах от старта `solution.py` (по умолчанию 10800; для стенда, например, 1800);
- `AGENT_UNLIMITED=1` — режим без внутренних ограничений: время ограничивают только `AGENT_TIME_BUDGET`
  и границы фаз (каждое действие — эксперимент, абляция, сиды, Optuna, диагностика, вызов LLM — может идти
  до конца своей фазы), `max_tokens` в запросы к LLM не передаётся, лимитов токенов на прогон нет;
- `AGENT_PHASE_SPEC`, `AGENT_PHASE_L1`, `AGENT_PHASE_DIAG`, `AGENT_PHASE_SEARCH`, `AGENT_PHASE_RESEED`,
  `AGENT_PHASE_FINAL` — границы фаз как доли бюджета, строго по возрастанию
  (по умолчанию 5/180, 20/180, 30/180, 140/180, 155/180, 178/180);
- `AGENT_TUNE_TRIALS` — максимум проб Optuna за один запуск Tune (по умолчанию 40);
- `AGENT_WORKDIR` — каталог прогона;
- `AGENT_TOKEN_LIMIT_TOTAL`, `AGENT_TOKEN_LIMIT_OUT` — лимит токенов LLM на одну задачу (без `AGENT_UNLIMITED`);
- `AGENT_RESEARCHER_MAX_TOKENS`, `AGENT_ENGINEER_MAX_TOKENS` — лимит длины ответа (без `AGENT_UNLIMITED`);
- `AGENT_N_JOBS` — потоки (по умолчанию из cgroup контейнера);
- `AGENT_LLM_CACHE` — каталог кэша ответов LLM (только для разработки).

Если ответ LLM обрезан по длине (`finish_reason = "length"`) или пуст, делается один повтор без
размышлений. Если обучение модели остановлено по дедлайну, прогон помечается как обрезанный: в журнале и
логе есть CV по пройденным фолдам и дельта к родителю, но в решениях он не участвует.

## Файлы прогона

- `run.log` — ход прогона; `journal.jsonl` — все эксперименты; `final.json` — итог и расход токенов;
- `llm_log.jsonl` — вызовы LLM, включая `finish_reason` и объём размышлений;
- `logs/<задание>.log` — ход каждого дочернего процесса (фолды, время) и итоговая сводка;
  `jobs/<задание>.result.json` — его результат.

## Стенд

```bash
# задача с известными ответами из train исходной
python bench/make_task.py regression/task.md /tmp/bench_reg --group Year_Factor
# прогон и оценка
uv run solution.py /tmp/bench_reg/task.md
python bench/score.py /tmp/bench_reg --baseline 50.0
# проверка харнесса без настоящей LLM
python tests/fake_llm_server.py 8765 &
OPENAI_API_KEY=x OPENAI_MODEL=fake OPENAI_BASE_URL=http://127.0.0.1:8765/v1 AGENT_TIME_BUDGET=900 \
  uv run solution.py path/to/task.md
```
