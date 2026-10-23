# AI-агент для ML-задач — v3: анализ данных → L1 → поиск → финалисты

Детерминированный харнесс считает метрику и решает «оставить / откатить»; LLM (Qwen3.8-27B)
в ролях Аналитика, Исследователя и Инженера только предлагает. Критик — заглушка.

## Запуск

Как на платформе (интерфейс `baseline.py`):

```bash
OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py path/to/task.md
```

Результат — `submission.csv` в каталоге `task.md`. Рабочие файлы прогона пишутся во временный
каталог; путь печатается в начале и в итоговой строке.

## Этапы

| Этап | Время по умолчанию | Что происходит |
| --- | --- | --- |
| 0. TaskSpec | 0–3 мин | Правила + один JSON-вызов LLM, константный сабмит, статистика харнесса |
| 1. Анализ данных | до 45 мин (потолок) | Аналитик пишет код в Python-сессии, фиксирует разбиение (`submit_split`), сдаёт отчёт A + B; в фоне — страховочный L1 |
| 2. L1 | сразу после анализа | LightGBM на фолдах Аналитика — корень цепочки поиска |
| 3. Поиск | до 140 мин | Preprocessing (15%) → Features (60%) → Model (25%); Исследователь с инструментами, Инженер пишет код `transform`; повторный вызов Аналитика после подэтапов и на плато |
| 4. Финалисты | 140–178 мин | Сиды, бленд, выбор на selection-holdout, refit |

Если LLM недоступна: запасная схема валидации профайлера, L1 и библиотека гипотез по подэтапам.

## Архив для платформы

```bash
python bench/pack.py submission.zip
```

В архив попадают только `*.py` из корня: `solution.py` и модули агента, без папок.

| Префикс | Файлы | Назначение |
| --- | --- | --- |
| — | `solution.py` | точка входа, watchdog |
| `core_` | `config`, `timer`, `orchestrator` | бюджеты и фазы, часы и дочерние процессы, этапы и решения keep/discard |
| `task_` | `spec`, `profiler`, `metrics` | TaskSpec; статистика датасета и запасная схема сплита; реестр метрик |
| `eval_` | `protocol`, `acceptance` | разбиение Аналитика и запасное, сейф меток; критерий принятия |
| `search_` | `population`, `scheduler`, `operators`, `registry` | цепочка текущего решения; подэтапы; патчи и библиотека; отчёт Аналитика и реестр гипотез |
| `llm_` | `client`, `roles`, `analyst`, `researcher`, `engineer`, `prompts` | клиент; TaskSpec и Критик; Аналитик; Исследователь; Инженер; промпты |
| `exp_` | `runner`, `kernel`, `journal` | эксперимент в отдельном процессе; Python-сессия; journal.jsonl |
| `feat_` | `transform`, `sandbox`, `builtin` | цепочка `transform`; песочница кода Инженера; операции библиотеки и кодирования |
| `model_` | `lgbm`, `catboost`, `xgb`, `linear`, `extratrees`, `mlp` | обёртки моделей |
| `escape_` | `contract` | весь пайплайн от Инженера (`fit_predict`) |
| `final_` | `ensemble`, `submission` | бленд, постобработка, выбор на holdout; запись сабмита |

## Настройки

Любое поле `core_config.Config` переопределяется переменной `AGENT_<ПОЛЕ>`, например:

- `AGENT_TIME_BUDGET` — бюджет в секундах от старта `solution.py` (по умолчанию 10800);
- `AGENT_PHASE_SPEC`, `AGENT_PHASE_ANALYSIS`, `AGENT_PHASE_L1`, `AGENT_PHASE_SEARCH`, `AGENT_PHASE_RESEED`,
  `AGENT_PHASE_FINAL` — границы фаз как доли бюджета;
- `AGENT_STAGE_PRE_FRAC`, `AGENT_STAGE_FEAT_FRAC`, `AGENT_STAGE_MODEL_FRAC`, `AGENT_PLATEAU_N` — подэтапы поиска;
- `AGENT_ANALYSIS_STEPS`, `AGENT_ANALYSIS_STEP_TIMEOUT_S`, `AGENT_RECALL_STEPS`, `AGENT_RECALL_MINUTES` — Аналитик;
- `AGENT_RESEARCHER_TOOL_STEPS`, `AGENT_ENGINEER_ATTEMPTS` — Исследователь и Инженер;
- `AGENT_CRITIC_ENABLED=1` — включить настоящего Критика;
- `AGENT_TOKEN_LIMIT_TOTAL`, `AGENT_TOKEN_LIMIT_ANALYST` — лимиты токенов (у Аналитика свой);
- `AGENT_UNLIMITED=1` — без внутренних ограничений: только бюджет и границы фаз.

## Файлы прогона

- `run.log`, `journal.jsonl` (подэтап и ссылка на гипотезу Аналитика у каждого эксперимента), `final.json`;
- `stats.txt` — статистика харнесса; `analyst_report.md`, `analyst_report.json`, `analyst_short.md`;
- `analyst_log.jsonl`, `analyst_delta_log.jsonl`, `researcher_log.jsonl`, `llm_log.jsonl` — шаги ролей;
- `analyst_split.npz`, `analyst_split_code.py` — разбиение Аналитика и код, которым оно построено;
- `registry.json`, `report_changes.jsonl` — реестр гипотез и журнал изменений отчёта;
- `dataset_current.txt`, `current/` — текущие данные решения (D3); `codes/` — код Инженера.
