# Прототип: Исследователь — Инженер — Критик

Реализация `prototype_spec.md` (Альфа Хак, AutoML-агент). Детерминированный харнесс исполняет эксперименты и сам решает keep/discard; LLM в трёх ролях предлагает гипотезы, пишет код признаков и проверяет подозрительные скачки.

## Запуск

```bash
OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py путь/к/task.md
```

- Результат: `submission.csv` в каталоге task.md. Он записывается после L1 и после каждого принятого улучшения, а в конце перезаписывается выбором финала.
- Журнал прогона: `agent_runs/<время>/` рядом с task.md:
  - `journal.jsonl`, `llm_log.jsonl`, `final.json`, `profile.txt`, `spec.json`;
  - `preds/*.npz` — OOF и test всех экспериментов;
  - `features/fe_XXX.py` — код Инженера.
- Кэш ответов LLM: `.agent_cache/llm/` рядом с task.md.
- `pyproject.toml` и inline-метаданных нет намеренно: иначе `uv run` попытается ставить пакеты, а в контейнере нет интернета.

Настройки задаются переменными окружения (`agent/config.py`), командная строка не меняется:

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `AGENT_ITERS` | 10 | число итераций цикла |
| `AGENT_MOCK_LLM` | 0 | `1` — подставная LLM без эндпоинта |
| `AGENT_SUBSAMPLE` | 0 | подвыборка train (например, 10000 для первого живого прогона) |
| `AGENT_THREADS` | все CPU | потоки моделей |
| `AGENT_LLM_CACHE` | 1 | `0` — выключить кэш ответов |
| `AGENT_LLM_EXTRA_BODY` | — | JSON в `extra_body`, например `{"chat_template_kwargs": {"enable_thinking": false}}` для Qwen3 |
| `AGENT_LLM_MAX_TOKENS`, `AGENT_TEMPERATURE` | —, 0.2 | параметры запроса |
| `AGENT_JUMP`, `AGENT_MIN_FOLDS`, `AGENT_FAIL_STREAK` | 0.15, 3, 6 | порог скачка для Критика, keep по фолдам, смена направления |

## Структура

```text
solution.py                 точка входа (формат хакатона)
agent/
  config.py                 настройки + переопределение из окружения
  orchestrator.py           L1 → цикл → финал; фильтр гипотез, keep/discard, смена направления
  spec.py                   TaskSpec по правилам + загрузка данных
  pipeline.py               JSON-конфигурация пайплайна: схема, L1-дефолты, применение патчей
  profiler.py               профиль и сводка для Исследователя (~3 тыс. токенов)
  metrics.py, cv.py         rmse / f1_macro; KFold / StratifiedKFold(5), общие фолды
  features/builtin.py       freq_enc, флаги пропусков, OOF target encoding (только в фолдах), редкие категории,
                            group_stats, drop
  features/sandbox.py       make_features в отдельном процессе (120 с), проверки, реестр fe_XXX
  models/                   lightgbm, catboost, xgboost, extratrees, linear, mlp, resnet (torch, CPU)
  runner.py                 run_experiment: фолды, OOF, test, постобработка; reuse OOF для postprocess-гипотез
  postprocess.py            множители классов под macro F1 (на OOF), клиппинг регрессии
  ensemble.py               бленд топ-5 по OOF, неотрицательные веса (SLSQP)
  journal.py                journal.jsonl, best(), summary(k), лог LLM, preds/
  submission.py             submission.csv по sample_submission, порядок строк test
  llm/client.py             OpenAI SDK, JSON, pydantic, 1 повтор с ошибкой, токены по ролям, кэш
  llm/mock.py               сценарий mock-LLM на все ветки
  llm/roles.py              Исследователь, Инженер, Критик (схемы ответов)
  llm/prompts/              шаблоны промптов
bench/
  make_synth.py             синтетические задачи в формате платформы (рег., мультикласс, бинарная)
  check_mock.py             шаг 1: mock-прогон и проверка всех веток по журналам
  analyze_logs.py           шаг 3: числа по ролям из journal.jsonl и llm_log.jsonl
  critic_control.py         шаг 3: признак из зашумлённого таргета → Критик должен отклонить
  make_holdout.py, score.py шаг 4: двойник задачи с отложенными 30% train и оценка L1/финала на нём
  fake_llm_server.py        локальный OpenAI-совместимый эндпоинт: smoke-тест живого пути (SDK, <think>, кэш)
```

## Проверка (по шагам спецификации)

```bash
# 1. Mock без эндпоинта: все ветки (≈15 мин на 2 CPU)
python bench/check_mock.py

# 2. Первый живой прогон: подвыборка, 2–3 итерации, затем полные данные
AGENT_SUBSAMPLE=10000 AGENT_ITERS=3 OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... \
  uv run solution.py путь/к/task.md

# 3. Разбор логов по ролям и контроль Критика
python bench/analyze_logs.py путь/к/agent_runs/<время>
OPENAI_...=... python bench/critic_control.py путь/к/task.md

# 4. Прирост над L1 на отложенных 30% train
python bench/make_holdout.py путь/к/task.md /tmp/holdout_reg
uv run solution.py /tmp/holdout_reg/task.md && python bench/score.py /tmp/holdout_reg/task.md
```

Чтобы измерить разброс между двумя одинаковыми прогонами, кэш выключают через `AGENT_LLM_CACHE=0`. С включённым кэшем повторный прогон воспроизводит те же ответы LLM.

Скорость: на 2 CPU синтетика в 6 тыс. строк проходит 10 итераций примерно за 7 минут. На полных данных мультикласса (87 тыс. строк, 10 классов) один эксперимент CatBoost или LightGBM занимает минуты. Поэтому первые живые прогоны лучше делать с `AGENT_SUBSAMPLE`.

## Отличия от контрактов спецификации

- `run_experiment(config, spec, folds, data, ...)` и `profile(spec, data)` принимают загруженные данные объектом `Data` (train+test признаки, закодированный таргет).
- Отдельный модуль `pipeline.py`: в нём схема конфигурации и применение патчей. Списки патчатся через `["+x", "-x"]`, а смена `model.name` сбрасывает `model.params`.
- Если Критик не дал валидного ответа на скачок, результат отклоняется. Если не дал валидного финального выбора, берётся лучший по OOF вариант.
- У OpenAI SDK остаются встроенные 2 повтора при сетевых ошибках; своих ретраев с backoff нет.
