"""Настройки агента: бюджеты времени, пороги, лимиты.

Значения по умолчанию рассчитаны на платформу (3 часа, 8 CPU, 32 GB).
Любое поле можно переопределить переменной окружения AGENT_<ИМЯ_ПОЛЯ_В_ВЕРХНЕМ_РЕГИСТРЕ>,
например AGENT_TIME_BUDGET=1800 для 30-минутного прогона.

Границы фаз — доли AGENT_TIME_BUDGET, переопределяются AGENT_PHASE_SPEC, AGENT_PHASE_ANALYSIS,
AGENT_PHASE_L1, AGENT_PHASE_SEARCH, AGENT_PHASE_RESEED, AGENT_PHASE_FINAL (числа в (0, 1],
строго по возрастанию; иначе берутся значения по умолчанию). Границы анализа и L1 — потолки:
если этап закончился раньше, следующий начинается сразу.

AGENT_UNLIMITED=1 — режим без внутренних ограничений: время ограничивают только AGENT_TIME_BUDGET
и границы фаз (каждое действие может идти до конца своей фазы), max_tokens в запросы к LLM не
передаётся, лимиты токенов на прогон и таймаут одного вызова LLM сняты.
"""

import os
import tempfile
from dataclasses import asdict, dataclass, field, fields

# Границы фаз — доли общего бюджета, считаются от старта solution.py.
# 0–3 мин: TaskSpec и константный сабмит; до 45: анализ данных (потолок); до 60: якорный L1 (потолок);
# до 140: поиск (Preprocessing → Features → Model); 140–155: финалисты на сидах;
# 155–178: финальная сборка; 178–180: запись.
PHASES = {
    "spec": 3 / 180,
    "analysis": 45 / 180,
    "l1": 60 / 180,
    "search": 140 / 180,
    "reseed": 155 / 180,
    "final": 178 / 180,
    "end": 1.0,
}
PHASE_ORDER = ("spec", "analysis", "l1", "search", "reseed", "final", "end")
STAGES = ("preprocessing", "features", "model")  # подэтапы поиска


def detect_cpus() -> int:
    """Число доступных CPU с учётом ограничений контейнера (cgroup), а не хоста."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            n = min(n, max(1, quota // period))
    except Exception:
        pass
    return max(1, n)


def detect_memory_gb() -> float:
    """Доступная память в гигабайтах с учётом лимита контейнера."""
    limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw != "max":
                value = int(raw)
                if value < 1 << 50:
                    limit = value
                    break
        except Exception:
            continue
    try:
        import psutil

        total = psutil.virtual_memory().total
        limit = min(limit, total) if limit else total
    except Exception:
        pass
    return (limit or 32 * 1024**3) / 1024**3


@dataclass
class Config:
    # Время
    time_budget: float = 3 * 3600.0
    unlimited: bool = False  # AGENT_UNLIMITED=1: только AGENT_TIME_BUDGET и границы фаз, без лимитов токенов
    # Ресурсы
    n_jobs: int = 0  # 0 — определить автоматически
    mem_limit_gb: float = 0.0  # 0 — 85% доступной памяти
    workdir: str = ""
    seed: int = 42

    # Протокол оценки
    n_folds: int = 5
    screen_folds: int = 3
    holdout_frac: float = 0.15
    holdout_frac_large: float = 0.10
    large_rows: int = 200_000
    group_unseen_frac: float = 0.5  # доля строк test с невиданными значениями, чтобы считать колонку группой
    adv_shift_auc: float = 0.6  # AUC adversarial validation, выше которого считаем, что есть сдвиг
    adv_weight_max_auc: float = 0.95  # выше — веса важности ненадёжны и не используются

    # Критерий принятия и проверки
    accept_min_fold_frac: float = 0.6  # улучшение минимум на 3 фолдах из 5
    noise_se_mult: float = 1.0
    min_delta_rel: float = 1e-4
    jump_rel: float = 0.15
    fast_train_ratio: float = 0.3
    screen_if_full_s: float = 90.0  # скрининг включается, если полный прогон дольше

    # Анализ данных (Аналитик в Python-сессии)
    analysis_steps: int = 60  # {steps} в промпте Аналитика
    analysis_step_timeout_s: float = 120.0  # {timeout} в промпте: лимит одного запуска кода
    analysis_max_out: int = 4000  # {max_out}: длиннее — вывод сокращается до начала и конца
    analysis_step_max_tokens: int = 16000  # max_tokens одного шага Аналитика
    analysis_temperature: float = 0.6
    analysis_keep_full: int = 8  # столько последних обменов идёт в историю полностью, старые сокращаются
    short_report_tokens: int = 2000  # объём короткой части A
    kernel_threads: int = 2  # потоков у Python-сессии (параллельно идёт страховочный L1)
    holdout_min_frac: float = 0.05  # допустимая доля holdout в submit_split
    holdout_max_frac: float = 0.40
    recall_steps: int = 15  # повторный вызов Аналитика: шаги и минуты
    recall_minutes: float = 8.0

    # Поиск: подэтапы Preprocessing → Features → Model, жадная цепочка от текущего решения
    stage_pre_frac: float = 0.15
    stage_feat_frac: float = 0.60
    stage_model_frac: float = 0.25
    plateau_n: int = 6  # столько отклонений подряд = плато подэтапа
    near_miss_per_accept: int = 2  # после принятия: сколько «почти прошедших» гипотез повторить на новом решении
    near_miss_total: int = 4  # и сколько таких повторов всего за прогон
    tune_at_frac: float = 2 / 3  # Optuna текущей модели — после этой доли подэтапа Model
    tune_trials: int = 40  # максимум проб Optuna за один запуск Tune
    max_queue: int = 12
    researcher_tool_steps: int = 8  # инструментальных шагов (python, read_report) на один вызов Исследователя
    researcher_hypotheses: int = 4
    critic_enabled: bool = False  # Критик скачков и финалистов — заглушка: всегда «принять» и «никого не исключать»
    code_critic: bool = True  # Критик кода: сверка просьбы, кода transform и сводки результата; одна попытка исправить
    critic_max_tokens: int = 16000

    # LLM
    llm_timeout_s: float = 300.0
    llm_retries: int = 3
    temperature: float = 0.3
    token_limit_total: int = 3_000_000  # на один запуск всех ролей, кроме Аналитика (Исследователь с инструментами)
    token_limit_out: int = 400_000
    token_limit_analyst: int = 2_000_000  # отдельный лимит Аналитика (анализ + повторные вызовы)
    engineer_thinking: int = 0  # 0 — просить сервер отключить размышления у Инженера
    researcher_max_tokens: int = 30000
    engineer_max_tokens: int = 15000

    # Инженер и песочница
    engineer_attempts: int = 3  # попыток написать код, который проходит запуск на подвыборке
    sandbox_timeout_s: float = 180.0  # transform проверяется на настоящем фолде на полных данных
    sample_rows: int = 2000  # подвыборка для проверки CustomModel и fit_predict
    cache_max_gb: float = 6.0  # кэш результата принятой цепочки transform по фолдам (на диске)

    # Финал
    finalists: int = 5
    reseed_seeds: int = 2
    bootstrap_n: int = 300
    bootstrap_conf: float = 0.8
    blend_max_steps: int = 30

    phases: dict = field(default_factory=lambda: dict(PHASES))
    notes: list = field(default_factory=list)  # замечания при чтении настроек, пишутся в лог

    def to_dict(self) -> dict:
        return asdict(self)


def _load_phases(cfg: Config):
    """Границы фаз из AGENT_PHASE_*: доли бюджета, строго по возрастанию."""
    phases = dict(PHASES)
    changed = False
    for key in PHASE_ORDER[:-1]:
        raw = os.environ.get("AGENT_PHASE_" + key.upper())
        if raw is None or raw.strip() == "":
            continue
        try:
            phases[key] = float(raw)
            changed = True
        except ValueError:
            cfg.notes.append(f"AGENT_PHASE_{key.upper()}={raw!r} — не число, оставлено {PHASES[key]:.4f}")
    if not changed:
        return
    values = [phases[k] for k in PHASE_ORDER]
    ok = all(0 < v <= 1 for v in values) and all(a < b for a, b in zip(values, values[1:]))
    if ok:
        cfg.phases = phases
        cfg.notes.append("границы фаз из окружения: " + ", ".join(f"{k}={phases[k]:.4f}" for k in PHASE_ORDER))
    else:
        cfg.notes.append("AGENT_PHASE_* не возрастают или вне (0, 1] — используются границы по умолчанию: "
                         + ", ".join(f"{k}={phases[k]:.4f}" for k in PHASE_ORDER))


def load_config() -> Config:
    cfg = Config()
    for f in fields(cfg):
        if f.name in ("phases", "notes"):
            continue
        raw = os.environ.get("AGENT_" + f.name.upper())
        if raw is None or raw == "":
            continue
        current = getattr(cfg, f.name)
        try:
            if isinstance(current, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(float(raw))
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(cfg, f.name, value)
        except ValueError:
            cfg.notes.append(f"AGENT_{f.name.upper()}={raw!r} — неверное значение, оставлено {current!r}")
    _load_phases(cfg)
    if cfg.unlimited:
        cfg.notes.append("режим AGENT_UNLIMITED: действия идут до конца своей фазы, лимитов токенов нет")
    if cfg.n_jobs <= 0:
        cfg.n_jobs = detect_cpus()
    if cfg.mem_limit_gb <= 0:
        cfg.mem_limit_gb = round(detect_memory_gb() * 0.85, 1)
    if not cfg.workdir:
        try:
            cfg.workdir = tempfile.mkdtemp(prefix="agent_run_")
        except OSError:  # нет доступного временного каталога — рядом с текущей директорией
            cfg.workdir = os.path.join(os.getcwd(), f"agent_run_{os.getpid()}")
    os.makedirs(cfg.workdir, exist_ok=True)
    return cfg
