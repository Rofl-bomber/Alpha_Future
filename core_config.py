"""Настройки агента: бюджеты времени, пороги, лимиты.

Значения по умолчанию рассчитаны на платформу (3 часа, 8 CPU, 32 GB).
Любое поле можно переопределить переменной окружения AGENT_<ИМЯ_ПОЛЯ_В_ВЕРХНЕМ_РЕГИСТРЕ>,
например AGENT_TIME_BUDGET=1800 для 30-минутного прогона на стенде.
"""

import os
import tempfile
from dataclasses import asdict, dataclass, field, fields

# Границы фаз — доли общего бюджета, считаются от старта solution.py.
# 0–5 мин: задача и данные; 5–20: L1; 20–30: диагностика; 30–140: поиск;
# 140–155: финалисты на сидах; 155–178: финальная сборка; 178–180: запись.
PHASES = {
    "spec": 5 / 180,
    "l1": 20 / 180,
    "diag": 30 / 180,
    "search": 140 / 180,
    "reseed": 155 / 180,
    "final": 178 / 180,
    "end": 1.0,
}


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
    leak_corr: float = 0.98
    fast_train_ratio: float = 0.3
    screen_if_full_s: float = 90.0  # скрининг включается, если полный прогон дольше

    # Поиск
    population_k: int = 4
    max_per_family: int = 2
    merge_every: int = 8
    plateau_restart: int = 6
    freeze_after: int = 4
    diagnose_every_accepted: int = 3
    tune_at_frac: float = 2 / 3
    escape_min_left_s: float = 1800.0
    max_queue: int = 12

    # LLM
    llm_timeout_s: float = 300.0
    llm_retries: int = 3
    temperature: float = 0.3
    token_limit_total: int = 1_200_000  # на один запуск (одну задачу); ~0,5 $ входа по ценам Qwen3.8-27B
    token_limit_out: int = 250_000  # ~0,75 $ выхода
    engineer_thinking: int = 0  # 0 — просить сервер отключить размышления у Инженера
    researcher_max_tokens: int = 8000
    engineer_max_tokens: int = 4000

    # Инженер и песочница
    react_max_steps: int = 8
    engineer_submit_attempts: int = 2
    sandbox_timeout_s: float = 120.0
    sample_rows: int = 2000
    max_new_features: int = 30

    # Финал
    finalists: int = 5
    reseed_seeds: int = 2
    bootstrap_n: int = 300
    bootstrap_conf: float = 0.8
    blend_max_steps: int = 30

    phases: dict = field(default_factory=lambda: dict(PHASES))

    def to_dict(self) -> dict:
        return asdict(self)


def load_config() -> Config:
    cfg = Config()
    for f in fields(cfg):
        if f.name == "phases":
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
            pass
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
