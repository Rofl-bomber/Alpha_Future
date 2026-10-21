"""Настройки прототипа. Командная строка фиксирована хакатоном, поэтому всё
переопределяется только переменными окружения (AGENT_*)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class Config:
    # --- цикл ---
    n_iters: int = 10                 # AGENT_ITERS
    max_hypotheses: int = 5           # сколько гипотез Исследователя берём за итерацию
    fail_streak_switch: int = 6       # неудач подряд в одной категории -> смена направления
    t_min: float = 2.0                # keep: средняя парная дельта по фолдам > 0 и t >= t_min
    t_noise: float = 1.0              # 1 <= t < t_min -> статус noise («на грани шума»)
    jump_threshold: float = 0.15      # относительный прирост, после которого зовём Критика

    # --- данные и валидация ---
    n_folds: int = 5
    seed: int = 42
    subsample: int = 0                # AGENT_SUBSAMPLE: >0 -> берём столько строк train
    threads: int = 8                  # AGENT_THREADS: 0 -> os.cpu_count()
    split_unseen: float = 0.95        # доля новых значений в test, с которой колонка считается колонкой сплита
    dup_min: float = 0.05             # доля дубликатов train, с которой фолды группируются по хэшу строки
    adv_rows: int = 50000             # максимум строк с каждой стороны для adversarial validation
    es_frac: float = 0.1              # доля обучающей части фолда под early stopping

    # --- анализ ошибок ---
    err_chars: int = 3500             # лимит сводки анализа ошибок для Исследователя

    # --- Инженер и песочница ---
    engineer_max_fixes: int = 2
    sandbox_timeout: int = 120
    max_new_cols: int = 20

    # --- финал ---
    blend_top_k: int = 5

    # --- LLM ---
    mock_llm: bool = False            # AGENT_MOCK_LLM=1
    temperature: float = 0.2
    llm_cache: bool = True            # AGENT_LLM_CACHE=0 выключает кэш
    llm_cache_dir: str = ""           # AGENT_LLM_CACHE_DIR; пусто -> <каталог задачи>/.agent_cache/llm
    llm_max_tokens: int = 0           # 0 -> не передавать max_tokens
    llm_timeout: float = 600.0
    llm_extra_body: str = ""          # JSON, напр. {"chat_template_kwargs": {"enable_thinking": false}}

    # --- профиль ---
    profile_max_chars: int = 9000     # ~3 тыс. токенов

    # --- пути ---
    runs_dir: str = ""                # AGENT_RUNS_DIR; пусто -> <каталог задачи>/agent_runs

    extra: dict = field(default_factory=dict)

    @property
    def n_threads(self) -> int:
        return self.threads if self.threads > 0 else (os.cpu_count() or 1)


_ENV = {
    "n_iters": ("AGENT_ITERS", int),
    "max_hypotheses": ("AGENT_MAX_HYP", int),
    "fail_streak_switch": ("AGENT_FAIL_STREAK", int),
    "t_min": ("AGENT_T_MIN", float),
    "t_noise": ("AGENT_T_NOISE", float),
    "jump_threshold": ("AGENT_JUMP", float),
    "n_folds": ("AGENT_FOLDS", int),
    "seed": ("AGENT_SEED", int),
    "subsample": ("AGENT_SUBSAMPLE", int),
    "threads": ("AGENT_THREADS", int),
    "split_unseen": ("AGENT_SPLIT_UNSEEN", float),
    "dup_min": ("AGENT_DUP_MIN", float),
    "adv_rows": ("AGENT_ADV_ROWS", int),
    "es_frac": ("AGENT_ES_FRAC", float),
    "err_chars": ("AGENT_ERR_CHARS", int),
    "engineer_max_fixes": ("AGENT_ENGINEER_FIXES", int),
    "sandbox_timeout": ("AGENT_SANDBOX_TIMEOUT", int),
    "max_new_cols": ("AGENT_MAX_NEW_COLS", int),
    "blend_top_k": ("AGENT_BLEND_TOP", int),
    "mock_llm": ("AGENT_MOCK_LLM", bool),
    "temperature": ("AGENT_TEMPERATURE", float),
    "llm_cache": ("AGENT_LLM_CACHE", bool),
    "llm_cache_dir": ("AGENT_LLM_CACHE_DIR", str),
    "llm_max_tokens": ("AGENT_LLM_MAX_TOKENS", int),
    "llm_timeout": ("AGENT_LLM_TIMEOUT", float),
    "llm_extra_body": ("AGENT_LLM_EXTRA_BODY", str),
    "profile_max_chars": ("AGENT_PROFILE_CHARS", int),
    "runs_dir": ("AGENT_RUNS_DIR", str),
}


def load_config() -> Config:
    cfg = Config()
    for f in fields(cfg):
        if f.name in _ENV:
            env_name, cast = _ENV[f.name]
            setattr(cfg, f.name, _env(env_name, getattr(cfg, f.name), cast))
    return cfg
