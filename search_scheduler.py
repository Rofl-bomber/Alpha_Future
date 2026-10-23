"""Планировщик поиска: подэтапы Preprocessing → Features → Model и следующее действие внутри подэтапа.

Подэтап хранит и переключает харнесс (жёсткий вариант). Время подэтапа: при входе в подэтап
оставшееся время поиска делится между ним и следующими подэтапами по долям cfg.stage_*_frac,
поэтому неистраченное время прошлых подэтапов переходит дальше.

Плато — cfg.plateau_n отклонённых гипотез подряд. Первое плато подэтапа → повторный вызов
Аналитика (в фоне, поиск продолжается), второе → переход к следующему подэтапу.

Действия внутри подэтапа:
  1. escape — в начале подэтапа Model, если задача вне библиотеки;
  2. recall / advance — на плато (см. выше); advance — и по истечении времени подэтапа;
  3. tune — Optuna текущей модели, когда прошла доля cfg.tune_at_frac подэтапа Model (один раз на узел);
  4. improve — следующая гипотеза подэтапа.
"""

import time
from dataclasses import dataclass, field

from core_config import STAGES


@dataclass
class Action:
    kind: str  # escape | recall | advance | tune | improve
    payload: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, cfg, population, timer, spec):
        self.cfg = cfg
        self.pop = population
        self.timer = timer
        self.spec = spec
        self.stage_i = -1
        self.stage = None
        self.stage_start = None
        self.stage_end = None
        self.no_accept = 0
        self.plateaus = 0
        self.escape_done = False
        self.tuned = set()
        self.iter = 0

    def _frac(self, i: int) -> float:
        return {0: self.cfg.stage_pre_frac, 1: self.cfg.stage_feat_frac, 2: self.cfg.stage_model_frac}[i]

    def enter(self, i: int) -> str:
        """Войти в подэтап i: его конец — доля оставшегося времени поиска."""
        self.stage_i = i
        self.stage = STAGES[i]
        now = time.time()
        end = self.timer.deadline("search")
        rest = sum(self._frac(j) for j in range(i, len(STAGES)))
        share = self._frac(i) / rest if rest > 0 else 1.0
        self.stage_start = now
        self.stage_end = now + max(0.0, end - now) * share
        self.no_accept = 0
        self.plateaus = 0
        return self.stage

    def stage_left(self) -> float:
        return max(0.0, (self.stage_end or time.time()) - time.time())

    def stage_frac(self) -> float:
        span = max(1.0, (self.stage_end or 0) - (self.stage_start or 0))
        return (time.time() - (self.stage_start or time.time())) / span

    def next(self) -> Action:
        self.iter += 1
        if self.stage == "model" and self.spec.out_of_library and not self.escape_done:
            self.escape_done = True
            return Action("escape")
        if self.stage_left() <= 0:
            return Action("advance", {"why": "время подэтапа вышло"})
        if self.no_accept >= self.cfg.plateau_n:
            self.no_accept = 0
            self.plateaus += 1
            if self.plateaus == 1:
                return Action("recall", {"why": f"плато: {self.cfg.plateau_n} отклонений подряд"})
            return Action("advance", {"why": "второе плато подэтапа"})
        best = self.pop.best()
        if self.stage == "model" and best is not None and self.stage_frac() >= self.cfg.tune_at_frac \
                and best.id not in self.tuned and best.family in ("lgbm", "catboost", "xgb", "linear", "extratrees",
                                                                  "mlp"):
            self.tuned.add(best.id)
            return Action("tune", {"node": best})
        return Action("improve")

    def feedback(self, accepted: bool, counts_for_plateau: bool = True):
        if accepted:
            self.no_accept = 0
        elif counts_for_plateau:
            self.no_accept += 1
