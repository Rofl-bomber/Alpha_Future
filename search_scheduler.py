"""Планировщик: какой оператор применить следующим.

Все операторы работают от текущего решения (см. search_population). Порядок приоритетов:
  1. диагностика, если запрошена (после L1, после N принятых изменений, на плато);
  2. аварийный оператор, если задача вне библиотеки;
  3. на плато (plateau_restart неудач подряд): аварийный оператор один раз (если времени
     достаточно), затем Restart — другое семейство моделей на текущем решении, и заново диагностика;
  4. Tune текущего решения после tune_at_frac бюджета поиска (один раз на узел);
  5. иначе Improve текущего решения.
"""

from dataclasses import dataclass, field


@dataclass
class Action:
    kind: str  # diagnose | escape | restart | tune | improve
    payload: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, cfg, population, timer, spec):
        self.cfg = cfg
        self.pop = population
        self.timer = timer
        self.spec = spec
        self.iter = 0
        self.no_accept = 0
        self.accepted_since_diag = 0
        self.diag_pending = True
        self.escape_done = False
        self.tuned = set()
        self.search_start = None

    def _search_frac(self) -> float:
        start = self.search_start or self.timer.now()
        end = self.timer.deadline("search")
        return (self.timer.now() - start) / max(1.0, end - start)

    def next(self, restart_options: list) -> Action:
        if self.search_start is None:
            self.search_start = self.timer.now()
        self.iter += 1
        if self.diag_pending:
            self.diag_pending = False
            return Action("diagnose")
        if self.spec.out_of_library and not self.escape_done:
            self.escape_done = True
            return Action("escape")
        if self.no_accept >= self.cfg.plateau_restart:
            self.no_accept = 0
            self.diag_pending = True
            enough = self.cfg.unlimited or self.timer.left("search") > self.cfg.escape_min_left_s
            if not self.escape_done and enough:
                self.escape_done = True
                return Action("escape")
            if restart_options:
                return Action("restart", {"option": restart_options[0]})
        best = self.pop.best()
        if best is not None and self._search_frac() >= self.cfg.tune_at_frac and best.id not in self.tuned \
                and best.family in ("lgbm", "catboost", "xgb", "linear", "extratrees", "mlp"):
            self.tuned.add(best.id)
            return Action("tune", {"node": best})
        return Action("improve")

    def feedback(self, accepted: bool, counts_for_plateau: bool = True):
        if accepted:
            self.no_accept = 0
            self.accepted_since_diag += 1
            if self.accepted_since_diag >= self.cfg.diagnose_every_accepted:
                self.accepted_since_diag = 0
                self.diag_pending = True
        elif counts_for_plateau:
            self.no_accept += 1
