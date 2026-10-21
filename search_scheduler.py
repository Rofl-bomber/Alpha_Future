"""Планировщик: какой оператор применить следующим.

Порядок приоритетов:
  1. диагностика, если запрошена (после L1, после N принятых изменений, на плато);
  2. аварийный оператор, если задача вне библиотеки;
  3. Merge каждые merge_every итераций (если есть пара веток);
  4. на плато (plateau_restart неудач подряд по всей популяции): аварийный оператор один раз
     (если времени достаточно), затем Restart, и заново диагностика;
  5. Tune лучшего узла после tune_at_frac бюджета поиска (один раз на узел);
  6. иначе Improve.
"""

from dataclasses import dataclass, field


@dataclass
class Action:
    kind: str  # diagnose | escape | merge | restart | tune | improve
    payload: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, cfg, population, timer, spec):
        self.cfg = cfg
        self.pop = population
        self.timer = timer
        self.spec = spec
        self.iter = 0
        self.since_merge = 0
        self.no_accept = 0
        self.accepted_since_diag = 0
        self.diag_pending = True
        self.escape_done = False
        self.tuned = set()
        self.merged = set()
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
        if self.since_merge >= self.cfg.merge_every:
            pair = self.pop.merge_pair(self.merged)
            self.since_merge = 0
            if pair:
                self.merged.add((pair[0].id, pair[1].id))
                return Action("merge", {"a": pair[0], "b": pair[1]})
        if self.no_accept >= self.cfg.plateau_restart:
            self.no_accept = 0
            self.diag_pending = True
            self.pop.unfreeze_all()
            if not self.escape_done and self.timer.left("search") > self.cfg.escape_min_left_s:
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
        self.since_merge += 1
        if accepted:
            self.no_accept = 0
            self.accepted_since_diag += 1
            if self.accepted_since_diag >= self.cfg.diagnose_every_accepted:
                self.accepted_since_diag = 0
                self.diag_pending = True
        elif counts_for_plateau:
            self.no_accept += 1
