"""Узлы поиска и цепочка текущего решения.

Узел = конфигурация пайплайна + артефакты слотов (код признаков, модели) + результат.
Поиск идёт цепочкой: каждая гипотеза применяется к текущему решению (incumbent), а принятый
узел сразу становится новым текущим решением. Поэтому в текущем решении всегда есть все
принятые изменения, а цепочка L1 → … → текущее решение — их история по порядку.
Отклонённые узлы (в том числе другие семейства моделей) хранятся для финального бленда.
"""

import time
from dataclasses import dataclass, field

import search_operators


@dataclass
class Node:
    id: str
    config: dict
    parent: str | None
    op: str
    desc: str
    branch: str
    hypothesis: dict | None = None
    result: dict | None = None
    screen: dict | None = None
    alt: dict | None = None
    status: str = "pending"
    flags: list = field(default_factory=list)
    reason: str = ""
    change: str = ""  # что изменено относительно родителя, человекочитаемо
    cmp: dict | None = None  # парное сравнение с родителем по фолдам (mean_gain, se, wins, k)
    reseed: dict | None = None
    stage: str = ""  # подэтап поиска, в котором проверялась гипотеза (preprocessing / features / model)
    created: float = field(default_factory=time.time)

    @property
    def ref(self):
        """id гипотезы Аналитика (H.., R.., T..), которую проверяет этот узел, или None."""
        return (self.hypothesis or {}).get("ref")

    @property
    def mean(self):
        return self.result["mean"] if self.result and self.result.get("ok") else None

    @property
    def fold_scores(self):
        return self.result["fold_scores"] if self.result and self.result.get("ok") else None

    @property
    def family(self) -> str:
        return search_operators.family(self.config)


class Population:
    def __init__(self, greater_is_better: bool):
        self.gib = greater_is_better
        self.nodes: dict[str, Node] = {}
        self.anchor_id: str | None = None
        self.incumbent_id: str | None = None
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"n{self._n:03d}"

    def gain(self, node: Node) -> float:
        return node.mean if self.gib else -node.mean

    def register(self, node: Node):
        self.nodes[node.id] = node

    def set_anchor(self, node: Node):
        self.anchor_id = node.id
        self.incumbent_id = node.id
        node.status = "anchor"
        self.register(node)

    def incumbent(self) -> Node | None:
        return self.nodes.get(self.incumbent_id) if self.incumbent_id else None

    def best(self) -> Node | None:
        """Текущее решение: принятый узел цепочки (метрика по цепочке только улучшается)."""
        return self.incumbent()

    def accept(self, node: Node):
        """Принятый узел становится текущим решением."""
        self.register(node)
        self.incumbent_id = node.id

    def lineage(self) -> list:
        """Принятые изменения по порядку: узлы цепочки от L1 (не включая) до текущего решения."""
        out, cur = [], self.incumbent()
        while cur is not None and cur.id != self.anchor_id:
            out.append(cur)
            cur = self.nodes.get(cur.parent) if cur.parent else None
        return out[::-1]

    def families_tried(self) -> set:
        return {n.family for n in self.nodes.values()}
