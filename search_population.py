"""Популяция узлов поиска: 3–5 лучших веток по CV плюс постоянный якорь L1.

Узел = конфигурация пайплайна + артефакты слотов (код признаков, модели) + результат.
Родитель выбирается по рангу (вероятность ∝ 1/ранг). Ветка, у которой несколько потомков
подряд не приняты, замораживается. В популяции не больше max_per_family узлов одного
семейства моделей — так рестарты других семейств получают место.
"""

import time
from dataclasses import dataclass, field

import numpy as np

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
    children_failed: int = 0
    frozen: bool = False
    reseed: dict | None = None
    created: float = field(default_factory=time.time)

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
    def __init__(self, k: int, max_per_family: int, greater_is_better: bool, freeze_after: int, seed: int = 0):
        self.k = k
        self.max_per_family = max_per_family
        self.gib = greater_is_better
        self.freeze_after = freeze_after
        self.nodes: dict[str, Node] = {}
        self.members: list[str] = []
        self.anchor_id: str | None = None
        self.rng = np.random.default_rng(seed)
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
        node.status = "anchor"
        self.register(node)

    def ranked(self) -> list:
        ids = [i for i in self.members if self.nodes[i].mean is not None]
        return sorted((self.nodes[i] for i in ids), key=lambda n: -self.gain(n))

    def best(self) -> Node | None:
        pool = [n for n in self.nodes.values() if n.status in ("accepted", "anchor") and n.mean is not None]
        return max(pool, key=self.gain) if pool else None

    def consider(self, node: Node) -> bool:
        """Принятый узел входит в популяцию, если обходит худшего участника с учётом лимита на семейство."""
        if node.mean is None:
            return False
        self.register(node)
        same = [self.nodes[i] for i in self.members if self.nodes[i].family == node.family]
        if len(same) >= self.max_per_family:
            weakest = min(same, key=self.gain)
            if self.gain(node) <= self.gain(weakest):
                return False
            self.members.remove(weakest.id)
        if len(self.members) >= self.k:
            weakest = min((self.nodes[i] for i in self.members), key=self.gain)
            if self.gain(node) <= self.gain(weakest):
                return False
            self.members.remove(weakest.id)
        self.members.append(node.id)
        return True

    def select_parent(self) -> Node:
        ranked = [n for n in self.ranked() if not n.frozen]
        anchor = self.nodes.get(self.anchor_id)
        if not ranked:
            return anchor
        w = np.array([1.0 / (i + 1) for i in range(len(ranked))])
        return ranked[int(self.rng.choice(len(ranked), p=w / w.sum()))]

    def note_child(self, parent_id: str, accepted: bool):
        p = self.nodes.get(parent_id)
        if p is None:
            return
        if accepted:
            p.children_failed = 0
        else:
            p.children_failed += 1
            if p.children_failed >= self.freeze_after and p.id in self.members and len(self.members) > 1:
                p.frozen = True

    def unfreeze_all(self):
        for i in self.members:
            self.nodes[i].frozen = False
            self.nodes[i].children_failed = 0

    def merge_pair(self, tried: set):
        """Пара участников из разных веток, ещё не сливавшаяся."""
        ranked = self.ranked()
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                if a.branch != b.branch and (a.id, b.id) not in tried:
                    return a, b
        return None

    def summary(self) -> str:
        lines = ["Популяция (лучшие ветки):"]
        for n in self.ranked():
            lines.append(f"  {n.id} [{n.family}, ветка {n.branch}] CV={n.mean:.6g}±{np.std(n.fold_scores):.3g}"
                         f"{' (заморожена)' if n.frozen else ''}: {n.desc[:120]}")
            ch = search_operators.diff(self.nodes[self.anchor_id].config, n.config) if self.anchor_id else {}
            if ch:
                lines.append("     отличия от L1: " + ", ".join(f"{k}={v}" for k, v in list(ch.items())[:8]))
        return "\n".join(lines)
