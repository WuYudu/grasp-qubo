"""One-hot (slot or skip) QUBO for ordered grasping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scene import Scene


@dataclass
class Pack:
    Q: dict
    T: int
    n: int
    lam: float
    offset: float = 0.0

    @property
    def nnz(self) -> int:
        return len(self.Q)

    def arrays(self):
        keys = list(self.Q.keys())
        a = np.array([k[0] for k in keys], dtype=np.int64)
        b = np.array([k[1] for k in keys], dtype=np.int64)
        v = np.array([self.Q[k] for k in keys], dtype=np.float64)
        return a, b, v

    def energy(self, x):
        a, b, v = self._arr
        x = np.asarray(x, dtype=np.float64)
        e = self.offset + v @ (x[a] * x[b])
        return float(e) if np.ndim(e) == 0 else e

    def __post_init__(self):
        self._arr = self.arrays()


def vid(obj: int, slot: int, S: int) -> int:
    return obj * S + slot


def build(scene: Scene, T: int | None = None, lam: float | None = None) -> Pack:
    T = scene.budget if T is None else T
    S = T + 1
    lam = float(2 * T if lam is None else lam)
    Q: dict = {}
    dq: dict = {}
    offset = 0.0

    def add(a: int, b: int, value: float) -> None:
        if a > b:
            a, b = b, a
        if a == b:
            dq[a] = dq.get(a, 0.0) + value
        else:
            Q[(a, b)] = Q.get((a, b), 0.0) + value

    for o in range(scene.n_obj):
        score = scene.score(o)
        for s in range(S):
            if s < T:
                add(vid(o, s, S), vid(o, s, S), -score)
            add(vid(o, s, S), vid(o, s, S), -lam)
        offset += lam
        for a in range(S):
            for b in range(a + 1, S):
                add(vid(o, a, S), vid(o, b, S), 2.0 * lam)

    for s in range(T):
        for a in range(scene.n_obj):
            for b in range(a + 1, scene.n_obj):
                add(vid(a, s, S), vid(b, s, S), lam)

    for o in range(scene.n_obj):
        for blocker in scene.blocked[o]:
            for q in range(T):
                for p in range(q, S):
                    add(vid(o, q, S), vid(blocker, p, S), lam)

    for k, v in dq.items():
        Q[(k, k)] = Q.get((k, k), 0.0) + v
    return Pack({k: v for k, v in Q.items() if abs(v) > 1e-12}, T, scene.n_obj * S, lam, offset)


def decode(scene: Scene, x, T: int) -> dict:
    S = T + 1
    steps = []
    for o in range(scene.n_obj):
        slots = [s for s in range(S) if x[vid(o, s, S)]]
        if len(slots) != 1:
            return {"feasible": False, "reason": "bad one-hot", "score": 0.0, "seq": []}
        if slots[0] < T:
            steps.append((slots[0], o))
    used = [s for s, _ in steps]
    if len(set(used)) != len(used):
        return {"feasible": False, "reason": "two grasps in one step", "score": 0.0, "seq": []}
    steps.sort()
    return evaluate(scene, [o for _, o in steps], T)


def evaluate(scene: Scene, seq: list[int], T: int | None = None) -> dict:
    if T is not None and len(seq) > T:
        return {"feasible": False, "reason": "over budget", "score": 0.0, "seq": seq}
    removed: set[int] = set()
    score = 0.0
    for o in seq:
        if o in removed:
            return {"feasible": False, "reason": "object already removed", "score": 0.0, "seq": seq}
        if scene.blocked[o] - removed:
            return {"feasible": False, "reason": "blocked", "score": 0.0, "seq": seq}
        removed.add(o)
        score += scene.score(o)
    return {"feasible": True, "reason": "", "score": score, "seq": seq}
