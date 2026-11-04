"""三个编排后端：Harness 启发式、CPU 退火、D-Wave 真机（需 token）。"""

import os
import time

import dimod
import numpy as np

try:
    from dwave.samplers import SimulatedAnnealingSampler
except Exception:  # pragma: no cover
    SimulatedAnnealingSampler = None

from world import blocked_map, damage, feasible, score_of


def _dist(a, b):
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _dist_pt(p, b):
    return ((p[0] - b.cx) ** 2 + (p[1] - b.cy) ** 2) ** 0.5


# ---------------------------------------------------------------- QUBO

def build(bodies, blocked, T, lam=None, w=0.0, start=(0.0, 0.0)):
    """变量 z[o, s]：物体 o 在第 s 步被抓；s = T 表示不抓。

    w > 0 时把机械臂移动距离计入目标：同一列连着干便宜，两步之间跨列就贵。
    """
    n_obj = len(bodies)
    S = T + 1
    n = n_obj * S
    span = 0.0
    if w > 0.0 and n_obj > 1:
        span = max(_dist(bodies[o], bodies[p])
                   for o in range(n_obj) for p in range(n_obj) if o != p)
    lam = float(lam if lam is not None else 2 * T) + w * span * T
    Q = {}

    def add(a, b, v):
        key = (a, b) if a <= b else (b, a)
        Q[key] = Q.get(key, 0.0) + v

    dmg = {o: damage(bodies, o) for o in range(n_obj)}
    for o in range(n_obj):
        sc = score_of(bodies, o) - 0.25 * dmg[o]
        for s in range(T):
            add(o * S + s, o * S + s, -sc)
        for s in range(S):
            add(o * S + s, o * S + s, -lam)
            for s2 in range(S):
                if s < s2:
                    add(o * S + s, o * S + s2, 2 * lam)
    offset = n_obj * lam

    for s in range(T):
        for a in range(n_obj):
            for b in range(a + 1, n_obj):
                add(a * S + s, b * S + s, lam)

    for o in range(n_obj):
        for p in blocked[o]:
            for q in range(T):
                for t in range(q, S):
                    add(o * S + q, p * S + t, lam)
    if w > 0.0:
        for o in range(n_obj):
            add(o * S, o * S, w * _dist_pt(start, bodies[o]))
        for s in range(T - 1):
            for o in range(n_obj):
                for p in range(n_obj):
                    if o != p:
                        add(o * S + s, p * S + s + 1, w * _dist(bodies[o], bodies[p]))

    return {"Q": Q, "n": n, "S": S, "T": T, "lam": lam, "offset": offset, "n_obj": n_obj}


def decode(pack, sample):
    S, T = pack["S"], pack["T"]
    slot = {}
    for o in range(pack["n_obj"]):
        picked = [s for s in range(S) if sample.get(o * S + s, 0) == 1]
        if len(picked) != 1:
            return None
        slot[o] = picked[0]
    seq = sorted(((s, o) for o, s in slot.items() if s < T))
    if len(set(s for s, _ in seq)) != len(seq):
        return None
    return [o for _, o in seq]


def to_bqm(pack):
    Q = {k: v for k, v in pack["Q"].items()}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=pack["offset"])
    return bqm


# ---------------------------------------------------------------- 后端

class HarnessHeuristic:
    """Harness 原逻辑的代理：按分数抓，带失败记忆、解锁偏好；w>0 时按"分数减路程"贪心。"""

    name = "harness"

    def __init__(self, w=0.0, start=(0.0, 0.0)):
        self.w, self.start = w, start

    def plan(self, bodies, blocked, T, mem):
        n = len(bodies)
        done = set()
        fails = mem.get("fails", {})
        stuck = mem.get("stuck")
        order, pos = [], self.start
        for _ in range(T):
            alive = set(range(n)) - done
            cand = [i for i in alive if not (blocked[i] & alive)]
            if not cand:
                break

            def key(i):
                return (score_of(bodies, i) - 0.5 * fails.get(i, 0)
                        - self.w * _dist_pt(pos, bodies[i]))

            pool = cand
            if stuck is not None and stuck in alive:
                unlock = [i for i in cand if i in blocked[stuck]]
                if unlock:
                    pool = unlock
            pick = max(pool, key=key)
            order.append(pick)
            done.add(pick)
            pos = (bodies[pick].cx, bodies[pick].cy)
        return order


class QuboSA:
    name = "sa"

    def __init__(self, reads=40, sweeps=600, seed=0, w=0.0, start=(0.0, 0.0)):
        self.reads, self.sweeps, self.seed = reads, sweeps, seed
        self.w, self.start = w, start

    def plan(self, bodies, blocked, T, mem):
        pack = build(bodies, blocked, T, w=self.w, start=self.start)
        S, fails = pack["S"], mem.get("fails", {})
        for o, n in fails.items():
            if n >= 2:
                for s in range(pack["T"]):
                    pack["Q"][(o * S + s, o * S + s)] = pack["Q"].get((o * S + s, o * S + s), 0.0) + 10.0 * pack["lam"]
        bqm = to_bqm(pack)
        ss = SimulatedAnnealingSampler().sample(
            bqm, num_reads=self.reads, num_sweeps=self.sweeps, seed=self.seed)
        best, best_e = None, None
        for rec in ss.record:
            seq = decode(pack, dict(zip(ss.variables, rec.sample)))
            if not seq:
                continue
            legal = self._first_legal(bodies, blocked, seq)
            if legal and (best is None or len(legal) > len(best)):
                best, best_e = legal, rec.energy
        return best or []

    @staticmethod
    def _first_legal(bodies, blocked, seq):
        out, gone = [], set()
        for o in seq:
            if blocked[o] & (set(range(len(bodies))) - gone):
                break
            out.append(o)
            gone.add(o)
        return out


class QuboQPU:
    name = "qpu"

    def __init__(self, reads=100, anneal=20.0, w=0.0, start=(0.0, 0.0)):
        self.reads, self.anneal = reads, anneal
        self.w, self.start = w, start

    def plan(self, bodies, blocked, T, mem):
        if not os.environ.get("DWAVE_API_TOKEN"):
            raise RuntimeError("no DWAVE_API_TOKEN: set it, then use the qpu backend")
        from dwave.system import DWaveSampler, EmbeddingComposite
        pack = build(bodies, blocked, T, w=self.w, start=self.start)
        sampler = EmbeddingComposite(DWaveSampler())
        ss = sampler.sample(to_bqm(pack), num_reads=self.reads,
                            annealing_time=self.anneal, label="harness-qubo")
        best = []
        for rec in ss.record:
            seq = decode(pack, dict(zip(ss.variables, rec.sample)))
            legal = QuboSA._first_legal(bodies, blocked, seq) if seq else []
            if len(legal) > len(best):
                best = legal
        return best


def make(name, w=0.0, start=(0.0, 0.0), **kw):
    if name == "harness":
        return HarnessHeuristic(w=w, start=start)
    if name == "sa":
        return QuboSA(w=w, start=start, **kw)
    if name == "qpu":
        return QuboQPU(w=w, start=start, **kw)
    raise ValueError(name)
