"""Candidate-level QUBO: one variable per (object, grasp pose, step).

The object-level encoder fixes one grasp pose per object before optimising the
order. Here the pose is a decision variable as well, which matters because a
pose that squeezes between two neighbours blocks fewer objects than one that
straddles them.
"""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass

import numpy as np

import qubo
import solvers


@dataclass
class Pack:
    Q: np.ndarray
    T: int
    k: int
    n_obj: int
    lam: float
    offset: float

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(np.triu(self.Q)))

    def arrays(self):
        return self.Q, self.offset

    def energy(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return x @ self.Q @ x.T + self.offset


def vid(o: int, c: int, s: int, k: int, T: int) -> int:
    return (o * k + c) * (T + 1) + s


def build(sc, T: int | None = None, lam: float | None = None, k: int = 3) -> Pack:
    T = sc.budget if T is None else T
    S, M = T + 1, sc.n_obj
    lam = 2.0 * T if lam is None else lam
    n = M * k * S
    Q = np.zeros((n, n))
    off = 0.0

    for o in range(M):
        for c in range(k):
            s_obj = sc.grasps[o][c].score
            for s in range(T):
                i = vid(o, c, s, k, T)
                Q[i, i] -= s_obj

    for o in range(M):
        idx = [vid(o, c, s, k, T) for c in range(k) for s in range(S)]
        for i in idx:
            Q[i, i] -= lam
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                Q[min(i, j), max(i, j)] += 2.0 * lam
        off += lam

    for s in range(T):
        idx = [vid(o, c, s, k, T) for o in range(M) for c in range(k)]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                Q[min(i, j), max(i, j)] += lam

    for o in range(M):
        for c in range(k):
            for s in range(T):
                i = vid(o, c, s, k, T)
                for blk in sc.cand_blocked[o][c]:
                    for c2 in range(k):
                        for p in range(s, S):
                            j = vid(blk, c2, p, k, T)
                            Q[min(i, j), max(i, j)] += lam
    return Pack(Q, T, k, M, lam, off)


def decode(sc, x, T: int | None = None, k: int = 3) -> dict:
    T = sc.budget if T is None else T
    S, M = T + 1, sc.n_obj
    x = np.asarray(x).astype(int)
    pick = {}
    for o in range(M):
        on = [(c, s) for c in range(k) for s in range(S) if x[vid(o, c, s, k, T)]]
        if len(on) != 1:
            return {"feasible": False, "reason": "bad one-hot", "score": 0.0,
                    "seq": [], "pose": []}
        pick[o] = on[0]
    slot = {}
    for o, (c, s) in pick.items():
        if s == T:
            continue
        if s in slot:
            return {"feasible": False, "reason": "two grasps in one step", "score": 0.0,
                    "seq": [], "pose": []}
        slot[s] = o
    step_of = {o: s for o, (c, s) in pick.items() if s < T}
    for o, s in step_of.items():
        for blk in sc.cand_blocked[o][pick[o][0]]:
            if step_of.get(blk, T) >= s:
                return {"feasible": False, "reason": "blocked", "score": 0.0,
                        "seq": [], "pose": []}
    order = sorted(step_of, key=lambda o: step_of[o])
    score = sum(sc.grasps[o][pick[o][0]].score for o in order)
    return {"feasible": True, "reason": "", "score": round(score, 6),
            "seq": order, "pose": [pick[o][0] for o in order]}


def evaluate(sc, x, T: int | None = None, k: int = 3) -> dict:
    return decode(sc, x, T, k)


def encode(sc, seq, poses, T: int | None = None, k: int = 3) -> np.ndarray:
    T = sc.budget if T is None else T
    S, M = T + 1, sc.n_obj
    x = np.zeros(M * k * S, dtype=int)
    for o in range(M):
        x[vid(o, 0, T, k, T)] = 1
    for o, c in zip(seq, poses):
        x[vid(o, 0, T, k, T)] = 0
        x[vid(o, c, seq.index(o), k, T)] = 1
    return x


def exact_milp_cand(sc, T: int | None = None, k: int = 3,
                    time_limit: float = 120.0) -> solvers.Result:
    import pulp

    T = sc.budget if T is None else T
    S, M = T + 1, sc.n_obj
    t0 = time.perf_counter()
    prob = pulp.LpProblem("grasp_cand", pulp.LpMaximize)
    z = {(o, c, s): prob.add_variable(f"z_{o}_{c}_{s}", cat=pulp.LpBinary)
         for o in range(M) for c in range(k) for s in range(S)}
    prob += pulp.lpSum(sc.grasps[o][c].score * z[(o, c, s)]
                       for o in range(M) for c in range(k) for s in range(T))
    for o in range(M):
        prob += pulp.lpSum(z[(o, c, s)] for c in range(k) for s in range(S)) == 1
    for s in range(T):
        prob += pulp.lpSum(z[(o, c, s)] for o in range(M) for c in range(k)) <= 1
    for o in range(M):
        for c in range(k):
            for s in range(T):
                for blk in sc.cand_blocked[o][c]:
                    prob += z[(o, c, s)] + pulp.lpSum(
                        z[(blk, c2, p)] for c2 in range(k) for p in range(s, S)) <= 1
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    ms = (time.perf_counter() - t0) * 1e3
    ok = stats is not None and getattr(stats, "has_solution", False)
    seq, poses = [], []
    if ok:
        for s in range(T):
            for o in range(M):
                for c in range(k):
                    v = pulp.value(z[(o, c, s)])
                    if v and v > 0.5:
                        seq.append(o)
                        poses.append(c)
    out = decode(sc, encode(sc, seq, poses, T, k), T, k)
    return solvers.Result("milp-cand", float(out["score"]), out["feasible"], seq, ms,
                          {"poses": poses, "status": str(getattr(stats, "status", "")),
                           "reason": out["reason"]})
