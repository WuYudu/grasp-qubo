"""Perception error, execution with recovery moves, and three planners."""
import math
import random
import time

import numpy as np

import world2 as W


# ---------------------------------------------------------------- perception

def observe(sc, rng, sigma=2.5, p_miss=0.10, bloat=0.05):
    """What the planner sees: shifted bodies, slightly fattened, sometimes missing."""
    bodies = []
    for b in sc.bodies:
        jx, jy = rng.gauss(0, sigma), rng.gauss(0, sigma)
        p = b.poly - np.array([b.cx, b.cy])
        p = p * (1.0 + bloat)
        p = p + np.array([b.cx + jx, b.cy + jy])
        nb = W.Body(p, b.mass, b.level, b.cluster, b.target, b.fragile)
        bodies.append(nb)
    seen = W.rebuild(sc, bodies)
    seen.targets = set(sc.targets)
    for o in list(seen.targets):
        if rng.random() < p_miss:
            seen.targets.discard(o)          # the detector missed it this round
    return seen


# ----------------------------------------------------------------- execution

def success_prob(sc, o, g):
    """Abstract execution model: geometry decides most of it."""
    blocked = sc.blocked(o)
    pen = 0.55 * min(2, len(blocked))
    touch_fragile = sum(1 for p in blocked if p in sc.fragile)
    mass_pen = 0.12 * max(0.0, sc.bodies[o].mass - 1.5)
    p = 0.97 * g.fit - pen - 0.25 * touch_fragile - mass_pen
    return float(min(0.97, max(0.03, p)))


def push(sc, o, rng, dist=18.0):
    """Shove a body aside to open a corridor. Returns (new_scene, ok)."""
    b = sc.bodies[o]
    a = rng.uniform(0, 2 * math.pi)
    nb = b.moved(dist * math.cos(a), dist * math.sin(a), rot=rng.uniform(-0.2, 0.2))
    bodies = list(sc.bodies)
    bodies[o] = nb
    ok = rng.random() < 0.78
    if not ok:                                # a failed shove still rattles the pile
        for _ in range(2):
            p = rng.randrange(sc.n)
            bodies[p] = bodies[p].moved(rng.gauss(0, 3), rng.gauss(0, 3))
    return W.rebuild(sc, bodies), ok


def settle(sc, rng, sigma=1.5):
    bodies = [b.moved(rng.gauss(0, sigma), rng.gauss(0, sigma)) for b in sc.bodies]
    return W.rebuild(sc, bodies)


# ------------------------------------------------------------------ planners

CLEAR_WEIGHT = 0.45          # a non-target grasp is worth this much of its score


def value(sc, o):
    """Targets are what we want; non-targets only earn their clearing value."""
    s = sc.best(o).score
    return s if o in sc.targets else CLEAR_WEIGHT * s


def feasible(sc, o):
    return not sc.blocked(o) and sc.best(o).score > 0.0


def greedy(sc, budget):
    """Highest-scoring target whose corridor is clear right now."""
    open_objs = [o for o in range(sc.n) if feasible(sc, o)]
    open_objs.sort(key=lambda o: -value(sc, o))
    return [(o, "grasp") for o in open_objs[:budget]]


def _window_problem(sc, T):
    """Sequence over targets inside the window, precedence from the current blocking."""
    cand = [o for o in sc.targets if sc.best(o).score > 0.0]
    cand.sort(key=lambda o: -sc.best(o).score)
    cand = cand[:max(3 * T, T + 4)]
    idx = {o: i for i, o in enumerate(cand)}
    S, n = T + 1, len(cand)
    Q = np.zeros((n * S, n * S))

    def v(i, s):
        return i * S + s

    for i, o in enumerate(cand):
        for s in range(S):
            Q[v(i, s), v(i, s)] -= value(sc, o) if s < T else 0.0
    lam = 2.0 * max(1, T)
    for i in range(n):                         # one slot per object
        for s in range(S):
            Q[v(i, s), v(i, s)] -= lam
            for t in range(s + 1, S):
                Q[v(i, s), v(i, t)] += 2 * lam
    for s in range(T):                         # one object per slot
        for i in range(n):
            for j in range(i + 1, n):
                Q[v(i, s), v(j, s)] += lam
    for i, o in enumerate(cand):               # precedence, only when o is actually grasped
        for blk in sc.blocked(o) & set(cand):
            j = idx[blk]
            for s in range(T):
                for t in range(s, S):
                    Q[v(i, s), v(j, t)] += lam
    return cand, S, Q


def milp(sc, T=5, time_limit=2.0):
    """Same window problem as the QUBO, written as linear constraints."""
    import io
    import contextlib
    import pulp
    cand = [o for o in range(sc.n) if sc.best(o).score > 0.0]
    cand.sort(key=lambda o: -value(sc, o))
    cand = cand[:max(3 * T, T + 4)]
    S, n = T + 1, len(cand)
    if n == 0:
        return []
    prob = pulp.LpProblem("win", pulp.LpMinimize)
    y = {(i, s): prob.add_variable("y_%d_%d" % (i, s), cat=pulp.LpBinary)
         for i in range(n) for s in range(S)}
    prob += pulp.lpSum(-value(sc, cand[i]) * y[i, s]
                       for i in range(n) for s in range(T))
    for i in range(n):
        prob += pulp.lpSum(y[i, s] for s in range(S)) == 1
    for s in range(T):
        prob += pulp.lpSum(y[i, s] for i in range(n)) <= 1
    for i, o in enumerate(cand):
        for blk in sc.blocked(o) & set(cand):
            j = cand.index(blk)
            for s in range(T):
                prob += y[i, s] <= pulp.lpSum(y[j, t] for t in range(s))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    seq = []
    for s in range(T):
        for i in range(n):
            if pulp.value(y[i, s]) and pulp.value(y[i, s]) > 0.5:
                seq.append((cand[i], "grasp"))
    return seq


def sa(sc, T=5, reads=40, sweeps=800, seed=0):
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    cand, S, Q = _window_problem(sc, T)
    qd = {(i, j): float(Q[i][j]) for i in range(Q.shape[0]) for j in range(Q.shape[1])
          if Q[i][j] != 0.0}
    bqm = dimod.BinaryQuadraticModel.from_qubo(qd)
    res = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=sweeps,
                                            seed=seed)
    best, seq = None, []
    for sample in res.samples():
        var = {k: v for k, v in sample.items() if v > 0.5}
        out = []
        for s in range(T):
            for i in range(len(cand)):
                if var.get(i * S + s, 0) > 0.5:
                    out.append((cand[i], "grasp"))
        # keep the first slot order, drop duplicates
        seen, clean = set(), []
        for o, a in out:
            if o in seen:
                continue
            seen.add(o)
            clean.append((o, a))
        clean = _legal(sc, clean)
        if best is None or len(clean) > len(best):
            best, seq = clean, clean
    return seq


def _legal(sc, seq):
    """Keep only the prefix that can actually be executed in order."""
    done, out = set(), []
    for o, a in seq:
        if sc.blocked(o) <= done:
            out.append((o, a))
            done.add(o)
    return out


PLANNERS = {"greedy": lambda s, T: greedy(s, T),
            "milp": lambda s, T: milp(s, T=T),
            "sa": lambda s, T: sa(s, T=T)}
