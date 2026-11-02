"""Plan a window of T pick-and-place moves on the current world state.

The QUBO decides which objects to move and in what order; the tray slot for each
object is assigned after the order is fixed, by taking the nearest free slot.
With six equal-width slots and T <= 6 that assignment is never the bottleneck,
which is checked separately on the small joint model.
"""

import time

import world100 as W

try:
    import pulp
except Exception:
    pulp = None
try:
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
except Exception:
    dimod = None


class Plan:
    def __init__(self, ids, order, ms, score, name):
        self.ids = ids          # window index -> object id
        self.order = order      # list of window indices, in schedule order
        self.ms = ms
        self.score = score
        self.name = name

    def __repr__(self):
        return "<%s n=%d %.1fms>" % (self.name, len(self.order), self.ms)


def _values(w, ids):
    out = {}
    for i, o in enumerate(ids):
        c = w.free_slots()
        best = min((w.cost(o, cc) for cc in c), default=0.0)
        out[i] = w.objs[o].score - W.MU * best / 100.0
    return out


def build(w, T, lam=None):
    ids = list(range(w.n_obj))
    ids = [o for o in ids if not w.objs[o].gone]
    M = len(ids)
    lam = 2.0 * T if lam is None else lam
    S = T + 1
    n = M * S

    def vid(i, s):
        return i * S + s

    val = _values(w, ids)
    blocked_idx = {i: {ids.index(b) for b in w.blocked[o] if not w.objs[b].gone}
                   for i, o in enumerate(ids)}
    q = {}
    off = 0.0

    def add(a, b, v):
        if a > b:
            a, b = b, a
        q[(a, b)] = q.get((a, b), 0.0) + v

    for i in range(M):
        for s in range(T):
            add(vid(i, s), vid(i, s), -val[i])
        idx = [vid(i, s) for s in range(S)]
        for a in idx:
            add(a, a, -lam)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], 2.0 * lam)
        off += lam
    for s in range(T):
        idx = [vid(i, s) for i in range(M)]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], lam)
    for i in range(M):
        for j in blocked_idx[i]:
            for s in range(T):
                a = vid(i, s)
                for t in range(s, S):
                    add(a, vid(j, t), lam)
    return {"q": q, "offset": off, "n": n, "M": M, "T": T, "ids": ids, "lam": lam}


def decode(pack, x):
    M, T, S = pack["M"], pack["T"], pack["T"] + 1
    order = []
    for i in range(M):
        ones = [s for s in range(S) if x[i * S + s]]
        if len(ones) != 1:
            return None
        if ones[0] < T:
            order.append((ones[0], i))
    order.sort()
    return [i for _, i in order]


def score_of(w, pack, order):
    ids = pack["ids"]
    free = w.free_slots()
    done, total, used = set(), 0.0, []
    for i in order:
        o = ids[i]
        if not free:
            break
        c = min(free, key=lambda cc: w.cost(o, cc))
        free.remove(c)
        total += w.value(o, c)
        used.append((o, c))
        done.add(o)
    return total, used


def exact_milp(w, T, time_limit=3.0):
    if pulp is None:
        raise RuntimeError("pulp missing")
    pack = build(w, T)
    ids, M, S = pack["ids"], pack["M"], pack["T"] + 1
    t0 = time.perf_counter()
    prob = pulp.LpProblem("win", pulp.LpMaximize)
    y = {(i, s): prob.add_variable("y_%d_%d" % (i, s), cat=pulp.LpBinary)
         for i in range(M) for s in range(S)}
    val = _values(w, ids)
    prob += pulp.lpSum(val[i] * y[(i, s)] for i in range(M) for s in range(T))
    for i in range(M):
        prob += pulp.lpSum(y[(i, s)] for s in range(S)) == 1
    for s in range(T):
        prob += pulp.lpSum(y[(i, s)] for i in range(M)) <= 1
    for i, o in enumerate(ids):
        for b in w.blocked[o]:
            if w.objs[b].gone:
                continue
            j = ids.index(b)
            for s in range(T):
                prob += y[(i, s)] + pulp.lpSum(y[(j, t)] for t in range(s, S)) <= 1
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    except Exception:
        pass
    x = [0] * pack["n"]
    for (i, s), v in y.items():
        if pulp.value(v) and pulp.value(v) > 0.5:
            x[i * S + s] = 1
    order = decode(pack, x) or []
    sc, _ = score_of(w, pack, order)
    return Plan(ids, order, 1000.0 * (time.perf_counter() - t0), sc, "milp")


def greedy(w, T, unlock=0.0, cut=True):
    t0 = time.perf_counter()
    pack = build(w, T)
    ids = pack["ids"]
    done = {i for i in range(w.n_obj) if w.objs[i].gone}
    order, free = [], w.free_slots()
    for _ in range(T):
        best, best_v = None, None
        for i, o in enumerate(ids):
            if o in done or (w.blocked[o] - done):
                continue
            if not free:
                break
            c = min(free, key=lambda cc: w.cost(o, cc))
            v = w.value(o, c) + unlock * len(w.blocked[o])
            if cut and v <= 0:
                continue
            if best_v is None or v > best_v:
                best, best_v, best_c = i, v, c
        if best is None:
            break
        order.append(best)
        done.add(ids[best])
        free.remove(best_c)
    sc, _ = score_of(w, pack, order)
    return Plan(ids, order, 1000.0 * (time.perf_counter() - t0), sc, "greedy")


def anneal(w, T, num_reads=30, num_sweeps=600, seed=0):
    if dimod is None:
        raise RuntimeError("dimod missing")
    pack = build(w, T)
    qd = dict(pack["q"])
    for i in range(pack["n"]):
        qd.setdefault((i, i), 0.0)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qd, offset=pack["offset"])
    t0 = time.perf_counter()
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    ms = 1000.0 * (time.perf_counter() - t0)
    best, best_sc = [], -1.0
    for row in ss.record:
        order = decode(pack, list(row[0]))
        if order is None:
            continue
        sc, _ = score_of(w, pack, order)
        if sc > best_sc:
            best, best_sc = order, sc
    return Plan(pack["ids"], best, ms, max(best_sc, 0.0), "sa")


PLANNERS = {"milp": exact_milp, "greedy": greedy, "sa": anneal,
            "greedy_all": lambda w, T: greedy(w, T, cut=False)}
