"""Four ways to answer the same question: exact MILP, greedy, SA, and the QUBO
round trip through a sampler."""

import contextlib
import io
import time

import place
import quboplace

try:
    import pulp
except Exception:                                    # pragma: no cover
    pulp = None
try:
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
except Exception:                                    # pragma: no cover
    dimod = None


class Result:
    def __init__(self, name, score, placed, ms, info=None):
        self.name = name
        self.score = score
        self.placed = placed
        self.ms = ms
        self.info = info or {}

    def __repr__(self):
        return "<%s score=%.4f n=%d %.2fms>" % (self.name, self.score, len(self.placed), self.ms)


def exact_milp(sc, T=None, time_limit=30.0):
    if pulp is None:
        raise RuntimeError("pulp not installed")
    N, K = sc.n_obj, sc.n_slot
    if T is None:
        T = max(1, min(K, N // 2))
    t0 = time.perf_counter()
    prob = pulp.LpProblem("place", pulp.LpMaximize)
    y, sk = {}, {}
    for o in range(N):
        sk[o] = prob.add_variable("k_%d" % o, cat=pulp.LpBinary)
        for c in range(K):
            if not sc.fit(o, c):
                continue
            for s in range(T):
                y[(o, c, s)] = prob.add_variable("y_%d_%d_%d" % (o, c, s), cat=pulp.LpBinary)
    prob += pulp.lpSum(sc.value(o, c) * v for (o, c, s), v in y.items())
    for o in range(N):
        prob += pulp.lpSum(v for (oo, c, s), v in y.items() if oo == o) + sk[o] == 1
    for s in range(T):
        prob += pulp.lpSum(v for (o, c, ss), v in y.items() if ss == s) <= 1
    for c in range(K):
        prob += pulp.lpSum(v for (o, cc, s), v in y.items() if cc == c) <= 1
    for o in range(N):
        for blk in sc.blocked[o]:
            for c in range(K):
                for s in range(T):
                    if (o, c, s) not in y:
                        continue
                    late = pulp.lpSum(v for (o2, c2, t), v in y.items() if o2 == blk and t >= s)
                    prob += y[(o, c, s)] + late + sk[blk] <= 1
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    placed = []
    for (o, c, s), v in y.items():
        if pulp.value(v) and pulp.value(v) > 0.5:
            placed.append((o, c, s))
    placed.sort(key=lambda t: t[2])
    score = sum(sc.value(o, c) for o, c, s in placed)
    return Result("milp", score, placed, 1000.0 * (time.perf_counter() - t0),
                  {"status": str(getattr(stats, "status", ""))})


def greedy(sc, T=None, unlock=0.0):
    N, K = sc.n_obj, sc.n_slot
    if T is None:
        T = max(1, min(K, N // 2))
    t0 = time.perf_counter()
    placed, done, used = [], set(), set()
    for s in range(T):
        best, best_c, best_v = None, None, None
        for o in range(N):
            if o in done or (sc.blocked[o] - done):
                continue
            c = sc.best_slot(o, used)
            if c is None:
                continue
            v = sc.value(o, c) + unlock * len(sc.blocked[o])
            if v <= 0.0:
                continue
            if best_v is None or v > best_v:
                best, best_c, best_v = o, c, v
        if best is None:
            break
        placed.append((best, best_c, s))
        done.add(best)
        used.add(best_c)
    score = sum(sc.value(o, c) for o, c, s in placed)
    return Result("greedy", score, placed, 1000.0 * (time.perf_counter() - t0))


def _to_bqm(pack):
    qd = {k: v for k, v in pack.q.items()}
    for i in range(pack.n):
        qd.setdefault((i, i), 0.0)
    return dimod.BinaryQuadraticModel.from_qubo(qd, offset=pack.offset)


def anneal(sc, T=None, num_reads=40, num_sweeps=1000, seed=0, label="sa"):
    if dimod is None:
        raise RuntimeError("dimod not installed")
    pack = quboplace.build(sc, T=T)
    bqm = _to_bqm(pack)
    t0 = time.perf_counter()
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    ms = 1000.0 * (time.perf_counter() - t0)
    best, bad = None, 0
    for row in ss.record:
        x = list(row[0])
        info = quboplace.decode(sc, pack, x)
        if not info["feasible"]:
            bad += 1
            continue
        if best is None or info["score"] > best["score"]:
            best = info
    if best is None:
        return Result(label, 0.0, [], ms, {"infeasible_reads": bad, "reads": len(ss.record)})
    placed = [(o, c, s) for o, c, s in best["placed"]]
    return Result(label, best["score"], placed, ms,
                  {"infeasible_reads": bad, "reads": len(ss.record), "nnz": pack.nnz, "n_var": pack.n})
