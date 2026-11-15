"""四种编排后端，共用同一份 plan(state, T, h) 接口。"""
import pulp
import dimod
import dwave.samplers as ds

from world import World, KINDS


class Backend:
    name = "base"

    def plan(self, w, T=6, h=2):
        raise NotImplementedError


def _pairs(w, use_belief=True):
    """所有可执行的 (part, slot)，用计划器信念判断槽位要求。"""
    out = []
    for p in w.bin_parts():
        for s in w.free_slots():
            need = s.belief if use_belief else s.kind
            out.append((p.i, s.i, p.value, p.tool, s.stale, need == p.kind))
    return out


class Greedy(Backend):
    name = "greedy"

    def __init__(self, w_stale=0.0):
        self.w_stale = w_stale

    def plan(self, w, T=6, h=2):
        cand = [c for c in _pairs(w) if c[5]]
        cand.sort(key=lambda c: -(c[2] - self.w_stale * (1 if c[4] else 0)))
        used_p, used_s = set(), set()
        seq = []
        for pi, si, v, tool, stale, ok in cand:
            if pi in used_p or si in used_s:
                continue
            seq.append((pi, si))
            used_p.add(pi)
            used_s.add(si)
            if len(seq) >= T:
                break
        return seq


class GreedySafe(Greedy):
    name = "greedy_safe"

    def plan(self, w, T=6, h=2):
        cand = [c for c in _pairs(w) if c[5] and not c[4]]
        cand.sort(key=lambda c: -c[2])
        used_p, used_s = set(), set()
        seq = []
        for pi, si, v, tool, stale, ok in cand:
            if pi in used_p or si in used_s:
                continue
            seq.append((pi, si))
            used_p.add(pi)
            used_s.add(si)
            if len(seq) >= T:
                break
        return seq


def _model(w, T, w_stale):
    xy = _pairs(w)
    if not xy:
        return None, xy
    weight = {}
    for pi, si, v, tool, stale, ok in xy:
        if not ok:
            continue
        weight[(pi, si)] = v - w_stale * (1.0 if stale else 0.0)
    if not weight:
        return None, xy
    return weight, xy


class Milp(Backend):
    name = "milp"

    def __init__(self, w_stale=0.35):
        self.w_stale = w_stale

    def plan(self, w, T=6, h=2):
        weight, xy = _model(w, T, self.w_stale)
        if not weight:
            return []
        prob = pulp.LpProblem("plan", pulp.LpMaximize)
        x = {k: prob.add_variable("x_%d_%d" % k, cat=pulp.LpBinary) for k in weight}
        prob += pulp.lpSum(weight[k] * v for k, v in x.items())
        for pi in {k[0] for k in weight}:
            prob += pulp.lpSum(v for k, v in x.items() if k[0] == pi) <= 1
        for si in {k[1] for k in weight}:
            prob += pulp.lpSum(v for k, v in x.items() if k[1] == si) <= 1
        prob += pulp.lpSum(x.values()) <= T
        prob.solve(pulp.HiGHS(msg=False, timeLimit=20))
        seq = [k for k, v in x.items() if v.value() and v.value() > 0.5]
        seq.sort(key=lambda k: -weight[k])
        return seq


class Anneal(Backend):
    name = "anneal"

    def __init__(self, reads=60, sweeps=800, w_stale=0.35, seed=0):
        self.reads = reads
        self.sweeps = sweeps
        self.w_stale = w_stale
        self.seed = seed

    def plan(self, w, T=6, h=2):
        weight, xy = _model(w, T, self.w_stale)
        if not weight:
            return []
        keys = list(weight)
        idx = {k: i for i, k in enumerate(keys)}
        n = len(keys) + T
        LAM = 3.0
        q = {}
        for k, i in idx.items():
            q[(i, i)] = q.get((i, i), 0.0) - weight[k]
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka, kb = keys[a], keys[b]
                if ka[0] == kb[0] or ka[1] == kb[1]:
                    q[(a, b)] = q.get((a, b), 0.0) + LAM
        aux0 = len(keys)
        for a in range(len(keys)):
            for t in range(T):
                q[(a, aux0 + t)] = q.get((a, aux0 + t), 0.0) - 2.0 * LAM
        for t in range(T):
            q[(aux0 + t, aux0 + t)] = q.get((aux0 + t, aux0 + t), 0.0) + LAM
            for t2 in range(t + 1, T):
                q[(aux0 + t, aux0 + t2)] = q.get((aux0 + t, aux0 + t2), 0.0) + 2.0 * LAM
        bqm = dimod.BinaryQuadraticModel.from_qubo(q, offset=LAM * T)
        sampler = ds.SimulatedAnnealingSampler()
        ss = sampler.sample(bqm, num_reads=self.reads, num_sweeps=self.sweeps, seed=self.seed)
        best, best_v = [], -1e9
        for sample, _e in ss.data(["sample", "energy"]):
            sel = [keys[i] for i in range(len(keys)) if sample[i]]
            used_p, used_s, seq = set(), set(), []
            for pi, si in sel:
                if pi in used_p or si in used_s:
                    continue
                used_p.add(pi)
                used_s.add(si)
                seq.append((pi, si))
                if len(seq) >= T:
                    break
            v = sum(weight[k] for k in seq)
            if v > best_v:
                best, best_v = seq, v
        return best


class GreedyTool(Backend):
    """按 价值 − 换工具代价 逐步选，顺序在这里才起作用。"""
    name = "greedy_tool"

    def __init__(self, lam=0.20, w_stale=0.30):
        self.lam = lam
        self.w_stale = w_stale

    def plan(self, w, T=6, h=2):
        used_p, used_s = set(), set()
        tool = w.tool
        seq = []
        while len(seq) < T:
            best, best_v = None, -1e9
            for pi, si, v, t, stale, ok in _pairs(w):
                if not ok or pi in used_p or si in used_s:
                    continue
                g = v - self.lam * (0.0 if t == tool else 1.0) - self.w_stale * (1.0 if stale else 0.0)
                if g > best_v:
                    best, best_v = (pi, si, t), g
            if best is None:
                break
            seq.append((best[0], best[1]))
            used_p.add(best[0])
            used_s.add(best[1])
            tool = best[2]
        return seq


class RiskAware(GreedyTool):
    """对过期槽位算期望净收益，负期望就不下注。"""
    name = "risk"

    def __init__(self, p_wrong=0.45, lam=0.20, spoil=0.60):
        GreedyTool.__init__(self, lam=lam, w_stale=0.0)
        self.p_wrong = p_wrong
        self.spoil = spoil

    def _gain(self, v, stale):
        if not stale:
            return v
        p_ok = 1.0 - self.p_wrong
        return p_ok * v - (1.0 - p_ok) * (self.spoil + v)

    def plan(self, w, T=6, h=2):
        used_p, used_s = set(), set()
        tool = w.tool
        seq = []
        while len(seq) < T:
            best, best_v = None, -1e9
            for pi, si, v, t, stale, ok in _pairs(w):
                if not ok or pi in used_p or si in used_s:
                    continue
                g = self._gain(v, stale) - self.lam * (0.0 if t == tool else 1.0)
                if g > best_v:
                    best, best_v = (pi, si, t), g
            if best is None:
                break
            seq.append((best[0], best[1]))
            used_p.add(best[0])
            used_s.add(best[1])
            tool = best[2]
        return seq


BACKENDS = {"greedy": Greedy, "greedy_safe": GreedySafe, "greedy_tool": GreedyTool,
            "risk": RiskAware, "milp": Milp, "anneal": Anneal}


def make(name, **kw):
    return BACKENDS[name](**kw)