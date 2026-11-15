"""把顺序写进 QUBO：变量 z[动作, 步]，工具切换是相邻两步的二次项。

对应 Harness VLA 的编排层：规划器输出的是一串原语调用（先取谁、再取谁），
顺序本身就是决策的一部分。
"""
import statistics

import dimod
import dwave.samplers as ds

import backend
import sim
import world

LAM = 3.0      # 约束罚
MU = 0.15      # 换工具罚，默认值；实际用 _mu(w) 按时间成本折算


def _mu(w):
    """换一次工具的时间，折成多少个动作的收益。"""
    ratio = world.T_TOOL / world.T_ACTION
    return 0.10 * ratio


def build(w, S=4, w_stale=0.5):
    acts = [(pi, si) for pi, si, v, t, stale, ok in backend._pairs(w) if ok]
    if not acts:
        return None
    A = len(acts)
    info = []
    for pi, si in acts:
        p = w.parts[pi]
        s = w.slots[si]
        val = p.value - w_stale * (1.0 if s.stale else 0.0)
        info.append((pi, si, p.tool, val))
    q = {}

    def add(i, j, v):
        if i > j:
            i, j = j, i
        q[(i, j)] = q.get((i, j), 0.0) + v

    for a in range(A):
        pi, si, tool, val = info[a]
        for s in range(S):
            add(a * S + s, a * S + s, -val)
    for a in range(A):
        for s in range(S):
            for t in range(s + 1, S):
                add(a * S + s, a * S + t, LAM)
    for s in range(S):
        for a in range(A):
            for b in range(a + 1, A):
                if info[a][0] == info[b][0] or info[a][1] == info[b][1]:
                    add(a * S + s, b * S + s, LAM)
                else:
                    add(a * S + s, b * S + s, 2.0 * LAM)
    for s in range(S - 1):
        for a in range(A):
            for b in range(A):
                if info[a][2] != info[b][2]:
                    add(a * S + s, b * S + s + 1, _mu(w))
                else:
                    add(a * S + s, b * S + s + 1, -world.W_STREAK)
    return q, info, A, S


class SeqAnneal(backend.Backend):
    name = "seq"

    def __init__(self, reads=80, sweeps=900, S=4, seed=0, w_stale=0.5):
        self.reads = reads
        self.sweeps = sweeps
        self.S = S
        self.seed = seed
        self.w_stale = w_stale

    def plan(self, w, T=6, h=2):
        built = build(w, S=min(self.S, T), w_stale=self.w_stale)
        if not built:
            return []
        q, info, A, S = built
        bqm = dimod.BinaryQuadraticModel.from_qubo(q)
        ss = ds.SimulatedAnnealingSampler().sample(
            bqm, num_reads=self.reads, num_sweeps=self.sweeps, seed=self.seed)
        best, best_v = [], -1e9
        for sample, _e in ss.data(["sample", "energy"]):
            used_p, used_s, seq, tools = set(), set(), [], []
            for s in range(S):
                for a in range(A):
                    if not sample[a * S + s]:
                        continue
                    pi, si, tool, val = info[a]
                    if pi in used_p or si in used_s:
                        continue
                    used_p.add(pi)
                    used_s.add(si)
                    seq.append((pi, si))
                    tools.append(tool)
            if not seq:
                continue
            v = sum(info[[i for i, x in enumerate(info) if (x[0], x[1]) == a][0]][3] for a in seq)
            sw = sum(1 for i in range(1, len(tools)) if tools[i] != tools[i - 1])
            v -= _mu(w) * sw
            v += world.W_STREAK * (len(tools) - 1 - sw)
            if v > best_v:
                best, best_v = seq, v
        return best


NAMES = ["greedy_safe", "greedy_tool", "risk", "seq"]


def main():
    for h in (1, 2, 3):
        row = []
        for n in NAMES:
            sc, sp, ck = [], [], []
            for s in range(5):
                w = world.World(n_part=12, n_slot=8, seed=s, p_stale=0.55,
                                q_correct=0.55, obs_cost=0.25)
                b = SeqAnneal() if n == "seq" else backend.make(n)
                r = sim.episode(w, b, h=h, budget_s=18.0, seed=100 + s)
                sc.append(r["score"])
                sp.append(r["spoiled"])
                ck.append(r["clock"])
            row.append("%-12s %5.2f/%3.1f" % (n, statistics.mean(sc), statistics.mean(sp)))
        print("h=%d  " % h + " | ".join(row))


if __name__ == "__main__":
    main()