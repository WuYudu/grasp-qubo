"""Column-batch (non-additive reward) task: closed-loop variant.

Reward is awarded only when an entire column (per_col stacked items) is
cleared within the action budget.  Greedy-by-item spreads and clears nothing;
a column-aware optimizer concentrates.  This file measures, in an open loop,
whether the annealing QUBO reaches the analytical optimum, and how far
item-greedy and column-greedy fall below it.
"""
import random
from dwave.samplers import SimulatedAnnealingSampler


def make(n_col=4, per_col=3, seed=0):
    rng = random.Random(seed)
    items = []
    for c in range(n_col):
        for l in range(per_col):
            items.append((c, l, rng.uniform(0.3, 1.0)))
    return {"n_col": n_col, "per_col": per_col, "items": items}


def col_of(items, c):
    return sorted([i for i in items if i[0] == c], key=lambda t: t[1])


def col_reward(w, c):
    return sum(v for _, _, v in col_of(w["items"], c))


def optimal(w, budget):
    k = budget // w["per_col"]
    cols = sorted(range(w["n_col"]), key=lambda c: -col_reward(w, c))[:k]
    return sum(col_reward(w, c) for c in cols), set(
        (c, l) for c in cols for l in range(w["per_col"]))


def _score(w, sel):
    s = 0.0
    for c in range(w["n_col"]):
        if all((c, l) in sel for l in range(w["per_col"])):
            s += col_reward(w, c)
    return s


def greedy_item(w, budget):
    sel = set()
    for _ in range(budget):
        best = None
        for c, l, v in w["items"]:
            if (c, l) in sel:
                continue
            if all((c, m) in sel for m in range(l)):  # top-down feasible
                if best is None or v > best[0]:
                    best = (v, (c, l))
        if best is None:
            break
        sel.add(best[1])
    return _score(w, sel), sel


def greedy_col(w, budget):
    sel = set()
    left = budget
    done = set()
    while left >= w["per_col"]:
        cand = [c for c in range(w["n_col"]) if c not in done]
        if not cand:
            break
        c = max(cand, key=lambda d: col_reward(w, d))
        for l in range(w["per_col"]):
            sel.add((c, l))
        done.add(c)
        left -= w["per_col"]
    return _score(w, sel), sel


def build(w, budget, B=3.0, LAM=8.0, LAMB=4.0):
    items = w["items"]
    n = len(items)
    nc = w["n_col"]
    idx = {(c, l): i for i, (c, l, _) in enumerate(items)}
    q = {}
    off = 0.0

    def add(a, b, v):
        if a == b:
            q[a] = q.get(a, 0.0) + v
        else:
            x, y = (a, b) if a < b else (b, a)
            q[(x, y)] = q.get((x, y), 0.0) + v

    # column-completion aux y_c: 2B*y - B*y*(sum x)
    for c in range(nc):
        y = n + c
        add(y, y, 2.0 * B)
        for l in range(w["per_col"]):
            add(y, idx[(c, l)], -B)
    # precedence: bottom requires mid requires top
    for c in range(nc):
        for l in range(1, w["per_col"]):
            low = idx[(c, l)]
            hi = idx[(c, l - 1)]
            add(low, low, LAM)
            add(low, hi, -LAM)
    # budget (sum x - T)^2
    T = budget
    for i in range(n):
        add(i, i, LAMB - 2.0 * LAMB * T)
    for i in range(n):
        for j in range(i + 1, n):
            add(i, j, 2.0 * LAMB)
    off += LAMB * T * T
    return q, n, nc, off, idx


def anneal(w, budget, reads=200, sweeps=1000, seed=0):
    q, n, nc, off, idx = build(w, budget)
    bqm = {}
    for k, v in q.items():
        a = k if isinstance(k, int) else k[0]
        b = k if isinstance(k, int) else k[1]
        bqm[(a, b)] = v
    from dimod import BinaryQuadraticModel
    BQ = BinaryQuadraticModel.from_qubo(bqm, offset=off)
    ss = SimulatedAnnealingSampler().sample(BQ, num_reads=reads,
                                            num_sweeps=sweeps, seed=seed)
    inv = {i: (c, l) for (c, l), i in idx.items()}
    val = {(c, l): v for c, l, v in w["items"]}
    best = None
    for rec in ss.record:
        bits = rec.sample
        sel = {inv[i] for i in range(n) if bits[i] > 0.5}
        while len(sel) > budget:
            sel.discard(min(sel, key=lambda t: val[t]))
        changed = True
        while changed:
            changed = False
            for c, l, _ in w["items"]:
                if l > 0 and (c, l) in sel and (c, l - 1) not in sel:
                    sel.discard((c, l))
                    changed = True
        s = _score(w, sel)
        if best is None or s > best[0]:
            best = (s, sel)
    return best


if __name__ == "__main__":
    print(f"{'n_col':5} {'budget':6} {'optimal':8} {'greedy_item':11} "
          f"{'greedy_col':10} {'anneal':8} {'gap%':6}")
    for n_col in (4, 5, 6):
        for seed in range(6):
            w = make(n_col=n_col, seed=seed)
            budget = int(0.6 * n_col * w["per_col"])
            o, _ = optimal(w, budget)
            gi, _ = greedy_item(w, budget)
            gc, _ = greedy_col(w, budget)
            a, _ = anneal(w, budget, seed=seed)
            gap = 0.0 if o == 0 else 100.0 * (o - gi) / o
            print(f"{n_col:5} {budget:6} {o:8.3f} {gi:11.3f} "
                  f"{gc:10.3f} {a:8.3f} {gap:6.1f}%")
