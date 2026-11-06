import random, itertools

def make(n_col=4, per_col=3, seed=0):
    rng = random.Random(seed)
    return {
        "n_col": n_col, "per_col": per_col,
        "x": [rng.uniform(0, 100) for _ in range(n_col)],
        "val": [[rng.uniform(0.3, 1.0) for _ in range(per_col)] for _ in range(n_col)],
    }

def col_reward(w, c):
    return sum(w["val"][c])

def feas_items(w, taken):
    out = []
    for c in range(w["n_col"]):
        for l in range(w["per_col"]):
            if (c, l) in taken:
                continue
            if all((c, m) in taken for m in range(l)):
                out.append((c, l))
    return out

def greedy(w, budget, objective):
    taken = set(); score = 0.0; travel = 0.0; last_x = 0.0
    for _ in range(budget):
        best = None
        for it in feas_items(w, taken):
            v = objective["val"](w, it, taken, last_x)
            if best is None or v > best[0]:
                best = (v, it)
        if best is None:
            break
        _, (c, l) = best
        taken.add((c, l)); travel += abs(w["x"][c] - last_x); last_x = w["x"][c]
        score += objective["reward"](w, c, l, taken)
    return score, travel, len(taken)

def optimal(w, budget, objective):
    ncol, per = w["n_col"], w["per_col"]
    if objective["kind"] == "column":
        k = budget // per
        cols = sorted(range(ncol), key=lambda c: -col_reward(w, c))[:k]
        return sum(col_reward(w, c) for c in cols), 0.0, k * per
    if objective["kind"] == "value":
        items = sorted(((w["val"][c][l], c, l) for c in range(ncol) for l in range(per)), reverse=True)
        taken = set(); score = 0.0
        for v, c, l in items:
            if len(taken) >= budget:
                break
            if all((c, m) in taken for m in range(l)):
                taken.add((c, l)); score += v
        return score, 0.0, len(taken)
    if objective["kind"] == "distance":
        best_t = 1e18; best_seq = None
        for order in itertools.permutations(range(ncol)):
            t = 0.0; last = 0.0
            for c in order:
                t += abs(w["x"][c] - last); last = w["x"][c]
            if t < best_t:
                best_t = t; best_seq = order
        return 0.0, best_t, ncol * per
    if objective["kind"] == "risk":
        p = objective["penalty"]
        best = -1e18; best_seq = None
        for order in itertools.permutations(range(ncol)):
            cleared = set(); s = 0.0
            for c in order:
                neigh = any(d in cleared and d != c for d in (c-1, c+1) if 0 <= d < ncol)
                # risk: clearing c while a neighbor still has items -> penalty
                neigh_present = any(d not in cleared and d != c for d in (c-1, c+1) if 0 <= d < ncol)
                s += col_reward(w, c) - (p if neigh_present else 0.0)
                cleared.add(c)
            if s > best:
                best = s; best_seq = order
        return best, 0.0, ncol * per
    return 0.0, 0.0, 0

def greedy_risk(w, budget, p):
    taken = set(); cleared = set(); s = 0.0
    for _ in range(budget):
        best = None
        for it in feas_items(w, taken):
            c, l = it
            neigh_present = any(d not in cleared and d != c for d in (c-1, c+1) if 0 <= d < w["n_col"])
            v = w["val"][c][l]
            if best is None or v > best[0]:
                best = (v, it)
        if best is None:
            break
        _, (c, l) = best
        taken.add((c, l))
        if all((c, m) in taken for m in range(w["per_col"])):
            neigh_present = any(d not in cleared and d != c for d in (c-1, c+1) if 0 <= d < w["n_col"])
            s += col_reward(w, c) - (p if neigh_present else 0.0)
            cleared.add(c)
    return s

OBJS = {
    "value":    {"kind": "value",    "val": lambda w,i,t,x: w["val"][i[0]][i[1]], "reward": lambda w,c,l,t: w["val"][c][l]},
    "column":   {"kind": "column",   "val": lambda w,i,t,x: w["val"][i[0]][i[1]], "reward": lambda w,c,l,t: (col_reward(w,c) if all((c,m) in t for m in range(w["per_col"])) else 0.0)},
    "distance": {"kind": "distance", "val": lambda w,i,t,x: -abs(w["x"][i[0]]-x),        "reward": lambda w,c,l,t: 0.0},
}
print(f"{'mode':9} {'seed':4} {'greedy':>8} {'optimal':>8} {'gap%':>7}")
for mode, obj in OBJS.items():
    for seed in range(5):
        w = make(seed=seed)
        budget = 2 * w["per_col"]
        gs, gt, gn = greedy(w, budget, obj)
        os_, ot, on_ = optimal(w, budget, obj)
        gap = 0.0 if os_ == 0 else 100.0 * (os_ - gs) / os_
        print(f"{mode:9} {seed:4} {gs:8.3f} {os_:8.3f} {gap:6.1f}%")
# risk mode (separate greedy)
print("--- risk (penalty=0.4, full clear) ---")
for seed in range(5):
    w = make(seed=seed)
    budget = w["n_col"] * w["per_col"]
    g = greedy_risk(w, budget, 0.4)
    o, _, _ = optimal(w, budget, {"kind": "risk", "penalty": 0.4})
    gap = 0.0 if o == 0 else 100.0 * (o - g) / abs(o)
    print(f"risk     {seed:4} {g:8.3f} {o:8.3f} {gap:6.1f}%")
