"""扫前缀长度与观测不确定性：时间预算固定，看分数与报废。"""
import argparse
import csv
import statistics

import world
import backend
import sim


def one(p_stale, h, name, seeds=5, budget_s=18.0):
    rows = []
    for s in range(seeds):
        w = world.World(n_part=12, n_slot=8, seed=s, p_stale=p_stale)
        b = backend.make(name)
        rows.append(sim.episode(w, b, h=h, budget_s=budget_s, seed=100 + s))
    keys = ("score", "placed", "spoiled", "actions", "clock", "replans")
    agg = {k: statistics.mean(r[k] for r in rows) for k in keys}
    agg.update(name=name, h=h, p_stale=p_stale)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=18.0)
    ap.add_argument("--out", default="../results/calib.csv")
    a = ap.parse_args()
    out = []
    for p_stale in (0.25, 0.50):
        for h in (1, 2, 4, 6):
            for n in ("greedy", "greedy_safe", "greedy_tool", "risk", "milp", "anneal"):
                r = one(p_stale, h, n, seeds=a.seeds, budget_s=a.budget)
                out.append(r)
                print("stale=%.2f h=%d %-12s score=%6.3f placed=%4.1f spoiled=%4.1f clock=%5.1f replans=%4.1f"
                      % (p_stale, h, n, r["score"], r["placed"], r["spoiled"], r["clock"], r["replans"]))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["p_stale", "h", "name", "score", "placed", "spoiled", "actions", "clock", "replans"])
        wr.writeheader()
        for r in out:
            wr.writerow(r)


if __name__ == "__main__":
    main()