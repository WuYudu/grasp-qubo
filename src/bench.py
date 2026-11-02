"""Size sweep: exact vs greedy vs annealer on the pick-and-place model."""

import argparse
import csv
import os
import statistics

import place
import quboplace
import solvers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def one(n_obj, seed, n_slot, T, reads, sweeps):
    sc = place.make_scene(n_obj=n_obj, seed=seed, cluster=3, n_slot=n_slot)
    pack = quboplace.build(sc, T=T)
    m = solvers.exact_milp(sc, T=T, time_limit=30.0)
    g = solvers.greedy(sc, T=T)
    a = solvers.anneal(sc, T=T, num_reads=reads, num_sweeps=sweeps, seed=seed)
    return {
        "n_obj": n_obj, "seed": seed, "n_slot": n_slot, "T": T,
        "n_var": pack.n, "nnz": pack.nnz,
        "milp": m.score, "milp_ms": m.ms, "milp_n": len(m.placed),
        "greedy": g.score, "greedy_ms": g.ms, "greedy_n": len(g.placed),
        "sa": a.score, "sa_ms": a.ms, "sa_n": len(a.placed),
        "sa_bad": a.info.get("infeasible_reads", 0), "reads": a.info.get("reads", 0),
        "gap_greedy": 100.0 * (m.score - g.score) / m.score if m.score else 0.0,
        "gap_sa": 100.0 * (m.score - a.score) / m.score if m.score else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", type=int, default=[8, 12, 16, 20, 24])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--reads", type=int, default=40)
    ap.add_argument("--sweeps", type=int, default=800)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in args.objects:
        for sd in range(args.seeds):
            r = one(n, sd, args.slots, args.steps, args.reads, args.sweeps)
            rows.append(r)
            print("N=%-3d seed=%d var=%-4d nnz=%-6d milp=%.4f(%5.0fms) greedy=%.4f(%.1f%%) "
                  "sa=%.4f(%.1f%%,bad%d)" % (r["n_obj"], r["seed"], r["n_var"], r["nnz"],
                                             r["milp"], r["milp_ms"], r["greedy"],
                                             r["gap_greedy"], r["sa"], r["gap_sa"], r["sa_bad"]),
                  flush=True)
    path = os.path.join(OUT, "bench_%s.csv" % args.tag)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", path)
    for key in ("gap_greedy", "gap_sa", "milp_ms", "greedy_ms", "sa_ms"):
        vals = [r[key] for r in rows]
        print("%-12s mean %8.3f  max %8.3f" % (key, statistics.mean(vals), max(vals)))


if __name__ == "__main__":
    main()
