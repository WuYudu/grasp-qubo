"""Static sweep: does the gap between greedy and the optimum survive at shelf scale?"""
import argparse
import csv
import os

import world
import solvers
import qubo

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GRID = [(3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 4), (10, 4), (12, 5)]


def one(n_clusters, per_cluster, T=6, seed=0, kind="piles", t_frac=0.0):
    sc = (world.big_piles(n_clusters, per_cluster, seed=seed) if kind == "piles"
          else world.big_clutter(n_clusters * per_cluster, seed=seed))
    n = sc.n_obj
    Tc = max(2, min(round(t_frac * n) if t_frac else T, n - 1))
    pack = qubo.build(sc, Tc)
    g = solvers.greedy(sc, Tc)
    m = solvers.exact_milp(sc, Tc, time_limit=120.0)
    best = max(1e-9, m.score)
    return dict(kind=kind, n=n, T=Tc, n_var=pack.n, nnz=pack.nnz,
                greedy=g.score, milp=m.score,
                gap=100.0 * (m.score - g.score) / best,
                t_greedy_ms=g.time_ms, t_milp_ms=m.time_ms,
                milp_opt=bool(m.extra.get("optimal", True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="scale")
    ap.add_argument("--t-frac", type=float, default=0.0,
                    help="if >0, window = round(frac*N) instead of a fixed 6")
    ap.add_argument("--grid", default="", help="override: '3x3,4x3,...'")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    grid = GRID
    if args.grid:
        grid = [tuple(int(v) for v in p.split("x")) for p in args.grid.split(",")]
    rows = []
    for kind in ("piles", "clutter"):
        for nc, pc in grid:
            for seed in range(args.seeds):
                try:
                    r = one(nc, pc, seed=seed, kind=kind, t_frac=args.t_frac)
                except Exception as exc:                      # placement can fail
                    print("skip", kind, nc, pc, seed, exc)
                    continue
                r["seed"] = seed
                rows.append(r)
                print("{kind:8s} N={n:3d} T={T:3d} var={n_var:5d} gap={gap:6.2f}% "
                      "greedy={t_greedy_ms:8.3f}ms milp={t_milp_ms:9.1f}ms opt={milp_opt}"
                      .format(**r))
    path = os.path.join(OUT, "bench_{}.csv".format(args.tag))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


if __name__ == "__main__":
    main()
