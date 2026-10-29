"""Benchmark: exact vs greedy vs classical annealing vs quantum-annealing simulation."""
from __future__ import annotations

import argparse
import csv
import json
import os

import qubo
import scene as S
import solvers as V

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GRID_PILES = [(2, 3), (3, 3), (4, 3), (3, 4), (5, 3), (4, 4), (6, 4)]
GRID_CLUTTER = [6, 9, 12, 15, 18, 24]


def make(family: str, size, seed: int, T: int):
    if family == "piles":
        cl, pc = size
        return S.make_pile_scene(n_clusters=cl, per_cluster=pc, seed=seed, budget=T)
    return S.make_scene(n_obj=size, seed=seed, max_overlap=0.45, budget=T)


def run_case(family: str, size, seed: int, frac: float, reads_sa: int, sweeps_sa: int,
             reads_sqa: int, sweeps_sqa: int, replicas: int) -> list[dict]:
    sc = make(family, size, seed, 1)
    M = sc.n_obj
    T = max(1, int(round(frac * M)))
    sc = make(family, size, seed, T)
    pack = qubo.build(sc, T)
    rows = []

    milp = V.exact_milp(sc, T, 120.0)
    best = max(milp.score, 1e-9)
    greedy = V.greedy(sc, T)
    sa = V.anneal_classical(sc, pack, num_reads=reads_sa, num_sweeps=sweeps_sa, seed=seed)
    sqa = V.anneal_quantum(sc, pack, num_reads=reads_sqa, num_sweeps=sweeps_sqa,
                           replicas=replicas, seed=seed)

    case = "{}_M{}_T{}_s{}".format(family, M, T, seed)
    for res in (milp, greedy, sa, sqa):
        rows.append({
            "case": case,
            "family": family,
            "M": M,
            "T": T,
            "n_var": pack.n,
            "nnz": pack.nnz,
            "lam": pack.lam,
            "solver": res.solver,
            "score": round(res.score, 6),
            "feasible": int(res.feasible),
            "plan_len": len(res.seq),
            "time_ms": round(res.time_ms, 3),
            "gap_pct": round((best - res.score) / best * 100.0, 4),
            "infeasible_rate": round(float(res.extra.get("infeasible_rate", 0.0)), 4),
        })
    print("{:<22} n_var={:<4} nnz={:<5} | milp {:6.2f} {:7.1f} ms | greedy {:6.2f} {:5.1f}% | "
          "sa {:6.2f} {:5.1f}% {:7.1f} ms | sqa {:6.2f} {:5.1f}% {:8.1f} ms".format(
              case, pack.n, pack.nnz, milp.score, milp.time_ms, greedy.score,
              (best - greedy.score) / best * 100, sa.score, (best - sa.score) / best * 100,
              sa.time_ms, sqa.score, (best - sqa.score) / best * 100, sqa.time_ms))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="both", choices=["piles", "clutter", "both"])
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reads-sa", type=int, default=40)
    ap.add_argument("--sweeps-sa", type=int, default=1000)
    ap.add_argument("--reads-sqa", type=int, default=16)
    ap.add_argument("--sweeps-sqa", type=int, default=400)
    ap.add_argument("--replicas", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    plans = []
    if args.family in ("piles", "both"):
        plans += [("piles", s) for s in GRID_PILES]
    if args.family in ("clutter", "both"):
        plans += [("clutter", s) for s in GRID_CLUTTER]
    if args.quick:
        plans = plans[:3]

    rows: list[dict] = []
    for family, size in plans:
        for seed in range(args.seeds):
            rows += run_case(family, size, seed, args.frac, args.reads_sa, args.sweeps_sa,
                             args.reads_sqa, args.sweeps_sqa, args.replicas)

    tag = args.family + ("_quick" if args.quick else "")
    csv_path = os.path.join(RES, "bench_{}.csv".format(tag))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for solver in ("milp", "greedy", "sa", "sqa"):
        sel = [r for r in rows if r["solver"] == solver]
        summary[solver] = {
            "cases": len(sel),
            "mean_gap_pct": round(sum(r["gap_pct"] for r in sel) / len(sel), 4),
            "max_gap_pct": round(max(r["gap_pct"] for r in sel), 4),
            "mean_time_ms": round(sum(r["time_ms"] for r in sel) / len(sel), 3),
            "mean_infeasible_rate": round(sum(r["infeasible_rate"] for r in sel) / len(sel), 4),
        }
    with open(os.path.join(RES, "summary_{}.json".format(tag)), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n== summary ({} cases) ==".format(len(rows) // 4))
    for k, v in summary.items():
        print("{:<7} mean_gap={:6.2f}%  max_gap={:6.2f}%  mean_time={:9.1f} ms  infeasible={:.1%}".format(
            k, v["mean_gap_pct"], v["max_gap_pct"], v["mean_time_ms"], v["mean_infeasible_rate"]))
    print("\nwrote", csv_path)


if __name__ == "__main__":
    main()
