"""Fixed best pose vs joint pose+order, both solved exactly.

    python src/bench_cand.py --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os

import qubo_cand as QC
import scene as S
import solvers

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

POSES = ((-0.26, 0.0), (0.0, 0.0), (0.26, 0.0), (0.0, 32.0))
GRID = [6, 9, 12, 15, 18]


def best_pose(sc, k=3):
    return [max(range(k), key=lambda c: sc.grasps[o][c].score) for o in range(sc.n_obj)]


def fixed_solve(sc, T, k=3):
    bp = best_pose(sc, k)
    blocked = [sc.cand_blocked[o][bp[o]] for o in range(sc.n_obj)]
    sc2 = dataclasses.replace(sc, blocked=blocked, chosen=bp)
    res = solvers.exact_milp(sc2, T)
    return res, bp


def run_case(n_obj, T_frac, seed, k=4, overlap=0.62):
    sc = S.make_scene(n_obj=n_obj, seed=seed, max_overlap=overlap, poses=POSES)
    m = sc.n_obj
    T = max(1, int(round(T_frac * m)))
    sc.budget = T

    fx, bp = fixed_solve(sc, T, k)
    jt = QC.exact_milp_cand(sc, T, k=k)
    delta = (jt.score - fx.score) / fx.score * 100.0 if fx.score > 0 else 0.0
    nonbest = sum(1 for o, c in zip(jt.seq, jt.extra["poses"]) if c != bp[o])
    blk_fixed = sum(len(sc.cand_blocked[o][bp[o]]) for o in range(m))
    blk_joint = sum(len(sc.cand_blocked[o][c]) for o, c in zip(jt.seq, jt.extra["poses"]))
    return {
        "case": "clutter-{}".format(n_obj),
        "n_obj": m, "T": T, "n_var_fixed": m * (T + 1), "n_var_cand": m * k * (T + 1),
        "seed": seed,
        "score_fixed": round(fx.score, 4), "score_joint": round(jt.score, 4),
        "delta_pct": round(delta, 2),
        "picked_nonbest": nonbest,
        "n_picked": len(jt.seq),
        "blockers_fixed": blk_fixed, "blockers_joint": blk_joint,
        "ms_fixed": round(fx.time_ms, 1), "ms_joint": round(jt.time_ms, 1),
        "feasible": jt.feasible,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", default="")
    args = ap.parse_args()
    grid = GRID
    if args.grid:
        grid = [int(x) for x in args.grid.split(",")]

    rows = []
    for n_obj in grid:
        for seed in range(args.seeds):
            r = run_case(n_obj, 0.6, seed)
            rows.append(r)
            print("{:<14} seed={} n={:<3} T={:<2} fixed={:.4f} joint={:.4f} "
                  "delta={:+.2f}%  nonbest={}/{}  ms {} / {}".format(
                      r["case"], seed, r["n_obj"], r["T"], r["score_fixed"],
                      r["score_joint"], r["delta_pct"], r["picked_nonbest"],
                      r["n_picked"], r["ms_fixed"], r["ms_joint"]))

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "bench_cand.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n->", path)

    n = len(rows)
    mean_d = sum(r["delta_pct"] for r in rows) / n
    win = sum(1 for r in rows if r["delta_pct"] > 0.05)
    nb = sum(r["picked_nonbest"] for r in rows)
    tot = sum(r["n_picked"] for r in rows)
    print("mean delta {:+.2f}%   instances improved {}/{}   non-best poses {}/{}".format(
        mean_d, win, n, nb, tot))


if __name__ == "__main__":
    main()
