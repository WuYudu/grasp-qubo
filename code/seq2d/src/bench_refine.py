"""Three-way comparison: fixed best pose / joint discrete pose / joint + local refine.

    python src/bench_refine.py --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import os

import bench_cand as B
import geom
import qubo_cand as QC
import scene as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
BLOCK_EPS = 0.12

FINE = ((-0.10, 0.0), (-0.05, 0.0), (0.05, 0.0), (0.10, 0.0),
        (0.0, -12.0), (0.0, 12.0), (-0.05, 12.0), (0.05, -12.0))


def _blocks(sc, obj, grasp):
    return {p for p, other in enumerate(sc.bodies)
            if p != obj and geom.ios(grasp.poly, other.poly) > BLOCK_EPS}


def refine(sc, seq, poses):
    """Local continuous polish: move each chosen pose a little, accept only if
    it does not add a blocker and the score goes up.  The order stays valid."""
    total = 0.0
    detail = []
    for o, c in zip(seq, poses):
        fx, fr = B.POSES[c]
        current = sc.cand_blocked[o][c]
        best_g, best_b = sc.grasps[o][c], current
        grid = tuple((fx + dx, fr + dr) for dx, dr in FINE)
        cands = S._candidates(sc.bodies[o], o, sc.quality[o], S.FINGER,
                              S.MAX_OPENING, poses=grid)
        for g in cands:
            blk = _blocks(sc, o, g)
            if blk <= current and g.score > best_g.score:
                best_g, best_b = g, blk
        total += best_g.score
        detail.append((o, round(best_g.score, 4), len(best_b)))
    return total, detail


def run(family, n_obj, seed, T_frac=0.6):
    if family == "clutter":
        sc = S.make_scene(n_obj=n_obj, seed=seed, max_overlap=0.62, poses=B.POSES)
        case = "clutter-{}".format(n_obj)
    else:
        per = 3
        sc = S.make_pile_scene(n_clusters=max(1, n_obj // per), per_cluster=per,
                               seed=seed, poses=B.POSES)
        case = "piles-{}".format(n_obj)
    T = max(1, int(round(T_frac * sc.n_obj)))
    sc.budget = T
    fx, _ = B.fixed_solve(sc, T, 4)
    jt = QC.exact_milp_cand(sc, T, k=4)
    rf, detail = refine(sc, jt.seq, jt.extra["poses"])
    return {
        "case": case, "family": family, "n_obj": sc.n_obj, "T": T, "seed": seed,
        "score_fixed": round(fx.score, 4),
        "score_joint": round(jt.score, 4),
        "score_refined": round(rf, 4),
        "d_joint_pct": round((jt.score - fx.score) / fx.score * 100.0, 2) if fx.score else 0.0,
        "d_refine_pct": round((rf - jt.score) / jt.score * 100.0, 2) if jt.score else 0.0,
        "n_picked": len(jt.seq),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rows = []
    for n_obj in B.GRID:
        for seed in range(args.seeds):
            rows.append(run("clutter", n_obj, seed))
    for n_obj in (9, 12):
        for seed in range(args.seeds):
            rows.append(run("piles", n_obj, seed))

    for r in rows:
        print("{:<11} seed={} n={:<3} fixed={:.4f} joint={:.4f} refined={:.4f} "
              "joint {:+.2f}%  refine {:+.2f}%".format(
                  r["case"], r["seed"], r["n_obj"], r["score_fixed"], r["score_joint"],
                  r["score_refined"], r["d_joint_pct"], r["d_refine_pct"]))

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "bench_refine.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n->", path)

    for fam in ("clutter", "piles"):
        sub = [r for r in rows if r["family"] == fam]
        n = len(sub)
        print("{:<8} n={:<3} joint vs fixed {:+.2f}%   refine vs joint {:+.2f}%".format(
            fam, n, sum(r["d_joint_pct"] for r in sub) / n,
            sum(r["d_refine_pct"] for r in sub) / n))


if __name__ == "__main__":
    main()
