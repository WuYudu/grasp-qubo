"""Closed-loop grasping under noise: detection noise + execution failures.

    python src/noisy.py --seeds 20

Noise is pre-drawn per (object, decision step) and (object, attempt number), so
every strategy faces exactly the same perturbations - the comparison is paired.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import random

import qubo
import scene as S
import solvers

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
POSES = ((-0.26, 0.0), (0.0, 0.0), (0.26, 0.0), (0.0, 32.0))


def make_noise(n_obj, T, seed):
    rng = random.Random(7000 + seed)
    det = [[rng.gauss(0.0, 1.0) for _ in range(T + 1)] for _ in range(n_obj)]
    fail = [[rng.random() for _ in range(T + 2)] for _ in range(n_obj)]
    return det, fail


def sub_scene(sc, alive, step, det, sigma):
    idx = sorted(alive)
    pos = {o: i for i, o in enumerate(idx)}
    grasps = []
    for o in idx:
        gs = []
        for g in sc.grasps[o]:
            v = g.score * (1.0 + sigma * det[o][step]) if sigma > 0 else g.score
            gs.append(dataclasses.replace(g, score=max(0.0, min(1.0, v))))
        grasps.append(gs)
    blocked = [{pos[p] for p in sc.blocked[o] if p in alive} for o in idx]
    return S.Scene([sc.bodies[o] for o in idx], grasps, [sc.chosen[o] for o in idx],
                   blocked, budget=len(idx),
                   levels=None if sc.levels is None else [sc.levels[o] for o in idx],
                   family=sc.family), idx


def run_one(n_obj, seed, p_fail, sigma, strategy, T_frac=0.4):
    base = S.make_scene(n_obj=n_obj, seed=seed, max_overlap=0.45, poses=POSES)
    T = max(1, int(round(T_frac * n_obj)))
    det, fail = make_noise(base.n_obj, T, seed)
    alive = set(range(base.n_obj))
    attempts = [0] * base.n_obj
    gained = 0.0
    removed = 0
    picks = 0
    fails = 0
    dead = 0
    ms = 0.0
    step = 0
    while picks < T and alive:
        sub, idx = sub_scene(base, alive, step, det, sigma)
        left = T - picks
        if strategy == "greedy":
            res = solvers.greedy(sub, min(left, len(idx)))
        elif strategy == "milp":
            res = solvers.exact_milp(sub, min(left, len(idx)))
        else:
            pack = qubo.build(sub, min(left, len(idx)))
            res = solvers.anneal_classical(sub, pack, num_reads=20, num_sweeps=400, seed=seed)
        ms += res.time_ms
        picks += 1
        step += 1
        if not res.feasible or not res.seq:
            dead += 1
            break
        o = idx[res.seq[0]]
        attempts[o] += 1
        if fail[o][attempts[o]] < p_fail:
            fails += 1
            continue
        removed += 1
        gained += base.score(o)
        alive.discard(o)
    return {"n_obj": n_obj, "seed": seed, "p_fail": p_fail, "sigma": sigma,
            "strategy": strategy, "removed": removed, "score": round(gained, 3),
            "picks": picks, "fails": fails, "deadlock": dead,
            "ms_per_decision": round(ms / max(1, picks), 1)}


CONDITIONS = ((0.0, 0.0), (0.0, 0.4), (0.25, 0.0), (0.25, 0.4))
STRATEGIES = ("greedy", "milp", "sa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--grid", default="12,18")
    args = ap.parse_args()
    grid = [int(x) for x in args.grid.split(",")]

    rows = []
    for n_obj in grid:
        for seed in range(args.seeds):
            for p_fail, sigma in CONDITIONS:
                for strategy in STRATEGIES:
                    rows.append(run_one(n_obj, seed, p_fail, sigma, strategy))

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "noisy.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("{:<8} {:>6} {:>6} {:>10} {:>10} {:>11} {:>7}".format(
        "strategy", "p_fail", "sigma", "score", "removed", "ms/decision", "deadlocks"))
    for strategy in STRATEGIES:
        for p_fail, sigma in CONDITIONS:
            sub = [r for r in rows if r["strategy"] == strategy
                   and r["p_fail"] == p_fail and r["sigma"] == sigma]
            n = len(sub)
            print("{:<8} {:>6.2f} {:>6.2f} {:>10.3f} {:>10.2f} {:>11.1f} {:>7}".format(
                strategy, p_fail, sigma,
                sum(r["score"] for r in sub) / n,
                sum(r["removed"] for r in sub) / n,
                sum(r["ms_per_decision"] for r in sub) / n,
                sum(r["deadlock"] for r in sub)))

    print("\nrelative loss vs the no-noise row (%)")
    for strategy in STRATEGIES:
        base = [r for r in rows if r["strategy"] == strategy
                and r["p_fail"] == 0.0 and r["sigma"] == 0.0]
        b = sum(r["score"] for r in base) / len(base)
        cells = []
        for p_fail, sigma in CONDITIONS[1:]:
            sub = [r for r in rows if r["strategy"] == strategy
                   and r["p_fail"] == p_fail and r["sigma"] == sigma]
            m = sum(r["score"] for r in sub) / len(sub)
            cells.append("p={:.2f} s={:.2f}: {:+.1f}%".format(p_fail, sigma, (m - b) / b * 100))
        print("{:<8} base {:.3f}   {}".format(strategy, b, "   ".join(cells)))
    print("\n->", path)


if __name__ == "__main__":
    main()
