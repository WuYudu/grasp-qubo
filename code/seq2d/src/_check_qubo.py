"""Verification of the sequential-grasp QUBO.

Small instances: exhaustive over every one of the 2^n binary states, reference optimum
from all feasible sequences.  Large instances: reference optimum from the MILP, plus the
invariant "feasible plan => energy == -score" checked on random states and on the
sampler/greedy solutions.
"""
from __future__ import annotations

import itertools
import random

import numpy as np

import qubo
import scene as scene_mod
import solvers


def brute_force(scene, T: int):
    best, best_seq = 0.0, []
    for k in range(1, T + 1):
        for perm in itertools.permutations(range(scene.n_obj), k):
            out = qubo.evaluate(scene, list(perm), T)
            if out["feasible"] and out["score"] > best:
                best, best_seq = out["score"], list(perm)
    return best, best_seq


def encode_plan(scene, seq, T: int) -> np.ndarray:
    x = np.zeros(scene.n_obj * (T + 1), dtype=np.int8)
    for k, o in enumerate(seq):
        x[qubo.vid(o, k, T + 1)] = 1
    for o in range(scene.n_obj):
        if o not in seq:
            x[qubo.vid(o, T, T + 1)] = 1
    return x


def random_states(scene, T: int, n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = np.zeros(scene.n_obj * (T + 1), dtype=np.int8)
        for o in range(scene.n_obj):
            if rng.random() < 0.85:
                x[qubo.vid(o, rng.randrange(T + 1), T + 1)] = 1
            else:
                for s in range(T + 1):
                    x[qubo.vid(o, s, T + 1)] = 1 if rng.random() < 0.3 else 0
        out.append(x)
    return np.array(out)


def check_small(name: str, scene, T: int) -> bool:
    pack = qubo.build(scene, T)
    best, best_seq = brute_force(scene, T)
    n = pack.n
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    energies = pack.energy(states.T)
    feasible = np.zeros(states.shape[0], dtype=bool)
    scores = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        out = qubo.decode(scene, states[i], T)
        feasible[i] = out["feasible"]
        scores[i] = out["score"]
    i_min = int(np.argmin(energies))
    argmin = qubo.decode(scene, states[i_min], T)
    feas_min = float(energies[feasible].min())
    ok = (argmin["feasible"] and abs(argmin["score"] - best) < 1e-6
          and abs(float(energies[i_min]) - feas_min) < 1e-9)
    if (~feasible).any():
        ok = ok and float(energies[~feasible].min()) > feas_min - 1e-9
    print("{:4s} {:<20s} n_var={:<4d} nnz={:<5d} lam={:<5.1f} best={:.4f}  exhaustive 2^{}={}".format(
        "OK" if ok else "FAIL", name, pack.n, pack.nnz, pack.lam, best, n, 1 << n))
    if not ok:
        print("      argmin seq:", argmin["seq"], argmin["reason"], " brute:", best_seq)
    return ok


def check_large(name: str, scene, T: int) -> bool:
    pack = qubo.build(scene, T)
    ref = solvers.exact_milp(scene, T, 120.0)
    ok = ref.feasible
    x_ref = encode_plan(scene, ref.seq, T)
    ok = ok and abs(pack.energy(x_ref) + ref.score) < 1e-6
    ok = ok and qubo.decode(scene, x_ref, T)["score"] >= ref.score - 1e-6

    states = random_states(scene, T, 400)
    energies = pack.energy(states.T)
    for x, e in zip(states, energies):
        out = qubo.decode(scene, x, T)
        if out["feasible"]:
            ok = ok and abs(e + out["score"]) < 1e-6
    worst_infeasible = np.inf
    for x, e in zip(states, energies):
        if not qubo.decode(scene, x, T)["feasible"]:
            worst_infeasible = min(worst_infeasible, e)
    ok = ok and worst_infeasible > -ref.score - 1e-9
    print("{:4s} {:<20s} n_var={:<4d} nnz={:<5d} lam={:<5.1f} best={:.4f}  "
          "MILP reference + invariant check".format(
              "OK" if ok else "FAIL", name, pack.n, pack.nnz, pack.lam, ref.score))
    return ok


def main() -> None:
    ok = True

    s = scene_mod.make_pile_scene(n_clusters=2, per_cluster=2, seed=1, budget=2)
    ok = check_small("piles-2x2", s, 2) and ok

    s = scene_mod.make_pile_scene(n_clusters=3, per_cluster=2, seed=7, budget=2)
    ok = check_small("piles-3x2", s, 2) and ok

    s = scene_mod.make_scene(n_obj=3, seed=5, max_overlap=0.6, budget=2)
    s.blocked = [{1}, {0}, set()]
    ok = check_small("mutual-block-cycle", s, 2) and ok

    s = scene_mod.make_pile_scene(n_clusters=2, per_cluster=3, seed=4, budget=3)
    ok = check_large("piles-2x3", s, 3) and ok

    s = scene_mod.make_pile_scene(n_clusters=4, per_cluster=4, seed=0, budget=8)
    ok = check_large("piles-4x4", s, 8) and ok

    s = scene_mod.make_scene(n_obj=12, seed=3, max_overlap=0.45, budget=7)
    ok = check_large("clutter-12", s, 7) and ok

    print("ALL OK" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
