"""Checks for the candidate-level encoder.  python src/_check_cand.py"""
from __future__ import annotations

import itertools
import random

import numpy as np

import qubo_cand as QC
import scene as S
import solvers


def _all_assignments(sc, T, k):
    S_ = T + 1
    opts = [[(c, s) for c in range(k) for s in range(S_)] for _ in range(sc.n_obj)]
    for combo in itertools.product(*opts):
        x = np.zeros(sc.n_obj * k * S_, dtype=int)
        for o, (c, s) in enumerate(combo):
            x[QC.vid(o, c, s, k, T)] = 1
        yield combo, x


def brute(sc, T, k):
    best, best_combo = 0.0, None
    for combo, x in _all_assignments(sc, T, k):
        out = QC.decode(sc, x, T, k)
        if out["feasible"] and out["score"] > best + 1e-9:
            best, best_combo = out["score"], combo
    return best, best_combo


def check(name, sc, T, k=3, verbose=True):
    pack = QC.build(sc, T, k=k)
    best, _ = brute(sc, T, k)
    energies = []
    for combo, x in _all_assignments(sc, T, k):
        out = QC.decode(sc, x, T, k)
        e = float(pack.energy(x.astype(float))[0][0])
        energies.append((e, out["feasible"], out["score"], combo))
    e_min, feas_min, score_min, combo_min = min(energies, key=lambda t: t[0])
    bad_eq = [(e, s) for e, f, s, _ in energies if f and abs(e + s) > 1e-6]
    bad_ineq = [(e, s) for e, f, s, _ in energies if not f and e <= -best + 1e-9]
    ok = (feas_min and abs(score_min - best) < 1e-6 and not bad_eq and not bad_ineq)
    if verbose:
        print("{:<4} {:<22} n_var={:<4} k={} T={} lam={:>5.1f} best={:.4f} "
              "argmin_feas={} feasible_states={}/{}".format(
                  "OK" if ok else "FAIL", name, pack.n, k, T, pack.lam, best,
                  feas_min, sum(1 for t in energies if t[1]), len(energies)))
        if bad_eq:
            print("     identity violated in {} states, e.g. {}".format(len(bad_eq), bad_eq[:2]))
        if bad_ineq:
            print("     infeasible state beat the optimum: {}".format(bad_ineq[:2]))
    return ok


def invariants(name, sc, T, k=3, trials=400, seed=0):
    pack = QC.build(sc, T, k=k)
    rng = random.Random(seed)
    milp = QC.exact_milp_cand(sc, T, k=k)
    best = milp.score
    n_var = pack.n
    bad = 0
    for _ in range(trials):
        m = rng.choice([1, 2, 3])
        x = np.zeros(n_var, dtype=int)
        for _ in range(m):
            x[rng.randrange(n_var)] = 1
        e = float(pack.energy(x.astype(float))[0][0])
        out = QC.decode(sc, x, T, k)
        if out["feasible"]:
            if abs(e + out["score"]) > 1e-6:
                bad += 1
        elif e <= -best + 1e-9:
            bad += 1
    ok = bad == 0
    print("{:<4} {:<22} n_var={:<4} milp={:.4f} feasible={} ({:.1f} ms)  random {} states".format(
        "OK" if ok else "FAIL", name, n_var, best, milp.feasible, milp.time_ms, trials))
    return ok


def main():
    ok = True
    sc = S.make_pile_scene(n_clusters=2, per_cluster=2, seed=1, budget=2)
    ok &= check("piles-2x2", sc, 2)
    sc = S.make_pile_scene(n_clusters=3, per_cluster=1, seed=2, budget=2)
    ok &= check("three-free", sc, 2)
    sc = S.make_scene(n_obj=4, seed=3, budget=2, max_overlap=0.6)
    ok &= check("clutter-4", sc, 2)

    sc = S.make_pile_scene(n_clusters=3, per_cluster=2, seed=4, budget=3)
    ok &= invariants("piles-3x2", sc, 3)
    sc = S.make_pile_scene(n_clusters=4, per_cluster=3, seed=5, budget=5)
    ok &= invariants("piles-4x3", sc, 5)
    print("ALL OK" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
