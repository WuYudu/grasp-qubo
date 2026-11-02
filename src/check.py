"""Validation: for tiny instances the QUBO is checked against exhaustive search."""

import itertools
import random

import place
import quboplace


def _assignments(sc, T):
    """Every way to give each object one slot-step or the skip."""
    opts = [(o, c, s) for o in range(sc.n_obj) for c in range(sc.n_slot)
            if sc.fit(o, c) for s in range(T)]
    per_obj = {o: [None] + [t for t in opts if t[0] == o] for o in range(sc.n_obj)}
    for combo in itertools.product(*[per_obj[o] for o in range(sc.n_obj)]):
        yield combo


def brute(sc, T):
    """Best feasible score by enumeration, plus the argparse-free reason string."""
    best, arg = None, None
    for combo in _assignments(sc, T):
        used_c, used_s, done, ok = set(), set(), set(), True
        order = []
        for item in combo:
            if item is None:
                continue
            o, c, s = item
            if c in used_c or s in used_s:
                ok = False
                break
            used_c.add(c)
            used_s.add(s)
            order.append((o, c, s))
        if not ok:
            continue
        order.sort(key=lambda t: t[2])
        for o, c, s in order:
            if sc.blocked[o] - done:
                ok = False
                break
            done.add(o)
        if not ok:
            continue
        sc_val = sum(sc.value(o, c) for o, c, s in order)
        if best is None or sc_val > best:
            best, arg = sc_val, order
    return best, arg


def exhaustive_energy(sc, pack, T):
    """Minimum energy over all one-hot assignment vectors that pass decode."""
    best_e, best_info = None, None
    for combo in _assignments(sc, T):
        x = [0] * pack.n
        for item in combo:
            if item is None:
                continue
            o, c, s = item
            x[o * (sc.n_slot * T + 1) + c * T + s] = 1
        # skip bits: an object with no slot-step is skipped

        chosen = {v[0] for v in combo if v}
        for o in range(sc.n_obj):
            if o not in chosen:
                x[o * (sc.n_slot * T + 1) + sc.n_slot * T] = 1
        e = pack.energy(x)
        if best_e is None or e < best_e:
            best_e, best_info = e, (x, combo)
    return best_e, best_info


def check(name, sc, T):
    pack = quboplace.build(sc, T=T)
    best, arg = brute(sc, T)
    e_min, (x, combo) = exhaustive_energy(sc, pack, T)
    info = quboplace.decode(sc, pack, x)
    ok = True
    msgs = []
    if not info["feasible"]:
        ok = False
        msgs.append("argmin infeasible: %s" % info.get("reason"))
    if ok and abs(info["score"] - best) > 1e-6:
        ok = False
        msgs.append("score %.6f != brute %.6f" % (info["score"], best))
    if ok and abs(e_min + best) > 1e-6:
        ok = False
        msgs.append("E_min %.6f != -best %.6f" % (e_min, -best))
    rng = random.Random(0)
    for _ in range(400):
        xr = [1 if rng.random() < 0.08 else 0 for _ in range(pack.n)]
        ir = quboplace.decode(sc, pack, xr)
        e = pack.energy(xr)
        if ir["feasible"] and abs(e + ir["score"]) > 1e-6:
            ok = False
            msgs.append("feasible state breaks E=-score (%.6f vs %.6f)" % (e, -ir["score"]))
            break
    print("%-16s n_var=%3d nnz=%5d lam=%.1f  %s" % (name, pack.n, pack.nnz, pack.lam,
                                                   "OK" if ok else "FAIL " + "; ".join(msgs)))
    return ok


def main():
    good = True
    for name, n, k, t, seed in (("tiny-3x2", 3, 2, 2, 0), ("tiny-4x3", 4, 3, 2, 1),
                                ("tiny-5x3", 5, 3, 2, 2), ("mid-4x2", 4, 2, 1, 3)):
        sc = place.make_scene(n_obj=n, seed=seed, cluster=2, n_slot=max(k, 3))
        good &= check(name, sc, t)
    print("ALL OK" if good else "SOME FAILED")


if __name__ == "__main__":
    main()
