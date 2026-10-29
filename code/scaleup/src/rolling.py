"""Rolling-horizon replanning: plan a window, execute part of it, replan."""
import random
import time

import qubo
import solvers
import world


def plan(sc, T, method):
    """One planning call on the current sub-scene."""
    if method == "greedy":
        return solvers.greedy(sc, T).seq
    if method == "milp":
        return solvers.exact_milp(sc, T, time_limit=20.0).seq
    if method in ("sa", "sqa"):
        pack = qubo.build(sc, T)
        fn = solvers.anneal_classical if method == "sa" else solvers.anneal_quantum
        return fn(sc, pack).seq
    raise ValueError(method)


def run_rolling(sc, *, T=6, horizon=3, method="milp", p_fail=0.15, sigma=2.0,
                slip_p=0.06, seed=0, max_attempts=400):
    """Closed loop: replan every `horizon` attempts, disturbances between rounds."""
    rng = random.Random(seed * 7919 + 13)
    cur = sc
    remaining = set(range(sc.n_obj))
    grasps = attempts = rounds = nudges = wasted = 0
    got_score = 0.0
    solver_ms = 0.0

    while remaining and attempts < max_attempts:
        sub, keep = world.subset(cur, remaining)
        T_eff = max(1, min(T, len(remaining)))
        t0 = time.perf_counter()
        seq = plan(sub, T_eff, method)
        solver_ms += (time.perf_counter() - t0) * 1000.0
        rounds += 1
        seq = [keep[i] for i in seq]

        if not seq:
            # nothing reachable: push one of the remaining bodies aside and retry
            nudges += 1
            attempts += 1
            cur = world.nudge(cur, rng, remaining)
            continue

        progressed = False
        for o in seq[:horizon]:
            if o not in remaining:
                continue
            attempts += 1
            if rng.random() < p_fail:
                continue
            remaining.discard(o)
            grasps += 1
            got_score += cur.score(o)
            progressed = True
            if not remaining:
                break
        if not progressed:
            wasted += 1
        cur = world.disturb(cur, rng, sigma=sigma, slip_p=slip_p)

    return dict(grasps=grasps, attempts=attempts, rounds=rounds, nudges=nudges,
                wasted=wasted, left=len(remaining), score=got_score,
                solver_ms=solver_ms, ms_per_round=solver_ms / max(1, rounds))


def run_oneshot(sc, *, T, method="milp", p_fail=0.15, seed=0, max_attempts=400):
    """Open loop: plan once for the whole scene, execute it, no replanning."""
    rng = random.Random(seed * 7919 + 13)
    t0 = time.perf_counter()
    seq = plan(sc, T, method)
    solver_ms = (time.perf_counter() - t0) * 1000.0
    remaining = set(range(sc.n_obj))
    grasps = attempts = 0
    got_score = 0.0
    for o in seq:
        if attempts >= max_attempts or not remaining:
            break
        if o not in remaining:
            continue
        attempts += 1
        if rng.random() < p_fail:
            continue
        remaining.discard(o)
        grasps += 1
        got_score += sc.score(o)
    return dict(grasps=grasps, attempts=attempts, rounds=1, nudges=0, wasted=0,
                left=len(remaining), score=got_score, solver_ms=solver_ms,
                ms_per_round=solver_ms)
