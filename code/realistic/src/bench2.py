"""Episode runner: perceive, plan a window, act, recover, replan."""
import argparse
import csv
import os
import random
import statistics as st
import time

import world2 as W
import harness as H

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_episode(sc, planner, *, T=5, horizon=2, max_actions=90, seed=0,
                sigma=2.5, p_miss=0.10, bloat=0.05):
    rng = random.Random(seed * 977 + 7)
    true = sc
    total = len(sc.targets)
    got = set()
    actions = pushes = retries = clears = 0
    lat = []
    fails = 0
    attempts = {}

    while len(got) < total and actions < max_actions:
        obs = H.observe(true, rng, sigma=sigma, p_miss=p_miss, bloat=bloat)
        t0 = time.perf_counter()
        plan = planner(obs, T)
        lat.append((time.perf_counter() - t0) * 1000.0)
        if not plan:
            # nothing the planner can see is reachable: shove something to open up
            if pushes >= 8:
                break
            blockers = set()
            for o in true.targets - got:
                blockers |= true.blocked(o)
            if blockers:
                b = sorted(blockers)[rng.randrange(len(blockers))]
                true, _ = H.push(true, b, rng)
                pushes += 1
                actions += 1
            else:
                break
            continue

        acted = False
        a0 = actions
        for o, _act in plan[:horizon]:
            if o in got or actions >= max_actions:
                continue
            g = true.best(o)
            p = H.success_prob(true, o, g)
            actions += 1
            attempts[o] = attempts.get(o, 0) + 1
            if rng.random() < p:
                if o in true.targets:
                    got.add(o)
                else:
                    clears += 1
                acted = True
                bodies = list(true.bodies)
                bodies[o] = bodies[o].moved(0, 0)      # keeps index, marks nothing
                true = W.rebuild(true, bodies)
                true = H.settle(true, rng)
            else:
                fails += 1
                blk = true.blocked(o)
                if blk:
                    b = sorted(blk)[rng.randrange(len(blk))]
                    true, _ok = H.push(true, b, rng)
                    pushes += 1
                    actions += 1
                else:
                    retries += 1
                    true = H.settle(true, rng)
            if actions >= max_actions:
                break
        if actions == a0:
            # the plan was stale (everything in it is already gone): shove, or give up
            if pushes >= 8:
                break
            blockers = set()
            for o in true.targets - got:
                blockers |= true.blocked(o)
            if blockers:
                b = sorted(blockers)[rng.randrange(len(blockers))]
                true, _ = H.push(true, b, rng)
                pushes += 1
                actions += 1
            else:
                break

    lat_sorted = sorted(lat) or [0.0]
    p50 = lat_sorted[int(0.5 * (len(lat_sorted) - 1))]
    p99 = lat_sorted[min(len(lat_sorted) - 1, int(0.99 * len(lat_sorted)))]
    return dict(targets=total, got=len(got), success=int(len(got) == total),
                actions=actions, pushes=pushes, retries=retries, fails=fails, clears=clears,
                lat_p50=p50, lat_p99=p99,
                actions_per_target=actions / max(1, len(got)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, nargs="+", default=[24, 40, 60])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--miss", type=float, default=0.10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for nobj in args.objects:
        ntg = max(4, min(args.targets, nobj // 3))
        for planner in ("greedy", "milp", "sa"):
            for seed in range(args.seeds):
                sc = W.make_scene(n_obj=nobj, seed=seed, n_targets=ntg)
                r = run_episode(sc, H.PLANNERS[planner], T=args.window,
                                horizon=args.horizon, seed=seed, p_miss=args.miss)
                r.update(planner=planner, seed=seed, n=sc.n)
                rows.append(r)
                print("{:3d} {:7s} s{} got={:2d}/{:2d} act={:3d} push={:2d} retry={:2d} "
                      "p50={:6.1f}ms p99={:6.1f}ms".format(
                          nobj, planner, seed, r["got"], r["targets"], r["actions"],
                          r["pushes"], r["retries"], r["lat_p50"], r["lat_p99"]))
    path = os.path.join(OUT, "bench_{}.csv".format(args.tag))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n== summary ==")
    print("{:8s} {:>6s} {:>8s} {:>9s} {:>8s} {:>9s} {:>9s}".format(
        "planner", "succ", "got", "actions", "pushes", "p50ms", "p99ms"))
    for p in ("greedy", "milp", "sa"):
        sel = [r for r in rows if r["planner"] == p]
        print("{:8s} {:6.2f} {:8.2f} {:9.1f} {:8.2f} {:9.2f} {:9.2f}".format(
            p, st.mean(r["success"] for r in sel), st.mean(r["got"] for r in sel),
            st.mean(r["actions"] for r in sel), st.mean(r["pushes"] for r in sel),
            st.mean(r["lat_p50"] for r in sel), st.mean(r["lat_p99"] for r in sel)))
    print("wrote", path)


if __name__ == "__main__":
    main()
