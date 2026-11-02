"""Closed loop: plan a window, execute it, watch placements fail, replan."""

import argparse
import csv
import os
import random
import statistics
import time

import plan100
import world100

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def episode(n_obj, planner, seed, T=6, action_budget=None, factor=4.0):
    w = world100.make_world(n_obj=n_obj, seed=seed, cluster=3, n_slot=6)
    rng = random.Random(10000 + seed)
    budget = action_budget or int(factor * n_obj)
    actions = rounds = 0
    lat = []
    cycles = 0
    stall = 0
    while actions < budget:
        if not w.remaining():
            break
        if not w.free_slots():
            w.new_tray()
            cycles += 1
            continue
        p = plan100.PLANNERS[planner](w, T)
        rounds += 1
        lat.append(p.ms)
        if not p.order:
            break
        moved = False
        for i in p.order:
            o = p.ids[i]
            if w.objs[o].gone:
                continue
            if w.blocked[o] - {x for x in range(w.n_obj) if w.objs[x].gone}:
                continue
            free = w.free_slots()
            if not free:
                break
            c = min(free, key=lambda cc: w.cost(o, cc))
            outcome = w.place_with_failure(o, c, rng)
            actions += 1
            moved = True
            if actions >= budget:
                break
            if outcome == "slip":
                continue
        stall = 0 if moved else stall + 1
        if stall >= 5:
            break
    placed = len(w.placed)
    lat.sort()

    def q(p):
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, int(p * (len(lat) - 1)))]

    return {
        "n_obj": n_obj, "seed": seed, "planner": planner,
        "placed": placed, "rate": 100.0 * placed / n_obj,
        "actions": actions, "rounds": rounds, "fails": w.fails,
        "per_placed": (actions / placed) if placed else float("nan"),
        "p50": q(0.5), "p99": q(0.99),
        "cycles": cycles,
        "waste_slots": sum(1 for d in w.dirty if d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", type=int, default=[50, 100])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--planners", nargs="+", default=["greedy", "milp", "sa"])
    ap.add_argument("--factor", type=float, default=4.0, help="action budget = factor * N")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in args.objects:
        for pl in args.planners:
            for sd in range(args.seeds):
                t0 = time.perf_counter()
                r = episode(n, pl, sd, T=args.steps, factor=args.factor)
                r["wall_s"] = time.perf_counter() - t0
                rows.append(r)
                print("%-7s N=%-4d seed=%d  placed=%3d/%3d (%5.1f%%)  actions=%4d rounds=%3d "
                      "cycles=%2d fails=%3d  per=%.2f  p50=%.0f p99=%.0f  wall=%.1fs"
                      % (pl, n, sd, r["placed"], n, r["rate"], r["actions"], r["rounds"],
                         r["cycles"], r["fails"], r["per_placed"], r["p50"], r["p99"], r["wall_s"]), flush=True)
    path = os.path.join(OUT, "bench_%s.csv" % args.tag)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", path)
    for pl in args.planners:
        sel = [r for r in rows if r["planner"] == pl]
        print("%-7s rate %.1f%%  per_placed %.2f  p99 %.0f ms  wall %.1fs"
              % (pl, statistics.mean(r["rate"] for r in sel),
                 statistics.mean(r["per_placed"] for r in sel),
                 max(r["p99"] for r in sel),
                 statistics.mean(r["wall_s"] for r in sel)))


if __name__ == "__main__":
    main()
