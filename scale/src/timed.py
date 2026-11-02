"""15 版内核 + 时限：观测浅拷贝加噪、只执行计划头一步、头不可行就推一下再重规划。"""

import copy
import random
import time

import harness
import world2

SPEED = 600.0
T_FIX = 0.70
DROP = (450.0, 20.0)
T_WINDOW = 5
PUSH_T = 0.30
STALL_LIMIT = 6


def act_time(b):
    return T_FIX + (abs(b.cx - DROP[0]) + abs(b.cy - DROP[1])) / SPEED


def observe_light(sc, rng, sigma=2.5):
    seen = copy.deepcopy(sc)
    for gs in seen.grasps:
        for g in gs:
            try:
                g.cx = getattr(g, "cx", 0.0) + rng.gauss(0.0, sigma)
                g.cy = getattr(g, "cy", 0.0) + rng.gauss(0.0, sigma)
            except Exception:
                pass
    return seen


def episode(n_obj, planner, seed, deadline=60.0):
    canvas = (9.0 * n_obj, 260.0)
    sc = world2.make_scene(n_obj=n_obj, seed=seed, cluster=3, canvas=canvas,
                           n_targets=int(0.4 * n_obj), n_fragile=int(0.1 * n_obj))
    rng = random.Random(9000 + seed)
    sim, acts, fails, pushes, blocked_skip, lat, rounds = 0.0, 0, 0, 0, 0, [], 0
    fell_back = 0
    sorted_targets = 0
    stall = 0
    while sim < deadline and sc.bodies:
        seen = observe_light(sc, rng)
        t0 = time.perf_counter()
        if planner == "milp":
            plan = harness.milp(seen, T=T_WINDOW, time_limit=1.0)
        else:
            plan = getattr(harness, planner)(seen, T_WINDOW)
        idx = None
        for item in plan:
            j = item[0] if isinstance(item, tuple) else item
            if j >= len(sc.bodies):
                continue
            try:
                if harness.feasible(sc, j):
                    idx = j
                    break
            except ValueError:
                continue
        if idx is None:
            fell_back += 1
            for item in harness.greedy(sc, T_WINDOW):
                j = item[0] if isinstance(item, tuple) else item
                if j < len(sc.bodies) and harness.feasible(sc, j):
                    idx = j
                    break
        lat.append(1000.0 * (time.perf_counter() - t0))
        rounds += 1
        if idx is None:
            break
        ok = True
        moved = False
        if ok:
            b = sc.bodies[idx]
            dt = act_time(b)
            if sim + dt > deadline:
                break
            sim += dt
            acts += 1
            moved = True
            try:
                p = harness.success_prob(sc, idx, sc.best(idx))
            except ValueError:
                p = 0.5
            if rng.random() < min(0.97, max(0.05, p)):
                if getattr(b, "target", False):
                    sorted_targets += 1
                bodies = [x for j, x in enumerate(sc.bodies) if j != idx]
                sc = world2.rebuild(sc, bodies)
                sc.targets = {i for i, x in enumerate(sc.bodies) if getattr(x, "target", False)}
                sc.fragile = {i for i, x in enumerate(sc.bodies) if getattr(x, "fragile", False)}
            else:
                fails += 1
                if rng.random() < 0.5:
                    harness.push(sc, idx, rng)
                    pushes += 1
        else:
            blocked_skip += 1
            if sim + PUSH_T < deadline:
                harness.push(sc, idx, rng)
                pushes += 1
                sim += PUSH_T
                moved = True
        stall = 0 if moved else stall + 1
        if stall >= STALL_LIMIT:
            break
    lat.sort()
    p99 = lat[min(len(lat) - 1, int(0.99 * (len(lat) - 1)))] if lat else 0.0
    p50 = lat[len(lat) // 2] if lat else 0.0
    return {"n_obj": n_obj, "seed": seed, "planner": planner, "sorted": sorted_targets,
            "actions": acts, "fails": fails, "pushes": pushes, "skip": blocked_skip,
            "makespan": round(sim, 1), "per_min": round(60.0 * sorted_targets / max(sim, 1e-9), 2),
            "rounds": rounds, "fallback": fell_back, "p50": round(p50, 1), "p99": round(p99, 1)}


def main():
    rows = []
    for planner in ("greedy", "milp", "sa"):
        for seed in (0, 1):
            r = episode(100, planner, seed)
            rows.append(r)
            print("%-7s seed=%d 分拣 %2d 动作 %3d 失手 %2d 推 %2d 跳过 %2d 用时 %5.1fs "
                  "%6.2f 件/分 兜底 %2d p50 %4.0f p99 %4.0f ms" % (planner, seed, r["sorted"], r["actions"],
                                                          r["fails"], r["pushes"], r["skip"],
                                                          r["makespan"], r["per_min"], r["fallback"],
                                                          r["p50"], r["p99"]), flush=True)
    import csv, os
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "timed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["planner", "seed", "sorted", "actions", "fails", "pushes", "skip",
                    "makespan_s", "per_min", "fallback", "p50_ms", "p99_ms"])
        for r in rows:
            w.writerow([r["planner"], r["seed"], r["sorted"], r["actions"], r["fails"],
                        r["pushes"], r["skip"], r["makespan"], r["per_min"],
                        r["fallback"], r["p50"], r["p99"]])
    print("--- mean ---")
    for planner in ("greedy", "milp", "sa"):
        sel = [r for r in rows if r["planner"] == planner]
        print("%-7s 分拣 %.1f  %.2f 件/分 失手 %.1f 跳过 %.1f 用时 %.1fs"
              % (planner, sum(r["sorted"] for r in sel) / len(sel),
                 sum(r["per_min"] for r in sel) / len(sel),
                 sum(r["fails"] for r in sel) / len(sel),
                 sum(r["skip"] for r in sel) / len(sel),
                 sum(r["makespan"] for r in sel) / len(sel)))


if __name__ == "__main__":
    main()
