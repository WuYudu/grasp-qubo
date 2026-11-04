"""标定：在 60–80 变量的窗口里，找 Harness 启发式开始出灾难性重试的扰动区间。"""

import argparse
import itertools
import os

import sim

GRID = [(10, 5), (11, 5), (12, 4), (13, 4), (14, 4), (15, 4)]


def one(planner, n, T, seeds, p_add, p_fail, budget=0.7, slack=0, reads=40, w=0.0,
        hw=0.0, window=None, horizon=1):
    ww = hw if planner == "harness" else w
    rows = [sim.episode(n_obj=n, T=T, planner=planner, seed=s, p_add=p_add,
                        p_fail=p_fail, slack=slack, budget=budget, reads=reads, w=ww,
                        window=window, horizon=horizon)
            for s in range(seeds)]
    k = len(rows)
    return {
        "planner": planner, "n_obj": n, "T": T, "vars": n * (T + 1),
        "p_add": p_add, "p_fail": p_fail, "budget": budget, "w": w,
        "completion": sum(r["completion"] for r in rows) / k,
        "catastrophic": sum(r["catastrophic"] for r in rows) / k,
        "retries": sum(r["retries"] for r in rows) / k,
        "deadlocks": sum(r["deadlocks"] for r in rows) / k,
        "fallback": sum(r["fallback"] for r in rows) / k,
        "p99": sum(r["dec_p99"] for r in rows) / k,
        "clock": sum(r["clock"] for r in rows) / k,
        "travel": sum(r["travel_mm"] for r in rows) / k,
        "horizon": horizon,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--adds", default="0.10,0.30")
    ap.add_argument("--fails", default="0.20,0.35")
    ap.add_argument("--budgets", default="0.7")
    ap.add_argument("--planners", default="harness,sa")
    ap.add_argument("--ws", default="0.0")
    ap.add_argument("--hw", type=float, default=0.0)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--horizons", default="1")
    ap.add_argument("--out", default="../results/calib.csv")
    a = ap.parse_args()
    adds = [float(x) for x in a.adds.split(",")]
    fails = [float(x) for x in a.fails.split(",")]
    budgets = [float(x) for x in a.budgets.split(",")]
    ws = [float(x) for x in a.ws.split(",")]
    horizons = [int(x) for x in a.horizons.split(",")]
    planners = a.planners.split(",")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = []
    head = "{:8s} {:>3s} {:>2s} {:>4s} {:>5s} {:>6s} {:>6s} {:>6s} {:>5s} {:>5s} {:>6s} {:>7s} {:>6s} {:>7s} {:>2s}".format(
        "planner", "N", "T", "vars", "p_add", "p_fail", "budget", "compl", "cat",
        "retry", "p99", "w", "clock", "travel", "hz")
    print(head)
    for planner in planners:
        for (n, T), p_add, p_fail, budget, w, hz in itertools.product(
                GRID, adds, fails, budgets, ws, horizons):
            r = one(planner, n, T, a.seeds, p_add, p_fail, budget, w=w, hw=a.hw,
                    window=a.window, horizon=hz)
            rows.append(r)
            print("{:8s} {:3d} {:2d} {:4d} {:5.2f} {:6.2f} {:6.2f} {:6.1%} {:5.0%} {:5.1f} {:6.1f} {:7.4f} {:6.1f} {:7.0f} {:2d}".format(
                planner, n, T, r["vars"], p_add, p_fail, budget, r["completion"],
                r["catastrophic"], r["retries"], r["p99"], w, r["clock"], r["travel"], r["horizon"]))
    with open(a.out, "w", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print("written", a.out)


if __name__ == "__main__":
    main()
