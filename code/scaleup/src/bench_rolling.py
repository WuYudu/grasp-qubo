"""Open loop vs rolling horizon, with a failing gripper and moving objects."""
import argparse
import csv
import os
import statistics as st

import rolling
import world

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SCENES = [(6, 4), (8, 4), (10, 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--p-fail", type=float, default=0.15)
    ap.add_argument("--tag", default="rolling")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for nc, pc in SCENES:
        for seed in range(args.seeds):
            sc = world.big_piles(nc, pc, seed=seed)
            n = sc.n_obj
            runs = {
                "oneshot-milp": rolling.run_oneshot(sc, T=round(0.5 * n), method="milp",
                                                    p_fail=args.p_fail, seed=seed),
                "roll-greedy": rolling.run_rolling(sc, T=args.T, horizon=args.horizon,
                                                   method="greedy", p_fail=args.p_fail,
                                                   seed=seed),
                "roll-milp": rolling.run_rolling(sc, T=args.T, horizon=args.horizon,
                                                 method="milp", p_fail=args.p_fail,
                                                 seed=seed),
                "roll-sa": rolling.run_rolling(sc, T=args.T, horizon=args.horizon,
                                               method="sa", p_fail=args.p_fail,
                                               seed=seed),
            }
            for name, r in runs.items():
                rows.append(dict(scene="piles{}x{}".format(nc, pc), n=n, seed=seed,
                                 method=name, **r))
                print("{:8s} N={:2d} s{} {:12s} grasps={:2d} att={:3d} rounds={:3d} "
                      "nudge={:2d} left={:2d} score={:6.2f} solver={:7.1f}ms".format(
                          "piles", n, seed, name, r["grasps"], r["attempts"],
                          r["rounds"], r["nudges"], r["left"], r["score"], r["solver_ms"]))
    path = os.path.join(OUT, "bench_{}.csv".format(args.tag))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n== 汇总（按方法） ==")
    print("{:14s} {:>8s} {:>8s} {:>8s} {:>9s} {:>11s}".format(
        "method", "grasps", "attempts", "nudges", "left", "score", "solver_ms"))
    for name in ("oneshot-milp", "roll-greedy", "roll-milp", "roll-sa"):
        sel = [r for r in rows if r["method"] == name]
        if not sel:
            continue
        print("{:14s} {:8.2f} {:8.1f} {:8.2f} {:9.2f} {:8.2f} {:11.1f}".format(
            name, st.mean(r["grasps"] for r in sel), st.mean(r["attempts"] for r in sel),
            st.mean(r["nudges"] for r in sel), st.mean(r["left"] for r in sel),
            st.mean(r["score"] for r in sel),
            st.mean(r["solver_ms"] for r in sel)))
    print("wrote", path)


if __name__ == "__main__":
    main()
