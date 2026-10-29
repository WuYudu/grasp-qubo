"""Single-scene demo: build the QUBO, solve it four ways, print the plan."""
from __future__ import annotations

import argparse
import json

import qubo
import scene as S
import solvers as V


def describe(sc: S.Scene, T: int) -> None:
    print("场景: {} 个物体, 步数预算 T={}, 族={}".format(sc.n_obj, T, sc.family))
    for o in range(sc.n_obj):
        bl = sorted(sc.blocked[o])
        flag = "可抓" if not bl else "被挡(先移走 {})".format(bl)
        print("  物体 {:>2d}: 分数 {:.3f}  {}".format(o, sc.score(o), flag))


def print_plan(sc: S.Scene, res: V.Result, T: int) -> None:
    tag = "可行" if res.feasible else "不可行({})".format(res.extra.get("reason", ""))
    print("\n{} 方案: {} 步, 总分 {:.4f}, {:.1f} ms, {}".format(
        res.solver, len(res.seq), res.score, res.time_ms, tag))
    for k, o in enumerate(res.seq):
        g = sc.grasp(o)
        print("  {:>2d}. 抓物体 {:>2d}  (分数 {:.3f}, 中心 {:.1f},{:.1f}, 开口 {:.0f}°)".format(
            k + 1, o, sc.score(o), g.cx, g.cy, g.theta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="piles", choices=["piles", "clutter"])
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--objects", type=int, default=9)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reads", type=int, default=40)
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--reads-sqa", type=int, default=16)
    ap.add_argument("--sweeps-sqa", type=int, default=400)
    ap.add_argument("--replicas", type=int, default=6)
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.family == "piles":
        sc = S.make_pile_scene(n_clusters=args.clusters, per_cluster=args.per,
                               seed=args.seed, budget=args.budget or None)
    else:
        sc = S.make_scene(n_obj=args.objects, seed=args.seed, budget=args.budget or None)
    T = sc.budget

    describe(sc, T)
    pack = qubo.build(sc, T)
    print("\nQUBO: n_var={} nnz={} lam={:.0f} offset={:.1f}".format(
        pack.n, pack.nnz, pack.lam, pack.offset))

    results = [V.exact_milp(sc, T, 120.0), V.greedy(sc, T)]
    results.append(V.anneal_classical(sc, pack, num_reads=args.reads, num_sweeps=args.sweeps, seed=args.seed))
    results.append(V.anneal_quantum(sc, pack, num_reads=args.reads_sqa, num_sweeps=args.sweeps_sqa,
                                    replicas=args.replicas, seed=args.seed))
    if args.hw:
        results.append(V.anneal_hardware(sc, pack))

    best = max(r.score for r in results if r.feasible) or 1.0
    print("\n{:<8} {:>8} {:>8} {:>10} {:>8} {:>9}".format("求解器", "分数", "步数", "耗时(ms)", "差距%", "不可行率"))
    for r in results:
        print("{:<8} {:>8.4f} {:>8d} {:>10.1f} {:>8.1f} {:>9.0%}".format(
            r.solver, r.score, len(r.seq), r.time_ms,
            (best - r.score) / best * 100, float(r.extra.get("infeasible_rate", 0.0))))

    for r in results:
        print_plan(sc, r, T)

    if args.json:
        print(json.dumps([{"solver": r.solver, "score": r.score, "seq": r.seq,
                           "feasible": r.feasible, "time_ms": r.time_ms} for r in results],
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
