"""量一下贪心到底多快（微秒级），以及和 MILP / 退火的比值。"""
import time

import scene as S
import solvers as V

print("{:>6} {:>6} {:>9} {:>12} {:>12} {:>14}".format(
    "M", "n_var", "greedy_us", "milp_ms", "sa_ms", "sqa_ms"))
for cl, per in ((2, 3), (3, 3), (4, 3), (6, 4)):
    M = cl * per
    T = round(0.6 * M)
    sc = S.make_pile_scene(n_clusters=cl, per_cluster=per, seed=0, budget=T)
    # greedy：重复 200 次取中位数（单次太短，计时器噪声大）
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        V.greedy(sc, T)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    g_us = ts[len(ts) // 2]

    t0 = time.perf_counter()
    V.exact_milp(sc, T, 300.0)
    milp = (time.perf_counter() - t0) * 1e3

    pack = __import__("qubo").build(sc, T)
    t0 = time.perf_counter()
    V.anneal_classical(sc, pack, num_reads=40, num_sweeps=1000, seed=0)
    sa = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    V.anneal_quantum(sc, pack, num_reads=16, num_sweeps=400, replicas=6, seed=0)
    sqa = (time.perf_counter() - t0) * 1e3

    print("{:>6} {:>6} {:>9.1f} {:>12.2f} {:>12.1f} {:>14.1f}".format(
        M, M * (T + 1), g_us, milp, sa, sqa))
    print("       倍率：greedy vs MILP = {:.0f}x, vs SA = {:.0f}x, vs SQA = {:.0f}x".format(
        milp * 1e3 / g_us, sa * 1e3 / g_us, sqa * 1e3 / g_us))
