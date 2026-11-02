"""time-to-target：同一个窗口问题上，各求解器"累计花多少时间才达到目标质量"。

指标口径：对一个实例，先取一个目标分数（默认 = 各方法见过的最好分数的 99%），然后让每个
求解器按"努力阶梯"逐档加码，每档重复若干次，累计墙钟时间；第一次达到目标的累计时间就是它
的 time-to-target（TTT）。达不到就记失败。这比"跑一次计时"公平，也是能直接换成真机采样器
的口径（把 sampler 换掉、把每档的采样时间换成 qpu_access_time 即可）。
"""

import os
import random
import statistics
import time

import dimod
import harness
import world2
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler

T_WINDOW = 5
REPEAT = 3
TARGET_FRAC = 0.99

LADDER = {
    "sa": [(40, 400), (80, 800), (160, 1200), (320, 2000), (640, 3000)],
    "pimc": [(20, 300, 4), (40, 600, 6), (80, 900, 8), (160, 1200, 10)],
}


def window(sc, T=T_WINDOW):
    cand, S, Q = harness._window_problem(sc, T)
    qd = {(i, j): float(Q[i][j]) for i in range(Q.shape[0]) for j in range(Q.shape[1])
          if Q[i][j] != 0.0}
    return cand, S, dimod.BinaryQuadraticModel.from_qubo(qd)


def decode_score(sc, cand, S, sample, T=T_WINDOW):
    var = {k: v for k, v in sample.items() if v > 0.5}
    seq, seen = [], set()
    for s in range(T):
        for i in range(len(cand)):
            if var.get(i * S + s, 0) > 0.5 and cand[i] not in seen:
                seen.add(cand[i])
                seq.append((cand[i], "grasp"))
    legal = harness._legal(sc, seq)
    return sum(harness.value(sc, o) for o, _ in legal), legal


def one_sample(bqm, kind, effort, seed):
    if kind == "sa":
        reads, sweeps = effort
        res = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=sweeps, seed=seed)
    else:
        reads, sweeps, reps = effort
        res = PathIntegralAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=sweeps,
                                                   replicas=reps, seed=seed)
    return res


def ttt(sc, kind, target, budget_s=30.0):
    """按努力阶梯加码，返回（达到目标的累计秒数, 是否达标, 达到时的分数）。"""
    cand, S, bqm = window(sc)
    total, best = 0.0, -1.0
    for k, effort in enumerate(LADDER[kind]):
        for r in range(REPEAT):
            t0 = time.perf_counter()
            res = one_sample(bqm, kind, effort, 1000 * k + r)
            dt = time.perf_counter() - t0
            total += dt
            for sample in res.samples():
                sc_v, _ = decode_score(sc, cand, S, sample)
                best = max(best, sc_v)
            if best >= target - 1e-9:
                return total, True, best
            if total > budget_s:
                return total, False, best
    return total, False, best


def full_run(sc, kind, budget_s=20.0):
    """跑完整条阶梯、不提前退出，用来定目标分数；返回（最好分数, 累计秒数）。"""
    cand, S, bqm = window(sc)
    total, best = 0.0, -1.0
    for k, effort in enumerate(LADDER[kind]):
        res = one_sample(bqm, kind, effort, 7000 + k)
        total += 0.0
        t0 = time.perf_counter()
        one_sample(bqm, kind, effort, 7000 + k)
        total += time.perf_counter() - t0
        for sample in res.samples():
            sc_v, _ = decode_score(sc, cand, S, sample)
            best = max(best, sc_v)
        if total > budget_s:
            break
    return best, total


def main():
    rows = []
    for n_obj in (60, 100):
        for seed in (0, 1):
            sc = world2.make_scene(n_obj=n_obj, seed=seed, cluster=3,
                                   canvas=(9.0 * n_obj, 260.0),
                                   n_targets=int(0.4 * n_obj), n_fragile=int(0.1 * n_obj))
            best_known = -1.0
            for kind in ("sa", "pimc"):
                v, _ = full_run(sc, kind)
                best_known = max(best_known, v)
            target = TARGET_FRAC * best_known
            for kind in ("sa", "pimc"):
                t, ok, best = ttt(sc, kind, target, budget_s=30.0)
                rows.append({"n_obj": n_obj, "seed": seed, "solver": kind,
                             "target": round(target, 4), "best": round(best, 4),
                             "ttt_s": round(t, 3), "reached": int(ok)})
                print("N=%-3d seed=%d %-4s 目标 %.4f  最好 %.4f  累计 %.2fs  达标 %s"
                      % (n_obj, seed, kind, target, best, t, "是" if ok else "否"), flush=True)
    os.makedirs("results", exist_ok=True)
    import csv
    with open(os.path.join("results", "ttt.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("--- 汇总 ---")
    for kind in ("sa", "pimc"):
        sel = [r for r in rows if r["solver"] == kind]
        hit = [r["ttt_s"] for r in sel if r["reached"]]
        print("%-4s 达标 %d/%d  中位 TTT %s s"
              % (kind, len(hit), len(sel),
                 ("%.2f" % statistics.median(hit)) if hit else "—"))


if __name__ == "__main__":
    main()
