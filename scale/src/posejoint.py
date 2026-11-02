"""通道几何 + 位姿选择联合优化（小规模精确解）。

每个物体有多个候选抓取位姿，每个位姿有自己的指杆通道（corridor）。某一物体在某位姿下
会挡住哪些别的物体，由"通道多边形与对方实体相交"决定。位姿与顺序一起优化，与"只用
最高分位姿"对比。
"""

import itertools
import random

import geom
import world2

EPS = 0.06


def build(sc, k=3):
    """每个物体保留分数最高的前 k 个位姿，并算出每个位姿的阻塞集。"""
    cand, blocked = [], []
    for o, gs in enumerate(sc.grasps):
        gs = sorted(gs, key=lambda g: -g.score)[:k]
        cand.append(gs)
        row = []
        for g in gs:
            cor = g.corridor()
            blk = set(sc.above(o))
            for p, b in enumerate(sc.bodies):
                if p == o:
                    continue
                if geom.ios(cor, b.poly) > EPS:
                    blk.add(p)
            row.append(blk)
        blocked.append(row)
    return cand, blocked


def feasible_plan(sc, plan, cand, blocked):
    done, used_slot = set(), set()
    for s, (o, c) in enumerate(plan):
        if o in done:
            return False
        if blocked[o][c] - done:
            return False
        done.add(o)
    return True


def score_plan(sc, plan, cand, mu=0.05):
    return sum(cand[o][c].score - mu * (abs(sc.bodies[o].cx) + abs(sc.bodies[o].cy)) / 100.0
               for o, c in plan)


def brute(sc, cand, blocked, T, mu=0.05):
    N = len(cand)
    best, arg = None, None
    opts = [(o, c) for o in range(N) for c in range(len(cand[o]))]
    for r in range(0, min(T, N) + 1):
        for combo in itertools.permutations(opts, r):
            if len({o for o, _ in combo}) != r:
                continue
            if not feasible_plan(sc, list(combo), cand, blocked):
                continue
            v = score_plan(sc, list(combo), cand, mu)
            if best is None or v > best:
                best, arg = v, list(combo)
    return best, arg


def solve(sc, cand, blocked, T, joint=True, mu=0.05):
    """线性模型：位置 = 步；joint=False 时每个物体只允许最高分位姿。"""
    import pulp
    N = len(cand)
    prob = pulp.LpProblem("pose", pulp.LpMaximize)
    y = {}
    for o in range(N):
        for c in range(len(cand[o])):
            if not joint and c != 0:
                continue
            for s in range(T):
                y[(o, c, s)] = prob.add_variable("y_%d_%d_%d" % (o, c, s), cat=pulp.LpBinary)
    prob += pulp.lpSum(score_plan(sc, [(o, c)], cand, mu) * y[(o, c, s)]
                       for (o, c, s) in y)
    for s in range(T):
        prob += pulp.lpSum(v for (o, c, ss), v in y.items() if ss == s) <= 1
    for o in range(N):
        prob += pulp.lpSum(v for (oo, c, s), v in y.items() if oo == o) <= 1
    for (o, c, s), v in list(y.items()):
        for p in blocked[o][c]:
            prob += v + pulp.lpSum(yy for (oo, cc, ss), yy in y.items()
                                   if oo == p and ss >= s) <= 1
    prob.solve(pulp.HiGHS(msg=False, timeLimit=10))
    plan = sorted(((s, o, c) for (o, c, s), v in y.items()
                   if pulp.value(v) and pulp.value(v) > 0.5))
    return score_plan(sc, [(o, c) for _, o, c in plan], cand, mu), [(o, c) for _, o, c in plan]


def check():
    rng = random.Random(0)
    sc = world2.make_scene(n_obj=4, seed=3, cluster=2, canvas=(260.0, 200.0),
                           n_targets=3, n_fragile=1)
    cand, blocked = build(sc, k=2)
    T = 2
    b, arg = brute(sc, cand, blocked, T)
    v, plan = solve(sc, cand, blocked, T, joint=True)
    ok = abs(b - v) < 1e-6
    print("check: brute %.4f  milp %.4f  %s   plan=%s" % (b, v, "OK" if ok else "MISMATCH", plan))
    return ok


def main():
    check()
    for n in (6, 8, 10):
        for seed in (0, 1):
            sc = world2.make_scene(n_obj=n, seed=seed, cluster=2, canvas=(320.0, 220.0),
                                   n_targets=int(0.6 * n), n_fragile=1)
            cand, blocked = build(sc, k=3)
            T = min(3, n)
            v_fix, _ = solve(sc, cand, blocked, T, joint=False)
            v_jnt, plan = solve(sc, cand, blocked, T, joint=True)
            gain = 100.0 * (v_jnt - v_fix) / max(abs(v_fix), 1e-9)
            print("N=%-3d seed=%d  固定最高分位姿 %.4f  位姿+顺序 %.4f  提升 %+.1f%%"
                  % (n, seed, v_fix, v_jnt, gain), flush=True)


if __name__ == "__main__":
    main()
