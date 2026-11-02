"""按时限分拣：三类物料、三个料箱、两种抓取位姿，比的是每分钟分拣多少个。

和前几版的不同：目标不再是"分数"，而是"在固定时间窗口里分拣多少件"。物体要送到
各自类别的料箱，因此每个动作的时间由搬运距离决定；抓取位姿有两档（快但容易掉、慢但稳），
选择位姿是一项在线决策。
"""

import random
import statistics
import time

import world100

CLASSES = ("A", "B", "C")
BIN_DX = {"A": -240.0, "B": 0.0, "C": 240.0}
SPEED = 700.0          # 画布单位/秒
T_GRASP = 0.40
T_RELEASE = 0.30
DEADLINE = 60.0        # 秒


def make_world(n=100, seed=0):
    w = world100.make_world(n_obj=n, seed=seed, cluster=3, n_slot=6)
    rng = random.Random(seed + 7)
    w.cls = {o: CLASSES[rng.randrange(3)] for o in range(n)}
    return w


def bin_xy(w, o):
    return w.mid_x + BIN_DX[w.cls[o]], 20.0


def cycle_time(w, o):
    b = w.objs[o]
    bx, by = bin_xy(w, o)
    carry = abs(b.cx - bx) + abs(b.cy - by)
    return T_GRASP + T_RELEASE + carry / SPEED


def poses(w, o):
    """两档位姿：快档开口窄、失手多；稳档多花 0.45 秒、失手少。"""
    b = w.objs[o]
    crowd = sum(1 for o2 in w.remaining() if o2 != o and abs(w.objs[o2].cx - b.cx) < 30)
    fast = {"name": "fast", "p": max(0.55, 0.88 - 0.05 * crowd), "extra": 0.0}
    safe = {"name": "safe", "p": max(0.70, 0.98 - 0.02 * crowd), "extra": 0.45}
    return fast, safe


def choose_pose(w, o, policy):
    fast, safe = poses(w, o)
    if policy == "fast":
        return fast
    if policy == "safe":
        return safe
    return safe if fast["p"] < 0.78 else fast


def pick(w, mode, done):
    """返回 (object, pose_policy_hint)；mode 决定按什么排序。"""
    avail = [o for o in w.remaining() if not (w.blocked[o] - done)]
    if not avail:
        return None
    if mode == "throughput":
        return max(avail, key=lambda o: w.objs[o].score / cycle_time(w, o))
    return max(avail, key=lambda o: w.objs[o].score)


def episode(n_obj, planner, policy, seed, deadline=DEADLINE):
    w = make_world(n_obj, seed)
    rng = random.Random(1000 + seed)
    done, t, ok, bad, acts = set(), 0.0, 0, 0, 0
    while t < deadline:
        o = pick(w, planner, done)
        if o is None:
            break
        pose = choose_pose(w, o, policy)
        ct = cycle_time(w, o) + pose["extra"]
        if t + ct > deadline:
            break
        t += ct
        acts += 1
        if rng.random() < pose["p"]:
            w.objs[o].gone = True
            done.add(o)
            ok += 1
        else:
            bad += 1
    return {"n_obj": n_obj, "seed": seed, "planner": planner, "policy": policy,
            "sorted": ok, "fails": bad, "actions": acts, "makespan": round(t, 2),
            "per_min": round(60.0 * ok / max(t, 1e-9), 2)}


def main():
    t0 = time.perf_counter()
    rows = []
    for planner in ("score", "throughput"):
        for policy in ("fast", "safe", "adaptive"):
            for seed in (0, 1):
                r = episode(100, planner, policy, seed)
                rows.append(r)
                print("%-10s %-9s seed=%d  分拣 %3d 件  失手 %2d  用时 %5.1fs  %.2f 件/分钟"
                      % (planner, policy, seed, r["sorted"], r["fails"], r["makespan"], r["per_min"]),
                      flush=True)
    print("--- 汇总（两个种子均值） ---")
    for planner in ("score", "throughput"):
        for policy in ("fast", "safe", "adaptive"):
            sel = [r for r in rows if r["planner"] == planner and r["policy"] == policy]
            print("%-10s %-9s 分拣 %.1f 件  件/分钟 %.2f  失手 %.1f"
                  % (planner, policy, statistics.mean(r["sorted"] for r in sel),
                     statistics.mean(r["per_min"] for r in sel),
                     statistics.mean(r["fails"] for r in sel)))
    print("总耗时 %.1fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
