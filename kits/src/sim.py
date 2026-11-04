"""长程任务闭环：观察 -> 后端规划 -> 执行一步 -> 扰动 -> 重规划。"""

import copy
import random
import time

import backend as B
import world as W

MAX_STEPS = 30


def episode(n_obj=10, T=5, planner="harness", seed=0, steps=None, slack=6, budget=0.7,
            p_jitter=0.10, p_fail=0.20, p_add=0.10, obs_sigma=1.5,
            reads=40, sweeps=600, w=0.0, stochastic=True, verbose=False,
            window=None, speed=600.0, t_grasp=0.70, start=(0.0, 0.0),
            horizon=1):
    rng = random.Random(10000 + seed)
    bodies = W.make_world(n_obj=n_obj, seed=seed)
    if steps is None:
        if window is not None:
            steps = MAX_STEPS          # 时间窗口计费时，上限按动作数兜底
        else:
            n_tg0 = sum(1 for b in bodies if b.target)
            steps = max(1, int(round(budget * n_tg0)) + slack)
    clock, travel_mm, pos = 0.0, 0.0, start
    plan_seq, plan_left = [], 0
    done, fails, consec = set(), {}, {}
    age = {}
    actions = retries = deadlocks = fallback = 0
    cat = False
    decisions = []
    be = (B.make(planner, w=w, reads=reads, sweeps=sweeps, seed=seed)
          if planner in ("sa", "qpu") else B.make(planner, w=w))

    for step in range(steps):
        if window is not None and clock >= window:
            break
        blocked = W.blocked_map(bodies)
        if plan_left <= 0:
            obs = [b.moved(rng.gauss(0, obs_sigma), rng.gauss(0, obs_sigma * 0.5)) for b in bodies]
            ob_blocked = W.blocked_map(obs)
            mem = {"fails": dict(fails), "stuck": None, "age": dict(age),
                   "damage": {i: W.damage(bodies, i) for i in range(len(bodies))}}
            stuck = [i for i in range(n_obj) if i not in done
                     and (blocked[i] & (set(range(n_obj)) - done))]
            if stuck:
                mem["stuck"] = max(stuck, key=lambda i: consec.get(i, 0))
            t0 = time.perf_counter()
            seq = be.plan(obs, ob_blocked, T, mem)
            decisions.append((time.perf_counter() - t0) * 1000.0)
            plan_seq, plan_left = list(seq), horizon
        else:
            seq = plan_seq
        plan_left -= 1
        alive_true = set(range(len(bodies))) - done
        free_true = [i for i in alive_true if not (blocked[i] & alive_true)]
        if not free_true:
            deadlocks += 1
            if deadlocks >= 2:
                cat = True
                break
            W.jitter(bodies, rng, sigma=3.0)
            continue
        target = next((o for o in seq if W.feasible(bodies, blocked, o, done)), None)
        if target is None:
            target = max(free_true, key=lambda i: W.score_of(bodies, i))
            fallback += 1
        deadlocks = 0
        if window is not None:
            trav = ((pos[0] - bodies[target].cx) ** 2
                    + (pos[1] - bodies[target].cy) ** 2) ** 0.5
            travel_mm += trav
            clock += t_grasp + trav / speed
            pos = (bodies[target].cx, bodies[target].cy)
        actions += 1
        p = W.grasp_prob(bodies, target, age.get(target, 0))
        ok = (rng.random() < p) if stochastic else True
        if ok:
            done.add(target)
            consec[target] = 0
            W.age_cluster(bodies, target, age)
        else:
            retries += 1
            plan_left = 0          # 计划前缀执行失败就立刻重规划
            fails[target] = fails.get(target, 0) + 1
            consec[target] = consec.get(target, 0) + 1
            if consec[target] >= 4:
                cat = True
        if all(i in done for i in range(len(bodies)) if bodies[i].target):
            break

        r = rng.random()
        if r < p_add and len(bodies) > 0:
            alive = [i for i in range(len(bodies)) if i not in done]
            if alive:
                tgt = rng.choice(alive)
                bodies.append(W.drop_blocker(bodies, tgt, rng))
        elif r < p_add + p_jitter:
            W.jitter(bodies, rng, sigma=2.5)
        elif r < p_add + p_jitter + p_fail and done:
            o = rng.choice(sorted(done))
            fails[o] = fails.get(o, 0)

    n_tg = sum(1 for b in bodies if b.target)
    got_tg = sum(1 for i in done if i < len(bodies) and bodies[i].target)
    dec = sorted(decisions) or [0.0]
    return {
        "planner": planner, "n_obj": n_obj, "T": T, "vars": n_obj * (T + 1), "seed": seed, "w": w,
        "done": got_tg, "completion": got_tg / max(1, n_tg),
        "actions": actions, "retries": retries, "deadlocks": deadlocks, "fallback": fallback,
        "steps": steps, "budget": budget,
        "catastrophic": int(cat), "max_age": max(age.values()) if age else 0,
        "dec_p50": dec[len(dec) // 2], "dec_p99": dec[min(len(dec) - 1, int(len(dec) * 0.99))],
        "clock": clock, "travel_mm": travel_mm, "horizon": horizon,
    }


def run(planner="harness", n_obj=10, T=5, seeds=5, **kw):
    rows = [episode(n_obj=n_obj, T=T, planner=planner, seed=s, **kw) for s in range(seeds)]
    comp = sum(r["completion"] for r in rows) / len(rows)
    cat = sum(r["catastrophic"] for r in rows) / len(rows)
    ret = sum(r["retries"] for r in rows) / len(rows)
    p99 = sum(r["dec_p99"] for r in rows) / len(rows)
    return {"planner": planner, "n_obj": n_obj, "T": T, "vars": n_obj * (T + 1),
            "completion": comp, "catastrophic": cat, "retries": ret, "p99": p99}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="harness")
    ap.add_argument("--objects", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--add", type=float, default=0.10)
    ap.add_argument("--fail", type=float, default=0.20)
    a = ap.parse_args()
    r = run(a.planner, a.objects, a.window, a.seeds, p_add=a.add, p_fail=a.fail)
    print("{planner}: vars={vars} completion={completion:.1%} catastrophic={catastrophic:.0%} "
          "retries={retries:.1f} p99={p99:.1f}ms".format(**r))
