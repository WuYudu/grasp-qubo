"""前缀执行 + 重观测的闭环，按墙钟预算收尾。"""
import copy
import random


def episode(world, backend, h=2, budget_s=18.0, steps=40, seed=0, p_fix=0.55):
    rng = random.Random(seed)
    w = copy.deepcopy(world)
    plan, left, replans = [], 0, 0
    while w.clock < budget_s and w.actions < steps:
        if left <= 0 or not plan:
            plan = backend.plan(w, T=6, h=h)
            left = h
            replans += 1
            if not plan:
                w.observe(rng, p_fix)
                left = 0
                continue
        pi, si = plan.pop(0)
        left -= 1
        w.apply(pi, si)
        if left <= 0:
            w.observe(rng, p_fix)
    s = w.summary()
    s["replans"] = replans
    s["h"] = h
    return s