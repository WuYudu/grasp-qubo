"""把 assembly_qubo 的计划拿到 MuJoCo 里执行：计划层说成功的，物理层真成功吗。"""
import mujoco
import numpy as np

import backend
import world as W

CL, T, HH, BOTTOM, PH = 0.004, 0.005, 0.025, 0.005, 0.030
KW = {"A": 0.080, "B": 0.120, "C": 0.160}
NP, NS = 6, 8
GRID = [(-0.45 + 0.30 * (j % 4), -0.05 + 0.22 * (j // 4)) for j in range(NS)]


def xml(w):
    slots = []
    for j in range(NS):
        Wd = KW[w.slots[j].kind] + CL
        hw = Wd / 2 + T
        cx, cy = GRID[j]
        for sx, sy, px, py in ((T, hw, hw, 0.0), (T, hw, -hw, 0.0), (hw, T, 0.0, hw), (hw, T, 0.0, -hw)):
            slots.append('<geom type="box" size="{:.4f} {:.4f} {:.4f}" pos="{:.4f} {:.4f} {:.4f}"/>'
                         .format(sx, sy, HH, cx + px, cy + py, BOTTOM + HH))
    parts = []
    for i in range(NP):
        pw = KW[w.parts[i].kind]
        parts.append('<body name="p{}" pos="{:.3f} {:.3f} {:.3f}"><freejoint/>'
                     '<geom type="box" size="{:.4f} {:.4f} {:.4f}" rgba="0.2 0.4 0.9 1"/></body>'
                     .format(i, -0.75 + 0.10 * i, 0.55, 0.06, pw / 2, pw / 2, PH))
    return ('<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>'
            '<geom name="floor" type="plane" size="3 3 0.1" pos="0 0 {:.4f}"/>'
            '<geom name="plate" type="box" size="0.9 0.8 {:.4f}" pos="0 0 {:.4f}"/>{} {}</worldbody></mujoco>'
            ).format(BOTTOM - 0.1, BOTTOM / 2, BOTTOM / 2, "".join(slots), "".join(parts))


def run(w, plan, order_tag):
    m = mujoco.MjModel.from_xml_string(xml(w))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ok, rows = 0, []
    for pi, si in plan:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "p%d" % pi)
        adr = m.jnt_qposadr[m.body_jntadr[bid]]
        cx, cy = GRID[si]
        z_top = BOTTOM + 2 * HH + PH + 0.03
        p0 = np.array(d.xpos[bid], dtype=float)
        for k in range(150):                      # 抬起、平移、对准槽口
            t = k / 149.0
            d.qpos[adr:adr + 3] = [(1 - t) * p0[0] + t * cx, (1 - t) * p0[1] + t * cy, p0[2] + 0.05 * t]
            d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(m, d)
        for k in range(80):                       # 放下去
            t = k / 79.0
            d.qpos[adr + 2] = (p0[2] + 0.05) * (1 - t) + BOTTOM * t
            mujoco.mj_step(m, d)
        for _ in range(600):                      # 释放后自由沉降 1.2 秒
            mujoco.mj_step(m, d)
        x, y, z = d.xpos[bid]
        inside = (abs(x - cx) < 0.010) and (abs(y - cy) < 0.010) and (z < BOTTOM + 2 * HH + 0.004)
        ok += inside
        rows.append((pi, si, w.parts[pi].kind, w.slots[si].kind, z, np.hypot(x - cx, y - cy), inside))
    print("%s：计划 %d 步，物理成功 %d 步" % (order_tag, len(plan), ok))
    for pi, si, pk, sk, z, r, ins in rows:
        print("   零件 %d(%s) -> 槽 %d(%s)  落位 %.4f  偏差 %.4f  %s"
              % (pi, pk, si, sk, z, r, "成功" if ins else "失败"))
    return ok, len(plan)


def main():
    for seed in (0, 1):
        w = W.World(n_part=NP, n_slot=NS, seed=seed, p_stale=0.0, q_correct=0.6, obs_cost=0.25)
        plan = backend.make("risk").plan(w, T=6, h=2)
        if plan:
            run(w, plan, "seed %d 计划（同类配对）" % seed)
    w = W.World(n_part=NP, n_slot=NS, seed=0, p_stale=0.0, q_correct=0.6, obs_cost=0.25)
    cross = [(pi, si) for pi, si in [(0, 1), (1, 4)]]
    print("\n故意跨类（我们模型判报废）：", [(pi, si, w.parts[pi].kind, w.slots[si].kind) for pi, si in cross])
    run(w, cross, "跨类放置")


if __name__ == "__main__":
    main()