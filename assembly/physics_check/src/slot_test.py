"""在 MuJoCo 里验证一件事：我们的"只有同类槽位能放"是物理规律还是约定。"""
import mujoco
import numpy as np

KIND_W = {"A": 0.080, "B": 0.120, "C": 0.160}
T, HH, BOTTOM, PH = 0.005, 0.025, 0.005, 0.030


def scene(pw, W):
    hw = W / 2 + T
    top = BOTTOM + 2 * HH
    walls = []
    for sx, sy, px, py in ((T, hw, hw, 0.0), (T, hw, -hw, 0.0),
                           (hw, T, 0.0, hw), (hw, T, 0.0, -hw)):
        walls.append(
            '<geom type="box" size="{:.4f} {:.4f} {:.4f}" pos="{:.4f} {:.4f} {:.4f}"/>'
            .format(sx, sy, HH, px, py, BOTTOM + HH))
    xml = (
        '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>'
        '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 {:.4f}"/>'
        '<geom name="plate" type="box" size="0.25 0.25 {:.4f}" pos="0 0 {:.4f}" rgba="0.8 0.8 0.8 1"/>'
        '{}'
        '<body name="part" pos="0.004 0.003 {:.4f}"><freejoint/>'
        '<geom type="box" size="{:.4f} {:.4f} {:.4f}" rgba="0.2 0.4 0.9 1"/></body>'
        '</worldbody></mujoco>'
    ).format(BOTTOM - 0.1, BOTTOM / 2, BOTTOM / 2, "".join(walls),
             top + PH + 0.02, pw / 2, pw / 2, PH)
    return mujoco.MjModel.from_xml_string(xml)


def drop(pw, W):
    m = scene(pw, W)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(2000):
        mujoco.mj_step(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
    x, y, z = d.xpos[bid]
    return z, float(np.hypot(x, y)), z < BOTTOM + 2 * HH + 0.005


print("零件   槽口   落位高度  水平偏移  落进去了  我们的规则")
for pk, pw in KIND_W.items():
    for sk, W in KIND_W.items():
        z, r, inside = drop(pw, W)
        print("%s %.3f  %s %.3f  %7.4f  %7.4f   %s        %s"
              % (pk, pw, sk, W, z, r, "是" if inside else "否", "匹配" if pk == sk else "报废"))