"""箱内堆叠场景：物体是轴对齐矩形，上层压住下层，抓取需要一条向上的通道。"""

import math
import random
from dataclasses import dataclass

EPS = 1e-9
CANVAS = (260.0, 200.0)
AIR = 18.0
N_SLOT = 0
MASS_RANGE = (0.6, 2.4)


def overlap(a, b):
    """交叠面积 / 较小面积。"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(small, EPS)


@dataclass
class Body:
    cx: float
    cy: float
    w: float
    h: float
    level: int = 0
    cluster: int = 0
    quality: float = 1.0
    mass: float = 1.0
    fragile: bool = False
    target: bool = True

    def rect(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def corridor(self):
        top = self.cy - self.h / 2
        return (self.cx - self.w / 2, top - AIR, self.cx + self.w / 2, top - 1.0)

    def moved(self, dx, dy, rot=0.0):
        return Body(self.cx + dx, self.cy + dy, self.w, self.h, self.level,
                    self.cluster, self.quality, self.mass, self.fragile, self.target)


def make_world(n_obj=10, seed=0, n_cluster=3):
    """按若干堆摆放，堆内上下叠压。"""
    rng = random.Random(seed)
    bodies = []
    per = max(1, int(math.ceil(n_obj / n_cluster)))
    for c in range(n_cluster):
        x = 40.0 + c * (CANVAS[0] - 60.0) / max(1, n_cluster - 1) if n_cluster > 1 else CANVAS[0] / 2
        y = CANVAS[1] - 25.0
        for i in range(per):
            if len(bodies) >= n_obj:
                break
            w = rng.uniform(28.0, 52.0)
            h = rng.uniform(20.0, 34.0)
            bodies.append(Body(
                cx=min(max(x + rng.uniform(-6, 6), w / 2 + 4), CANVAS[0] - w / 2 - 4),
                cy=y - h / 2,
                w=w, h=h, level=i, cluster=c,
                quality=rng.uniform(0.55, 1.0),
                mass=rng.uniform(*MASS_RANGE),
                fragile=rng.random() < 0.15,
                target=rng.random() > 0.30,
            ))
            y -= h + rng.uniform(1.0, 3.0)
    return bodies


def blocked_map(bodies, eps=0.10):
    """每个物体的阻挡集合：同堆中层级更高的，或挡住向上通道的。"""
    bl = []
    for i, b in enumerate(bodies):
        s = {j for j, p in enumerate(bodies)
             if j != i and p.level > b.level and p.cluster == b.cluster}
        s |= {j for j, p in enumerate(bodies)
              if j != i and p.level > b.level and overlap(b.corridor(), p.rect()) > eps}
        bl.append(s)
    return bl


def score_of(bodies, i):
    b = bodies[i]
    if not b.target:
        return 0.0
    depth = 1.0 if b.level == 0 else max(0.35, 1.0 - 0.18 * b.level)
    centre = 1.0 - abs(b.cx - CANVAS[0] / 2) / (CANVAS[0] / 2)
    fit = min(1.0, b.w / 40.0)
    return b.quality * (0.50 * fit + 0.30 * centre + 0.20 * depth)


def feasible(bodies, blocked, i, done):
    alive = set(range(len(bodies))) - set(done)
    return i not in done and not (blocked[i] & alive) and score_of(bodies, i) > 0


def grasp_prob(bodies, i, age=0):
    b = bodies[i]
    p = 0.94 - 0.10 * max(0.0, b.mass - 1.5) - (0.18 if b.fragile else 0.0) - 0.05 * age
    return max(0.25, min(0.97, p))


def damage(bodies, i):
    """搬走 i 会让同堆下面还留着的目标物体错位一次，代价计入目标函数。"""
    b = bodies[i]
    return sum(1 for j, q in enumerate(bodies)
               if j != i and q.cluster == b.cluster and q.level < b.level and q.target)


def age_cluster(bodies, i, age):
    b = bodies[i]
    for j, q in enumerate(bodies):
        if j != i and q.cluster == b.cluster and q.level < b.level:
            age[j] = age.get(j, 0) + 1


def drop_blocker(bodies, target, rng, seed_level=None):
    """在中途新增一个压住 target 的物体。"""
    t = bodies[target]
    lvl = t.level + 1 if seed_level is None else seed_level
    for b in bodies:
        if b.level >= lvl:
            b.level += 1
    w = min(t.w * 0.9, 40.0)
    h = 18.0
    top = t.cy - t.h / 2
    return Body(cx=t.cx + rng.uniform(-4, 4), cy=top - h / 2 - 1.0, w=w, h=h,
                level=lvl, cluster=t.cluster, quality=rng.uniform(0.55, 1.0),
                mass=rng.uniform(*MASS_RANGE), fragile=False, target=False)


def jitter(bodies, rng, sigma=2.0, k=3):
    idx = rng.sample(range(len(bodies)), min(k, len(bodies)))
    for i in idx:
        b = bodies[i]
        bodies[i] = b.moved(rng.gauss(0, sigma), rng.gauss(0, sigma * 0.6))
