"""A rawer bin: irregular bodies, a real gripper, perception error, recovery moves."""
import math
import random
from dataclasses import dataclass, field

import numpy as np

import geom

JAW_MAX = 46.0          # widest opening of the parallel gripper, px
JAW_MIN = 8.0
FINGER_T = 3.0          # finger thickness
FINGER_L = 52.0         # finger length (== approach corridor length)
CORRIDOR_PAD = 2.0
AIR_PROBE = 20.0      # how far above the grasp we actually test for obstruction
N_ANGLES = 2          # candidate rotations per body (2 x 3 offsets = 6 grasps)
EPS = 0.02


def hull(points):
    pts = sorted({(round(p[0], 4), round(p[1], 4)) for p in points})
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


@dataclass
class Body:
    poly: np.ndarray
    mass: float = 1.0
    level: int = 0
    cluster: int = 0
    target: bool = False
    fragile: bool = False

    def __post_init__(self):
        x, y = self.poly[:, 0], self.poly[:, 1]
        self.cx, self.cy = float(x.mean()), float(y.mean())
        self.w, self.h = float(x.max() - x.min()), float(y.max() - y.min())
        self.area = geom.area(self.poly)

    def moved(self, dx, dy, rot=0.0):
        p = self.poly - np.array([self.cx, self.cy])
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            p = p @ np.array([[c, s], [-s, c]])
        p = p + np.array([self.cx + dx, self.cy + dy])
        return Body(p, self.mass, self.level, self.cluster, self.target, self.fragile)


def _shape(rng, kind, w, h):
    if kind == "rect":
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    elif kind == "trap":
        pts = [(-w / 2, -h / 2), (w / 2, -h * 0.3), (w * 0.2, h / 2), (-w * 0.45, h * 0.35)]
    elif kind == "ellipse":
        pts = [(w / 2 * math.cos(t), h / 2 * math.sin(t))
               for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    else:                                      # blob
        pts = [(rng.uniform(-0.5, 0.5) * w, rng.uniform(-0.5, 0.5) * h) for _ in range(9)]
    return hull(pts)


def _place_bin(rng, n_obj, canvas, cluster_size, overlap=0.28, kinds=("rect", "trap", "ellipse", "blob")):
    bodies, cl = [], 0
    cw, ch = canvas
    cols = max(1, int(round(math.sqrt(n_obj / cluster_size))))
    rows = max(1, int(math.ceil(n_obj / (cols * cluster_size))))
    cw, ch = cw / cols, ch / rows
    for c in range(cols * rows):
        n = min(cluster_size, n_obj - len(bodies))
        if n <= 0:
            break
        gx = (c % cols) * cw + cw / 2
        gy = (c // cols) * ch + ch / 2
        for lv in range(n):
            w = rng.uniform(26, 44)
            h = rng.uniform(18, 32)
            kind = rng.choice(kinds)
            poly = _shape(rng, kind, w, h)
            jx = rng.uniform(-5, 5)
            jy = rng.uniform(-6, 6)
            poly = poly + np.array([gx + jx, gy + (n - 1 - lv) * h * 0.55 + jy])
            bodies.append(Body(poly, mass=rng.uniform(0.4, 2.0), level=lv, cluster=cl))
        cl += 1
    return bodies


@dataclass
class Grasp:
    obj: int
    cx: float
    cy: float
    width: float
    theta: float          # jaw opening direction
    approach: float       # approach direction (radians), usually pointing out of the bin
    fit: float = 0.0
    score: float = 0.0
    blocked: tuple = ()

    def corridor(self):
        """Polygon swept by the two fingers while approaching."""
        ux, uy = math.cos(self.approach), math.sin(self.approach)
        vx, vy = -uy, ux
        half = self.width / 2 + FINGER_T + CORRIDOR_PAD
        pts = []
        for s in (-1, 1):
            for t in (self.width / 2 + 2.0, self.width / 2 + 2.0 + AIR_PROBE):
                pts.append((self.cx + ux * t + vx * half * s,
                            self.cy + uy * t + vy * half * s))
        return hull(pts)

    def jaw(self):
        hw = self.width / 2
        ux, uy = math.cos(self.approach), math.sin(self.approach)
        vx, vy = -uy, ux
        pts = []
        for t in (0.0, 1.0):
            for s in (-1, 1):
                pts.append((self.cx + ux * t + vx * hw * s,
                            self.cy + uy * t + vy * hw * s))
        return hull(pts)


def candidates(body, obj_id, rng=None, n_angles=None):
    n_angles = N_ANGLES if n_angles is None else n_angles
    """Parallel-jaw grasps across the minor axis, three offsets, slightly rotated."""
    ang = math.atan2(body.poly[:, 1].std(), body.poly[:, 0].std())
    out = []
    axis = ang
    for k in range(n_angles):
        for off in (-0.22, 0.0, 0.22):
            theta = axis + (k - 1) * 0.25
            cx = body.cx + math.cos(axis) * off * body.w
            cy = body.cy + math.sin(axis) * off * body.h
            width = min(JAW_MAX, max(JAW_MIN, 0.82 * body.h))
            appr = theta + math.pi / 2
            if math.sin(appr) > 0:          # approach must come from above
                appr += math.pi
            g = Grasp(obj_id, cx, cy, width, theta, approach=appr)
            corr = g.corridor()
            hit = geom.inter_area(corr, body.poly) / max(1e-9, geom.area(corr))
            g.fit = float(max(0.0, 1.0 - 2.5 * hit))
            out.append(g)
    return out


def score_grasps(bodies, grasps):
    """Score every grasp: fit against its own body, corridor interference, clearance."""
    for g in grasps:
        corr = g.corridor()
        pen, top = 0.0, []
        for p, b in enumerate(bodies):
            if p == g.obj:
                continue
            ia = geom.inter_area(corr, b.poly)
            if ia > 0:
                pen += ia / max(1e-9, geom.area(corr))
                top.append(p)
        centre = 1.0 - min(1.0, abs(g.cx - bodies[g.obj].cx) / max(1e-6, bodies[g.obj].w))
        width_ok = 1.0 if JAW_MIN <= g.width <= JAW_MAX else 0.0
        depth = 1.0 - min(1.0, bodies[g.obj].mass / 3.0)
        g.score = float(max(0.0, (0.46 * g.fit + 0.22 * centre + 0.14 * depth
                                  + 0.18 * width_ok) - 1.5 * pen))
        g.blocked = tuple(sorted(set(top)))


@dataclass
class Scene:
    bodies: list
    grasps: list                  # per body: list of Grasp
    targets: set = field(default_factory=set)
    fragile: set = field(default_factory=set)
    canvas: tuple = (320.0, 240.0)

    @property
    def n(self):
        return len(self.bodies)

    def best(self, o):
        gs = self.grasps[o]
        return max(gs, key=lambda g: g.score)

    def above(self, o):
        """Bodies in the same stack that sit higher (smaller cy)."""
        b = self.bodies[o]
        return {p for p, q in enumerate(self.bodies)
                if p != o and q.cluster == b.cluster and q.cy < b.cy - 1.0}

    def blocked(self, o):
        return set(self.above(o)) | set(self.best(o).blocked)


def make_scene(n_obj=24, seed=0, cluster=3, canvas=(320.0, 240.0), n_targets=8,
               n_fragile=3, kinds=("rect", "trap", "ellipse", "blob")):
    rng = random.Random(seed)
    bodies = _place_bin(rng, n_obj, canvas, cluster, kinds=kinds)
    grasps = [candidates(b, i) for i, b in enumerate(bodies)]
    score_grasps(bodies, [g for gs in grasps for g in gs])
    sc = Scene(bodies, grasps, canvas=canvas)
    order = sorted(range(n_obj), key=lambda i: -sc.best(i).score)
    sc.targets = set(order[:n_targets])
    sc.fragile = set(order[-n_fragile:])
    for i in sc.fragile:
        bodies[i].fragile = True
    for i in sc.targets:
        bodies[i].target = True
    return sc


def rebuild(sc, bodies):
    """Recompute grasps and scores after the pile moved."""
    grasps = [candidates(b, i) for i, b in enumerate(bodies)]
    score_grasps(bodies, [g for gs in grasps for g in gs])
    return Scene(bodies, grasps, set(sc.targets), set(sc.fragile), sc.canvas)
