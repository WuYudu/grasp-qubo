"""Cluttered 2-D scenes and the sequential peeling-grasp model."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

import geom

CANVAS = (280.0, 215.0)
FINGER = 46.0
MAX_OPENING = 34.0
BLOCK_EPS = 0.04


@dataclass(frozen=True)
class Body:
    cx: float
    cy: float
    w: float
    h: float
    theta: float

    @property
    def poly(self) -> np.ndarray:
        return geom.corners(self.cx, self.cy, self.w, self.h, self.theta)


@dataclass(frozen=True)
class Grasp:
    obj: int
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float

    @property
    def poly(self) -> np.ndarray:
        return geom.corners(self.cx, self.cy, self.w, self.h, self.theta)


@dataclass
class Scene:
    bodies: list[Body]
    grasps: list[list[Grasp]]
    chosen: list[int]
    blocked: list[set[int]]
    budget: int
    levels: list[int] = None
    family: str = "clutter"
    cand_blocked: list = None
    quality: list = None

    @property
    def n_obj(self) -> int:
        return len(self.bodies)

    def grasp(self, obj: int) -> Grasp:
        return self.grasps[obj][self.chosen[obj]]

    def score(self, obj: int) -> float:
        return self.grasp(obj).score


def _place(rng: random.Random, n: int, canvas, max_overlap: float, margin: float) -> list[Body]:
    bodies: list[Body] = []
    for _ in range(n):
        for _try in range(500):
            w = rng.uniform(30.0, 46.0)
            h = rng.uniform(22.0, 34.0)
            theta = rng.uniform(0.0, 180.0)
            r = 0.5 * math.hypot(w, h) + margin
            cx = rng.uniform(r, canvas[0] - r)
            cy = rng.uniform(r, canvas[1] - r)
            cand = Body(cx, cy, w, h, theta)
            if all(geom.ios(cand.poly, b.poly) <= max_overlap for b in bodies):
                bodies.append(cand)
                break
    return bodies


def _candidates(b: Body, obj: int, quality: float, finger: float, max_opening: float,
                poses=((-0.26, 0.0), (0.0, 0.0), (0.26, 0.0))) -> list[Grasp]:
    major, minor, major_deg = (b.w, b.h, b.theta) if b.w >= b.h else (b.h, b.w, b.theta + 90.0)
    wg = min(0.90 * minor, max_opening)
    fit = max(0.0, 1.0 - abs(wg - minor) / minor)
    depth = min(1.0, finger / (1.2 * minor))
    out = []
    for frac, rot in poses:
        off = frac * major
        rad = math.radians(major_deg)
        cx = b.cx + off * math.cos(rad)
        cy = b.cy + off * math.sin(rad)
        centre = max(0.0, 1.0 - abs(off) / (0.5 * major))
        tilt = max(0.0, 1.0 - abs(rot) / 45.0)
        raw = (0.50 * fit + 0.30 * centre + 0.20 * depth) * tilt
        out.append(Grasp(obj, cx, cy, wg, finger, major_deg + 90.0 + rot,
                         round(min(1.0, raw) * quality, 4)))
    return out


def make_scene(
    n_obj: int = 10,
    seed: int = 0,
    canvas=CANVAS,
    max_overlap: float = 0.45,
    budget: int | None = None,
    margin: float = 14.0,
    quality_lo: float = 0.55,
    finger: float = 26.0,
    block_eps: float = 0.12,
    poses=None,
) -> Scene:
    rng = random.Random(seed)
    bodies = _place(rng, n_obj, canvas, max_overlap, margin)
    quality = [rng.uniform(quality_lo, 1.0) for _ in bodies]
    kw = {} if poses is None else {"poses": poses}
    grasps = [_candidates(b, o, quality[o], finger, MAX_OPENING, **kw)
              for o, b in enumerate(bodies)]
    chosen = [max(range(len(grasps[o])), key=lambda c: grasps[o][c].score) for o in range(len(bodies))]
    blocked = []
    for o in range(len(bodies)):
        poly = grasps[o][chosen[o]].poly
        blocked.append({p for p, other in enumerate(bodies)
                        if p != o and geom.ios(poly, other.poly) > block_eps})
    if budget is None:
        budget = max(1, int(round(0.7 * n_obj)))
    cand_blocked = [[{p for p, other in enumerate(bodies)
                      if p != o and geom.ios(g.poly, other.poly) > block_eps}
                     for g in grasps[o]] for o in range(n_obj)]
    return Scene(bodies, grasps, chosen, blocked, budget, levels=[0] * n_obj,
                 family="clutter", cand_blocked=cand_blocked, quality=quality)


def make_pile_scene(
    n_clusters: int = 3,
    per_cluster: int = 3,
    seed: int = 0,
    canvas=CANVAS,
    budget: int | None = None,
    quality_lo: float = 0.55,
    finger: float = 26.0,
    stacked_ios: float = 0.15,
    pose_eps: float = 0.12,
    poses=None,
) -> Scene:
    """Layered piles: an object can only be grasped once everything on top of it is gone.

    cand_blocked[o][c] unions two reasons an object can be stuck: something above it
    (level rule) and the chosen grasp rectangle physically overlapping a neighbour.
    The second one is what makes the grasp pose matter, so only the candidate-level
    encoder uses it; `blocked` keeps the level rule alone.
    """
    rng = random.Random(seed)
    bodies: list[Body] = []
    levels: list[int] = []
    boxes = canvas[0] * canvas[1] / max(1, n_clusters)
    side = math.sqrt(boxes)
    cols = max(1, int(canvas[0] // side))
    for c in range(n_clusters):
        col, row = c % cols, c // cols
        ox = (col + 0.5) * canvas[0] / cols
        oy = (row + 0.5) * canvas[1] / max(1, (n_clusters + cols - 1) // cols)
        cx, cy = ox, oy
        for lv in range(per_cluster):
            w = rng.uniform(30.0, 46.0)
            h = rng.uniform(22.0, 34.0)
            theta = rng.uniform(0.0, 180.0)
            if lv > 0:
                r = 0.30 * min(w, h)
                ang = rng.uniform(0.0, 2.0 * math.pi)
                cx = cx + r * math.cos(ang)
                cy = cy + r * math.sin(ang)
            cx = min(max(cx, 0.5 * w + 4.0), canvas[0] - 0.5 * w - 4.0)
            cy = min(max(cy, 0.5 * h + 4.0), canvas[1] - 0.5 * h - 4.0)
            bodies.append(Body(cx, cy, w, h, theta))
            levels.append(lv)
    n_obj = len(bodies)
    quality = [rng.uniform(quality_lo, 1.0) for _ in bodies]
    kw = {} if poses is None else {"poses": poses}
    grasps = [_candidates(b, o, quality[o], finger, MAX_OPENING, **kw) for o, b in enumerate(bodies)]
    chosen = [max(range(len(grasps[o])), key=lambda c: grasps[o][c].score) for o in range(n_obj)]
    blocked = []
    for o in range(n_obj):
        blocked.append({p for p in range(n_obj)
                        if p != o and levels[p] > levels[o]
                        and geom.ios(bodies[p].poly, bodies[o].poly) > stacked_ios})
    if budget is None:
        budget = max(1, int(round(0.7 * n_obj)))
    cand_blocked = []
    for o in range(n_obj):
        per = []
        for g in grasps[o]:
            pose = {p for p in range(n_obj)
                    if p != o and levels[p] >= levels[o]
                    and geom.ios(g.poly, bodies[p].poly) > pose_eps}
            per.append(blocked[o] | pose)
        cand_blocked.append(per)
    return Scene(bodies, grasps, chosen, blocked, budget, levels=levels,
                 family="piles", cand_blocked=cand_blocked, quality=quality)

