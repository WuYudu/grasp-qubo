"""Pick-and-place ordering: object -> tray slot -> step, as one QUBO.

The scene is deliberately axis-aligned so that the assignment layer, not the
geometry, is what the experiment is about. Every object has one top-down grasp;
an object is blocked while another object in the same pile sits above it.
"""

import math
import random

CANVAS = (320.0, 240.0)
SLOT_W = (40.0, 40.0, 30.0, 30.0, 40.0, 40.0)
SLOT_GAP = 8.0
TRAY_Y = -70.0
MU = 0.12         # weight of transfer distance, per 100 canvas units


class Obj:
    __slots__ = ("i", "cx", "cy", "w", "h", "level", "cluster", "score", "empty")

    def __init__(self, i, cx, cy, w, h, level, cluster, score):
        self.i = i
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.level = level
        self.cluster = cluster
        self.score = score
        self.empty = False

    @property
    def x0(self):
        return self.cx - self.w / 2.0

    @property
    def x1(self):
        return self.cx + self.w / 2.0


class Scene:
    """Objects in a bin plus a row of tray slots."""

    def __init__(self, objs, blocked, n_slot=6, n_done=0):
        self.objs = objs
        self.blocked = blocked            # obj id -> set of ids that must go first
        self.n_slot = n_slot
        self.n_done = n_done

    @property
    def n_obj(self):
        return len(self.objs)

    def slot_x(self, c):
        offset = sum(SLOT_W[:c]) + SLOT_GAP * c
        total = sum(SLOT_W) + SLOT_GAP * (len(SLOT_W) - 1)
        return CANVAS[0] / 2.0 - total / 2.0 + offset + SLOT_W[c] / 2.0

    def fit(self, o, c):
        return self.objs[o].w <= SLOT_W[c] - 2.0

    def cost(self, o, c):
        b = self.objs[o]
        return abs(b.cx - self.slot_x(c)) + abs(b.cy - TRAY_Y)

    def value(self, o, c):
        return self.objs[o].score - MU * self.cost(o, c) / 100.0

    def best_slot(self, o, used):
        pick = None
        for c in range(self.n_slot):
            if c in used or not self.fit(o, c):
                continue
            if pick is None or self.value(o, c) > self.value(o, pick):
                pick = c
        return pick


def _score(rng, w, h):
    fit = min(1.0, 44.0 / max(w, 1.0)) * min(1.0, 40.0 / max(h, 1.0))
    quality = 0.55 + 0.45 * rng.random()
    return round((0.62 * fit + 0.38 * quality) * 100.0) / 100.0


def make_scene(n_obj=12, seed=0, cluster=3, canvas=CANVAS, n_slot=6):
    """Piles of axis-aligned boxes; blocking comes from the pile order."""
    rng = random.Random(seed)
    objs, blocked = [], {}
    n_cl = max(1, int(round(n_obj / float(cluster))))
    row_w = canvas[0] / float(n_cl)
    k = 0
    for cl in range(n_cl):
        cx0 = row_w * (cl + 0.5)
        level = 0
        while k < n_obj and level < cluster:
            w = round(rng.uniform(26.0, 46.0), 1)
            h = round(rng.uniform(18.0, 32.0), 1)
            jx = rng.uniform(-0.18, 0.18) * row_w
            cy = canvas[1] - 40.0 - level * 22.0 - rng.uniform(0.0, 6.0)
            objs.append(Obj(k, round(cx0 + jx, 1), round(cy, 1), w, h, level, cl, _score(rng, w, h)))
            k += 1
            level += 1
    for a in objs:
        blocked[a.i] = set()
    for a in objs:
        for b in objs:
            if a.cluster != b.cluster or b.level <= a.level:
                continue
            lo = max(a.x0, b.x0)
            hi = min(a.x1, b.x1)
            if hi - lo > 0.2 * min(a.w, b.w):
                blocked[a.i].add(b.i)
    return Scene(objs, blocked, n_slot=n_slot)


def feasible_order(sc, order):
    """True when the list can be executed left to right."""
    done = set()
    for o in order:
        if sc.blocked[o] - done:
            return False
        done.add(o)
    return True
