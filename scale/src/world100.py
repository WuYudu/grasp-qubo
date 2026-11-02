"""A hundred objects, six tray slots, and placements that do not always land."""

import math
import random

WIDTH = 40.0
TRAY_Y = 20.0
MU = 0.03
P_SLIP = 0.10        # falls back into the bin
P_OFF = 0.16         # lands but leans into the neighbour slot


class Obj:
    __slots__ = ("i", "cx", "cy", "w", "h", "level", "cluster", "score", "gone")

    def __init__(self, i, cx, cy, w, h, level, cluster, score):
        self.i = i
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.level = level
        self.cluster = cluster
        self.score = score
        self.gone = False

    @property
    def x0(self):
        return self.cx - self.w / 2.0

    @property
    def x1(self):
        return self.cx + self.w / 2.0


class World:
    def __init__(self, objs, blocked, n_slot=6):
        self.objs = objs
        self.blocked = blocked
        self.n_slot = n_slot
        self.slot = [None] * n_slot        # object id occupying the slot
        self.dirty = [False] * n_slot      # slot spoiled by a leaning placement
        self.placed = []
        self.fails = 0
        self.carry_miss = 0
        self.mid_x = (min(o.cx for o in objs) + max(o.cx for o in objs)) / 2.0 if objs else 0.0

    # ---- geometry helpers -------------------------------------------------
    @property
    def n_obj(self):
        return len(self.objs)

    def slot_x(self, c):
        span = self.n_slot * WIDTH
        return self.mid_x - span / 2.0 + (c + 0.5) * WIDTH

    def free_slots(self):
        return [c for c in range(self.n_slot) if self.slot[c] is None and not self.dirty[c]]

    def cost(self, o, c):
        b = self.objs[o]
        return abs(b.cx - self.slot_x(c)) + abs(b.cy - TRAY_Y)

    def best_cost(self, o):
        return min(self.cost(o, c) for c in self.free_slots()) if self.free_slots() else 0.0

    def value(self, o, c=None):
        c = c if c is not None else self.free_slots()[0]
        return self.objs[o].score - MU * self.cost(o, c) / 100.0

    # ---- state ------------------------------------------------------------
    def remaining(self):
        return [o for o in range(self.n_obj) if not self.objs[o].gone]

    def available(self, done):
        return [o for o in self.remaining() if not (self.blocked[o] - done)]

    def apply(self, o, c):
        self.objs[o].gone = True
        self.slot[c] = o
        self.placed.append((o, c))

    def new_tray(self):
        """The full tray is swapped out; spoiled slots go with it."""
        self.slot = [None] * self.n_slot
        self.dirty = [False] * self.n_slot

    def place_with_failure(self, o, c, rng, p_slip=P_SLIP, p_off=P_OFF):
        """Returns 'ok', 'slip' (back in the bin) or 'off' (slot spoiled)."""
        b = self.objs[o]
        tight = max(0.0, (WIDTH - b.w) / WIDTH)          # narrow slot, wider object
        near = sum(1 for c2 in (c - 1, c + 1) if 0 <= c2 < self.n_slot and self.slot[c2] is not None)
        pc = 1.0 + 1.2 * tight + 0.25 * near
        if rng.random() < p_slip * pc:
            self.fails += 1
            return "slip"
        self.apply(o, c)
        if rng.random() < p_off * pc:
            self.fails += 1
            for c2 in (c - 1, c + 1):
                if 0 <= c2 < self.n_slot and self.slot[c2] is None:
                    self.dirty[c2] = True
                    break
            return "off"
        return "ok"


def make_world(n_obj=100, seed=0, cluster=3, n_slot=6):
    rng = random.Random(seed)
    objs, blocked = [], {}
    n_cl = max(1, int(round(n_obj / float(cluster))))
    row_w = 52.0
    for cl in range(n_cl):
        cx0 = row_w * (cl + 0.5)
        for level in range(cluster):
            k = cl * cluster + level
            if k >= n_obj:
                break
            w = round(rng.uniform(28.0, 40.0), 1)
            h = round(rng.uniform(20.0, 30.0), 1)
            jx = rng.uniform(-0.16, 0.16) * row_w
            cy = 200.0 - level * 21.0 - rng.uniform(0.0, 5.0)
            score = round(0.55 + 0.45 * rng.random(), 2)
            objs.append(Obj(k, round(cx0 + jx, 1), round(cy, 1), w, h, level, cl, score))
    for a in objs:
        blocked[a.i] = set()
    for a in objs:
        for b in objs:
            if a.cluster != b.cluster or b.level <= a.level:
                continue
            lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
            if hi - lo > 0.2 * min(a.w, b.w):
                blocked[a.i].add(b.i)
    return World(objs, blocked, n_slot=n_slot)
