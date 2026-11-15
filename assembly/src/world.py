"""长程台面装配：不可逆放置、组内配套、观测不确定。"""
import random

KINDS = ("A", "B", "C")
TOOLS = ("t1", "t2")

T_ACTION = 0.70
T_TOOL = 2.00
OBS = 0.25

W_PART = 0.10
W_GROUP = 0.90
W_SPOIL = 0.60
W_STREAK = 0.06   # 连续用同一工具，每个动作的额外收益（预热）


class Part:
    __slots__ = ("i", "kind", "tool", "value", "state")

    def __init__(self, i, kind, tool, value):
        self.i = i
        self.kind = kind
        self.tool = tool
        self.value = value
        self.state = "bin"


class Slot:
    __slots__ = ("i", "kind", "stale", "belief", "filled", "spoiled")

    def __init__(self, i, kind, stale, belief):
        self.i = i
        self.kind = kind
        self.stale = stale
        self.belief = belief
        self.filled = False
        self.spoiled = False


class World:
    def __init__(self, n_part=12, n_slot=8, seed=0, p_stale=0.35, q_correct=0.55, obs_cost=OBS):
        rng = random.Random(seed)
        self.parts = [Part(i, KINDS[i % 3], TOOLS[i % 2], round(rng.uniform(0.40, 1.00), 3))
                      for i in range(n_part)]
        self.slots = []
        for i in range(n_slot):
            kind = KINDS[i % 3]
            stale = rng.random() < p_stale
            if not stale or rng.random() < q_correct:
                belief = kind
            else:
                belief = rng.choice([k for k in KINDS if k != kind])
            self.slots.append(Slot(i, kind, stale, belief))
        self.obs_cost = obs_cost
        self.last_tool = None
        self.bonus = 0.0
        self.tool = None
        self.clock = 0.0
        self.actions = 0
        self.spoiled = 0
        self.placed = []
        self.groups = {}
        for p in self.parts:
            self.groups[p.kind] = self.groups.get(p.kind, 0) + 1
        self.done = {}

    def fresh_ratio(self):
        free = [s for s in self.slots if not s.filled and not s.spoiled]
        if not free:
            return 1.0
        return sum(1 for s in free if not s.stale) / len(free)

    def observe(self, rng, p_fix=0.55):
        for s in self.slots:
            if s.stale and rng.random() < p_fix:
                s.stale = False
                s.belief = s.kind
        self.clock += self.obs_cost

    def bin_parts(self):
        return [p for p in self.parts if p.state == "bin"]

    def free_slots(self):
        return [s for s in self.slots if not s.filled and not s.spoiled]

    def apply(self, pi, si):
        p = self.parts[pi]
        s = self.slots[si]
        if p.state != "bin" or s.filled or s.spoiled:
            return "invalid"
        self.actions += 1
        self.clock += T_ACTION
        if p.tool != self.tool:
            self.clock += T_TOOL
            self.tool = p.tool
        if p.kind == s.kind:
            s.filled = True
            p.state = "placed"
            self.placed.append(pi)
            self.done[p.kind] = self.done.get(p.kind, 0) + 1
            return "ok"
        s.spoiled = True
        p.state = "wasted"
        self.spoiled += 1
        return "spoiled"

    def score(self):
        v = W_PART * len(self.placed)
        v -= W_SPOIL * self.spoiled
        for k, n in self.done.items():
            if n == self.groups[k]:
                v += W_GROUP
        return v

    def summary(self):
        return dict(score=round(self.score(), 3), placed=len(self.placed),
                    spoiled=self.spoiled, actions=self.actions,
                    clock=round(self.clock, 1), done=dict(self.done))