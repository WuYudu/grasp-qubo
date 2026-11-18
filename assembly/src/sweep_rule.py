"""两种报废判据的对照：种类约定 vs 几何尺寸。"""
import csv
import os

import backend
import sim
import world as W

KW = {"A": 0.080, "B": 0.120, "C": 0.160}


class GeoWorld(W.World):
    """只按几何判：零件宽度不超过槽口就放得下。"""

    def apply(self, pi, si):
        p, s = self.parts[pi], self.slots[si]
        if p.state != "bin" or s.filled or s.spoiled:
            return "invalid"
        self.actions += 1
        self.clock += W.T_ACTION
        if p.tool != self.tool:
            self.clock += W.T_TOOL
            self.tool = p.tool
        if KW[p.kind] <= KW[s.kind]:
            s.filled = True
            p.state = "placed"
            self.placed.append(pi)
            self.done[p.kind] = self.done.get(p.kind, 0) + 1
            return "ok"
        s.spoiled = True
        p.state = "wasted"
        self.spoiled += 1
        return "spoiled"


PAIRS_KIND = backend._pairs


def pairs_geo(w, use_belief=True):
    out = []
    for p in w.bin_parts():
        for s in w.free_slots():
            ref = s.belief if use_belief else s.kind
            out.append((p.i, s.i, p.value, p.tool, s.stale, KW[p.kind] <= KW[ref]))
    return out


def run(rule, h, name, cls, seed):
    if rule == "geom":
        backend._pairs = pairs_geo
        w = GeoWorld(n_part=12, n_slot=8, seed=seed, p_stale=0.55, q_correct=0.55, obs_cost=0.25)
    else:
        backend._pairs = PAIRS_KIND
        w = W.World(n_part=12, n_slot=8, seed=seed, p_stale=0.55, q_correct=0.55, obs_cost=0.25)
    r = sim.episode(w, cls(), h=h, seed=seed)
    return r["score"], r["spoiled"] / max(1, r["spoiled"] + r["placed"])


def main():
    names = [("greedy_safe", backend.GreedySafe), ("risk", backend.RiskAware),
             ("milp", backend.Milp), ("anneal", backend.Anneal)]
    hs = [1, 2, 3, 4]
    out = []
    for rule in ("kind", "geom"):
        for nm, cls in names:
            for h in hs:
                ss, sp = [], []
                for seed in range(3):
                    s, r = run(rule, h, nm, cls, seed)
                    ss.append(s)
                    sp.append(r)
                m = sum(ss) / len(ss)
                out.append(dict(rule=rule, backend=nm, h=h, score=round(m, 3),
                                spoiled=round(sum(sp) / len(sp), 3)))
                print("%-4s %-12s h=%d  score %6.3f  spoiled %4.2f" % (rule, nm, h, m, sum(sp) / len(sp)))
    p = os.path.join("..", "results", "calib_rule.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print("saved", p)


if __name__ == "__main__":
    main()