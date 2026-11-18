import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\backend.py"
src = io.open(p, encoding="utf-8").read()

add = '''

class GreedyTool(Backend):
    """按 价值 − 换工具代价 逐步选，顺序在这里才起作用。"""
    name = "greedy_tool"

    def __init__(self, lam=0.20, w_stale=0.30):
        self.lam = lam
        self.w_stale = w_stale

    def plan(self, w, T=6, h=2):
        used_p, used_s = set(), set()
        tool = w.tool
        seq = []
        while len(seq) < T:
            best, best_v = None, -1e9
            for pi, si, v, t, stale, ok in _pairs(w):
                if not ok or pi in used_p or si in used_s:
                    continue
                g = v - self.lam * (0.0 if t == tool else 1.0) - self.w_stale * (1.0 if stale else 0.0)
                if g > best_v:
                    best, best_v = (pi, si, t), g
            if best is None:
                break
            seq.append((best[0], best[1]))
            used_p.add(best[0])
            used_s.add(best[1])
            tool = best[2]
        return seq


class RiskAware(GreedyTool):
    """对过期槽位算期望净收益，负期望就不下注。"""
    name = "risk"

    def __init__(self, p_wrong=0.45, lam=0.20, spoil=0.60):
        GreedyTool.__init__(self, lam=lam, w_stale=0.0)
        self.p_wrong = p_wrong
        self.spoil = spoil

    def _gain(self, v, stale):
        if not stale:
            return v
        p_ok = 1.0 - self.p_wrong
        return p_ok * v - (1.0 - p_ok) * (self.spoil + v)
'''

src = src.replace("\n\nBACKENDS = {", add + "\n\nBACKENDS = {")
src = src.replace(
    'BACKENDS = {"greedy": Greedy, "greedy_safe": GreedySafe, "milp": Milp, "anneal": Anneal}',
    'BACKENDS = {"greedy": Greedy, "greedy_safe": GreedySafe, "greedy_tool": GreedyTool,\n            "risk": RiskAware, "milp": Milp, "anneal": Anneal}')
io.open(p, "w", encoding="utf-8").write(src)

q = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\calibrate.py"
c = io.open(q, encoding="utf-8").read()
c = c.replace('for n in ("greedy", "greedy_safe", "milp", "anneal"):',
              'for n in ("greedy", "greedy_safe", "greedy_tool", "risk", "milp", "anneal"):')
io.open(q, "w", encoding="utf-8").write(c)
print("patched")