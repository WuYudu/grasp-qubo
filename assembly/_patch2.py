import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\backend.py"
src = io.open(p, encoding="utf-8").read()
old = '''    def _gain(self, v, stale):
        if not stale:
            return v
        p_ok = 1.0 - self.p_wrong
        return p_ok * v - (1.0 - p_ok) * (self.spoil + v)'''
new = old + '''

    def plan(self, w, T=6, h=2):
        used_p, used_s = set(), set()
        tool = w.tool
        seq = []
        while len(seq) < T:
            best, best_v = None, -1e9
            for pi, si, v, t, stale, ok in _pairs(w):
                if not ok or pi in used_p or si in used_s:
                    continue
                g = self._gain(v, stale) - self.lam * (0.0 if t == tool else 1.0)
                if g > best_v:
                    best, best_v = (pi, si, t), g
            if best is None:
                break
            seq.append((best[0], best[1]))
            used_p.add(best[0])
            used_s.add(best[1])
            tool = best[2]
        return seq'''
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8").write(src)
print("ok")