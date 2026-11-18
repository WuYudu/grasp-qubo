import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\qpu.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('''def solve(q''', '''def split_by_tool(w, S=4, w_stale=0.5):
    """按工具拆子问题：每个工具有自己的动作集，互不耦合，适合分别上机。"""
    import backend as bk
    out = {}
    for tool in world.TOOLS:
        pairs = [(pi, si) for pi, si, v, t, stale, ok in bk._pairs(w) if ok and t == tool]
        if not pairs:
            continue
        acts = []
        for pi, si in pairs:
            p, s = w.parts[pi], w.slots[si]
            acts.append((pi, si, p.tool, p.value - w_stale * (1.0 if s.stale else 0.0)))
        A = len(acts)
        q = {}
        def add(i, j, v):
            if i > j:
                i, j = j, i
            q[(i, j)] = q.get((i, j), 0.0) + v
        for a in range(A):
            for st in range(S):
                add(a * S + st, a * S + st, -acts[a][3])
        for a in range(A):
            for st in range(S):
                for tt in range(st + 1, S):
                    add(a * S + st, a * S + tt, 3.0)
        for st in range(S):
            for a in range(A):
                for b in range(a + 1, A):
                    share = acts[a][0] == acts[b][0] or acts[a][1] == acts[b][1]
                    add(a * S + st, b * S + st, 3.0 if share else 6.0)
        for st in range(S - 1):
            for a in range(A):
                for b in range(A):
                    add(a * S + st, b * S + st + 1, -world.W_STREAK)
        out[tool] = q
    return out


def solve(q''')
s = s.replace('''    if a.dry_run:
        return''', '''    if a.split:
        for tool, q in split_by_tool(w).items():
            d = describe(q)
            d["physical_estimate"] = embedding_estimate(d)
            print("按工具 %s 拆开：变量 %d，耦合 %d，最大度数 %d，物理比特约 %d"
                  % (tool, d["variables"], d["interactions"], d["max_degree"], d["physical_estimate"]))
        return
    if a.dry_run:
        return''')
s = s.replace('    ap.add_argument("--dry-run", action="store_true")',
              '    ap.add_argument("--dry-run", action="store_true")\n    ap.add_argument("--split", action="store_true", help="按工具拆子问题，看能不能上机")')
io.open(p, "w", encoding="utf-8").write(s)
print("patched")