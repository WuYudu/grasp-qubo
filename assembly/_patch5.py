import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\seq.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('MU = 0.15      # 换工具罚（每换一次付出的等效分数）',
              'MU = 0.15      # 换工具罚，默认值；实际用 _mu(w) 按时间成本折算')
s = s.replace('def build(w, S=4, w_stale=0.5):',
              'def _mu(w):\n    """换一次工具的时间，折成多少个动作的收益。"""\n    ratio = world.T_TOOL / world.T_ACTION\n    return 0.10 * ratio\n\n\ndef build(w, S=4, w_stale=0.5):')
s = s.replace('                    add(a * S + s, b * S + s + 1, MU)',
              '                    add(a * S + s, b * S + s + 1, _mu(w))')
s = s.replace('            v -= MU * sw', '            v -= _mu(w) * sw')
io.open(p, "w", encoding="utf-8").write(s)
print("patched")