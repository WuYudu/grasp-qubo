import io
# 1) world: 连续用同一工具给收益（预热），换回来要重新预热
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\world.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('W_SPOIL = 0.60', 'W_SPOIL = 0.60\nW_STREAK = 0.06   # 连续用同一工具，每个动作的额外收益（预热）')
s = s.replace('        self.obs_cost = obs_cost', '        self.obs_cost = obs_cost\n        self.last_tool = None\n        self.bonus = 0.0')
s = s.replace('            if st == "ok":', '            if st == "ok":\n                if self.last_tool is not None and p.tool == self.last_tool:\n                    self.bonus += W_STREAK\n                self.last_tool = p.tool')
s = s.replace('        return 0.10 * fills - 0.60 * spoil + 0.90 * groups',
              '        return 0.10 * fills - 0.60 * spoil + 0.90 * groups + self.bonus')
io.open(p, "w", encoding="utf-8").write(s)

# 2) seq: 同工具的相邻两步给负二次项（奖励），换工具给正二次项
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\seq.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('                if info[a][2] != info[b][2]:\n                    add(a * S + s, b * S + s + 1, _mu(w))',
              '                if info[a][2] != info[b][2]:\n                    add(a * S + s, b * S + s + 1, _mu(w))\n                else:\n                    add(a * S + s, b * S + s + 1, -world.W_STREAK)')
s = s.replace('            v -= _mu(w) * sw', '            v -= _mu(w) * sw\n            v += world.W_STREAK * (len(tools) - 1 - sw)')
io.open(p, "w", encoding="utf-8").write(s)
print("patched")