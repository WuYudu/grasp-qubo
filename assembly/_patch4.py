import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\sweep_h.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('    W, H = 1200, 680\n    x0, y0, x1, y1 = 110, 96, 1140, 560',
              '    W, H = 1200, 720\n    x0, y0, x1, y1 = 110, 104, 1140, 556')
s = s.replace('d.text((30, 66), "12 件 / 8 槽', 'd.text((30, 74), "12 件 / 8 槽')
s = s.replace('    d.text((36, y0 - 40), "分数", font=f(16), fill=(90, 90, 90))\n', '')
s = s.replace('d.text((x0 + 330, y1 + 44), "前缀长度 h（一次规划、执行几步）", font=f(17), fill=(90, 90, 90))',
              'd.text((x0 + 330, y1 + 36), "前缀长度 h（一次规划、执行几步）", font=f(17), fill=(90, 90, 90))')
s = s.replace('        y = 600 + (i // 2) * 30', '        y = 620 + (i // 2) * 34')
s = s.replace('if __name__ == "__main__":\n    rows = run()',
              'def load():\n    out = []\n    with open("../results/calib_h.csv", encoding="utf-8") as fh:\n        for r in csv.DictReader(fh):\n            out.append(dict(h=int(r["h"]), name=r["name"], score=float(r["score"]),\n                            spoiled=float(r["spoiled"]), clock=float(r["clock"]), replans=float(r["replans"])))\n    return out\n\n\nif __name__ == "__main__":\n    import sys\n    if "--plot-only" in sys.argv:\n        plot(load(), "../figures/fig1_crossover.png")\n        print("figure ok")\n        raise SystemExit\n    rows = run()')
s = s.replace('    d.text((30, 34), "分数随前缀长度', '    d.text((30, 30), "分数随前缀长度')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")