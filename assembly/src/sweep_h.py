"""细扫前缀长度，画 risk 与 safe 的交点。"""
import csv
import os
import statistics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont

import backend
import sim
import world

HS = [1, 2, 3, 4, 5, 6]
NAMES = ["greedy_safe", "risk", "milp", "anneal"]
COL = {"greedy_safe": (31, 119, 180), "risk": (214, 39, 40),
       "milp": (44, 160, 44), "anneal": (150, 90, 200)}
LAB = {"greedy_safe": "保守贪心", "risk": "风险感知", "milp": "精确整数规划", "anneal": "退火（模拟）"}
FONT = r"C:\Windows\Fonts\msyh.ttc"


def run(seeds=5):
    rows = []
    for h in HS:
        for n in NAMES:
            sc, sp, ck, rp = [], [], [], []
            for s in range(seeds):
                w = world.World(n_part=12, n_slot=8, seed=s, p_stale=0.55, q_correct=0.55)
                r = sim.episode(w, backend.make(n), h=h, budget_s=18.0, seed=100 + s)
                sc.append(r["score"])
                sp.append(r["spoiled"])
                ck.append(r["clock"])
                rp.append(r["replans"])
            rows.append(dict(h=h, name=n, score=round(statistics.mean(sc), 3),
                             spoiled=round(statistics.mean(sp), 2),
                             clock=round(statistics.mean(ck), 1),
                             replans=round(statistics.mean(rp), 1)))
    return rows


def plot(rows, path):
    W, H = 1200, 720
    x0, y0, x1, y1 = 110, 104, 1140, 556
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = lambda n: ImageFont.truetype(FONT, n)
    sc = [r["score"] for r in rows]
    lo = min(min(sc), 0.0) - 0.15
    hi = max(sc) + 0.15
    px = lambda h: x0 + (h - HS[0]) / (HS[-1] - HS[0]) * (x1 - x0)
    py = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    d.text((30, 30), "分数随前缀长度：承诺一变长，优化就从有用变成有害", font=f(24), fill=(20, 20, 20))
    d.text((30, 74), "12 件 / 8 槽，过期率 0.55（过期信念仍有 55% 是对的），18 s 预算，5 种子", font=f(15), fill=(110, 110, 110))
    v = lo
    while v <= hi:
        if abs(v) > 1e-9:
            d.line([(x0, py(v)), (x1, py(v))], fill=(236, 236, 236))
            d.text((x0 - 56, py(v) - 9), "%.1f" % v, font=f(15), fill=(130, 130, 130))
        v = round(v + 0.2, 2)
    d.line([(x0, py(0.0)), (x1, py(0.0))], fill=(150, 150, 150), width=2)
    d.rectangle([x0, y0, x1, y1], outline=(205, 205, 205))
    for h in HS:
        d.text((px(h) - 5, y1 + 12), str(h), font=f(16), fill=(90, 90, 90))
    d.text((x0 + 330, y1 + 36), "前缀长度 h（一次规划、执行几步）", font=f(17), fill=(90, 90, 90))
    for n in NAMES:
        pts = [(px(r["h"]), py(r["score"])) for r in rows if r["name"] == n]
        d.line(pts, fill=COL[n], width=3)
        for (x, y) in pts:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=COL[n], outline=(255, 255, 255))
    for i, n in enumerate(NAMES):
        x = 150 + (i % 2) * 460
        y = 620 + (i // 2) * 34
        d.line([(x, y + 8), (x + 30, y + 8)], fill=COL[n], width=4)
        d.text((x + 40, y), LAB[n], font=f(17), fill=(40, 40, 40))
    img.save(path)


def load():
    out = []
    with open("../results/calib_h.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(dict(h=int(r["h"]), name=r["name"], score=float(r["score"]),
                            spoiled=float(r["spoiled"]), clock=float(r["clock"]), replans=float(r["replans"])))
    return out


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        plot(load(), "../figures/fig1_crossover.png")
        print("figure ok")
        raise SystemExit
    rows = run()
    for r in rows:
        print("h=%d %-12s score=%6.3f spoiled=%4.2f clock=%5.1f replans=%5.1f"
              % (r["h"], r["name"], r["score"], r["spoiled"], r["clock"], r["replans"]))
    with open("../results/calib_h.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["h", "name", "score", "spoiled", "clock", "replans"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    plot(rows, "../figures/fig1_crossover.png")
    print("figure ok")