"""Two figures: static gap versus scale, and rolling versus open loop."""
import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
FONT = "C:\\Windows\\Fonts\\msyh.ttc"
BG = (255, 255, 255)
INK = (32, 32, 32)
GREY = (120, 120, 120)
BLUE = (31, 119, 180)
GREEN = (44, 160, 44)
RED = (214, 39, 40)
ORANGE = (255, 127, 14)


def f(sz):
    return ImageFont.truetype(FONT, sz)


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Ax:
    def __init__(self, d, box, xr, yr, logy=False):
        self.d, self.box, self.xr, self.yr, self.logy = d, box, xr, yr, logy

    def px(self, x):
        x0, x1 = self.box[0], self.box[2]
        return x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (x1 - x0)

    def py(self, y):
        lo, hi = self.yr
        if self.logy:
            v = (math.log10(max(y, 1e-9)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            v = (y - lo) / (hi - lo)
        return self.box[3] - v * (self.box[3] - self.box[1])

    def frame(self, xticks, yticks, xlabel, ylabel):
        d = self.d
        d.rectangle(self.box, outline=(190, 190, 190))
        for t in xticks:
            x = self.px(t)
            d.line([(x, self.box[3]), (x, self.box[3] + 5)], fill=(150, 150, 150))
            d.text((x - 12, self.box[3] + 9), f"{t:g}", font=f(15), fill=GREY)
        for t in yticks:
            y = self.py(t)
            d.line([(self.box[0] - 5, y), (self.box[0], y)], fill=(150, 150, 150))
            d.line([(self.box[0], y), (self.box[2], y)], fill=(238, 238, 238))
            lbl = f"{t:g}" if not self.logy else (f"{t:g}" if t >= 1 else f"{t}")
            d.text((self.box[0] - 8 - 7 * len(lbl), y - 9), lbl, font=f(15), fill=GREY)
        d.text((self.box[0], self.box[3] + 34), xlabel, font=f(17), fill=INK)
        d.text((self.box[0] - 96, self.box[1] - 34), ylabel, font=f(16), fill=INK)

    def path(self, pts, col, width=3):
        if len(pts) > 1:
            self.d.line([(self.px(x), self.py(y)) for x, y in pts], fill=col, width=width, joint="curve")

    def dot(self, x, y, col, r=5):
        cx, cy = self.px(x), self.py(y)
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)


def legend(d, items, x, y, cols=2, w=330):
    for i, (col, label) in enumerate(items):
        cx = x + (i % cols) * w
        cy = y + (i // cols) * 30
        d.line([(cx, cy + 9), (cx + 30, cy + 9)], fill=col, width=4)
        d.text((cx + 40, cy), label, font=f(17), fill=INK)


def fig_scale():
    rs = [r for r in rows(os.path.join(RES, "bench_frac50.csv"))]
    img = Image.new("RGB", (1500, 660), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 26), "静态规模扫描：窗口 = N/2，贪心相对精确最优差多少", font=f(28), fill=INK)
    ax = Ax(d, (110, 120, 700, 500), (5, 65), (0, 3.6))
    ax.frame([9, 12, 20, 32, 40, 60], [0, 1, 2, 3], "物体数 N", "差距 (%)")
    for kind, col in (("piles", BLUE), ("clutter", ORANGE)):
        pts = sorted((int(r["n"]), float(r["gap"])) for r in rs if r["kind"] == kind)
        ax.path(pts, col)
        for x, y in pts:
            ax.dot(x, y, col, 4)
            if y > 1.0:
                d.text((ax.px(x) - 10, ax.py(y) - 24), f"{y:.1f}", font=f(14), fill=col)
    legend(d, [(BLUE, "堆叠料箱"), (ORANGE, "随机散乱")], 110, 578)

    ax2 = Ax(d, (880, 120, 1440, 500), (5, 65), (0.005, 5000), logy=True)
    ax2.frame([9, 12, 20, 32, 40, 60], [0.01, 1, 100, 5000], "物体数 N", "求解耗时 (ms)")
    for key, col, name in (("t_greedy_ms", GREEN, "贪心"), ("t_milp_ms", RED, "MILP 精确解")):
        pts = sorted((int(r["n"]), float(r[key])) for r in rs if r["kind"] == "piles")
        ax2.path(pts, col)
        for x, y in pts:
            ax2.dot(x, y, col, 4)
    legend(d, [(GREEN, "贪心"), (RED, "MILP 精确解")], 880, 578)
    d.text((40, 620), "N=60：精确解 1.9 s 仍最优，贪心差 3.08%，墙钟差四个数量级。",
           font=f(16), fill=GREY)
    img.save(os.path.join(FIG, "fig1_gap_vs_scale.png"))
    print("fig1_gap_vs_scale.png")


def fig_rolling():
    rs = rows(os.path.join(RES, "bench_main.csv"))
    scenes = sorted({(r["scene"], int(r["n"])) for r in rs}, key=lambda t: t[1])
    methods = ["oneshot-milp", "roll-greedy", "roll-milp", "roll-sa"]
    names = {"oneshot-milp": "只规划一次", "roll-greedy": "滚动+贪心",
             "roll-milp": "滚动+MILP", "roll-sa": "滚动+SA"}
    cols = {"oneshot-milp": GREY, "roll-greedy": GREEN, "roll-milp": BLUE, "roll-sa": RED}
    img = Image.new("RGB", (1500, 700), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 24), "闭环：只规划一次 vs 每三步重规划（抓取失败率 15%，物体在抖动）",
           font=f(26), fill=INK)
    ax = Ax(d, (110, 130, 760, 520), (0, len(scenes)), (0, 1.05))
    ax.frame([], [0, 0.25, 0.5, 0.75, 1.0], "", "清理比例 (抓走 / 总数)")
    for i, (scene, n) in enumerate(scenes):
        for k, m in enumerate(methods):
            sel = [r for r in rs if r["scene"] == scene and r["method"] == m]
            if not sel:
                continue
            val = sum(float(r["grasps"]) for r in sel) / len(sel) / n
            bw = 0.18
            x0 = i + 0.09 + k * bw
            y0 = ax.py(val)
            d.rectangle([ax.px(x0), y0, ax.px(x0 + bw * 0.92), ax.py(0)],
                        fill=cols[m], outline=None)
            d.text((ax.px(x0 + bw * 0.46) - 10, y0 - 22), f"{val:.2f}".rstrip("0").rstrip("."),
                   font=f(14), fill=cols[m])
        d.text((ax.px(i + 0.5) - 26, ax.box[3] + 10), f"N={n}", font=f(16), fill=INK)
    for i, (scene, n) in enumerate(scenes):
        d.text((ax.px(i + 0.5) - 10, ax.box[3] + 32), scene, font=f(12), fill=GREY)
    legend(d, [(cols[m], names[m]) for m in methods], 110, 600)

    ax2 = Ax(d, (920, 130, 1440, 520), (0, len(scenes)), (0.5, 20000), logy=True)
    ax2.frame([], [1, 10, 100, 1000, 10000],
              "", "所有轮次的求解耗时合计 (ms)")
    for i, (scene, n) in enumerate(scenes):
        for k, m in enumerate(methods):
            sel = [r for r in rs if r["scene"] == scene and r["method"] == m]
            if not sel:
                continue
            val = max(0.55, sum(float(r["solver_ms"]) for r in sel) / len(sel))
            bw = 0.18
            x0 = i + 0.09 + k * bw
            d.rectangle([ax2.px(x0), ax2.py(val), ax2.px(x0 + bw * 0.92), ax2.py(0.5)], fill=cols[m])
    d.text((40, 656),
           "重规划把清理比例从 0.4 拉到 1.0；同一段闭环里贪心与精确解的清理结果完全相同，耗时差三个数量级。",
           font=f(16), fill=GREY)
    img.save(os.path.join(FIG, "fig2_rolling_vs_oneshot.png"))
    print("fig2_rolling_vs_oneshot.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_scale()
    fig_rolling()
