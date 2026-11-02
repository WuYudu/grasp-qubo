"""Three figures: gap vs size, latency vs size, and one worked case."""

import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

import place
import quboplace
import solvers

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
RES = os.path.join(os.path.dirname(HERE), "results")
FONT = r"C:\Windows\Fonts\msyh.ttc"
INK = (32, 32, 32)
GREY = (120, 120, 120)
BLUE = (31, 119, 180)
ORANGE = (255, 127, 14)
GREEN = (44, 160, 44)
RED = (214, 39, 40)
PALE = (238, 242, 246)


def font(size):
    return ImageFont.truetype(FONT, size)


class Axes:
    def __init__(self, d, box, xr, yr, logy=False, ylabel="", xlabel=""):
        self.d, self.box, self.xr, self.yr = d, box, xr, yr
        self.logy, self.ylabel, self.xlabel = logy, ylabel, xlabel

    def px(self, x):
        x0, x1 = self.xr
        return self.box[0] + (x - x0) / (x1 - x0) * (self.box[2] - self.box[0])

    def py(self, y):
        y0, y1 = self.yr
        if self.logy:
            lo, hi = math.log10(max(y0, 1e-6)), math.log10(max(y1, 1e-6))
            v = (math.log10(max(y, 1e-6)) - lo) / (hi - lo)
        else:
            v = (y - y0) / (y1 - y0)
        return self.box[3] - v * (self.box[3] - self.box[1])

    def frame(self, xs, ys, yfmt="{:.0f}"):
        d = self.d
        d.rectangle(self.box, outline=(190, 196, 204))
        for y in ys:
            yy = self.py(y)
            d.line([(self.box[0], yy), (self.box[2], yy)], fill=(226, 231, 237))
            d.text((self.box[0] - 10, yy), yfmt.format(y), font=font(15), fill=GREY, anchor="rm")
        for x in xs:
            d.text((self.px(x), self.box[3] + 8), str(x), font=font(15), fill=GREY, anchor="ma")
        d.text((self.box[0] - 78, self.box[1] - 26), self.ylabel, font=font(16), fill=GREY)
        d.text((self.box[0], self.box[3] + 46), self.xlabel, font=font(16), fill=GREY)

    def line(self, pts, color, width=3):
        self.d.line([(self.px(x), self.py(y)) for x, y in pts], fill=color, width=width)

    def dot(self, x, y, color, r=5):
        cx, cy = self.px(x), self.py(y)
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def load(tag):
    with open(os.path.join(RES, "bench_%s.csv" % tag), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k in ("seed", "n_obj", "n_slot", "T", "n_var", "nnz", "milp_n", "greedy_n", "sa_n", "sa_bad", "reads"):
                r[k] = int(float(r[k]))
            else:
                r[k] = float(r[k])
    return rows


def agg(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["n_obj"], []).append(r[key])
    ns = sorted(out)
    return ns, [sum(out[n]) / len(out[n]) for n in ns]


def fig_gap(rows):
    ns, g_greedy = agg(rows, "gap_greedy")
    _, g_sa = agg(rows, "gap_sa")
    img = Image.new("RGB", (1180, 660), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 24), "抓-放联合排序：相对精确最优的差距", font=font(28), fill=INK)
    top = max(max(g_greedy), max(g_sa)) * 1.25 + 1.0
    ax = Axes(d, (110, 110, 1120, 520), (min(ns) - 1, max(ns) + 1), (0, top),
              ylabel="差距 (%)", xlabel="")
    ax.frame(ns, [top * k / 4 for k in range(5)], yfmt="{:.0f}")
    ax.line(list(zip(ns, g_greedy)), BLUE)
    for x, y in zip(ns, g_greedy):
        ax.dot(x, y, BLUE)
        d.text((ax.px(x), ax.py(y) - 16), "%.1f" % y, font=font(15), fill=BLUE, anchor="mb")
    ax.line(list(zip(ns, g_sa)), RED)
    for x, y in zip(ns, g_sa):
        ax.dot(x, y, RED)
        d.text((ax.px(x), ax.py(y) + 8), "%.1f" % y, font=font(15), fill=RED, anchor="mt")
    d.rectangle([110, 578, 132, 590], fill=BLUE)
    d.text((140, 576), "贪心", font=font(17), fill=INK)
    d.rectangle([310, 578, 332, 590], fill=RED)
    d.text((340, 576), "退火（模拟）", font=font(17), fill=INK)
    d.text((40, 620), "每点是两个种子的均值；差距 = (最优 − 该解) / 最优。",
           font=font(16), fill=GREY)
    img.save(os.path.join(FIG, "fig1_gap.png"))


def fig_time(rows):
    ns, t_milp = agg(rows, "milp_ms")
    _, t_greedy = agg(rows, "greedy_ms")
    _, t_sa = agg(rows, "sa_ms")
    img = Image.new("RGB", (1180, 660), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 24), "求解耗时（对数轴，同一台机器）", font=font(28), fill=INK)
    lo = max(min(t_greedy) * 0.5, 0.01)
    hi = max(t_sa) * 3
    ax = Axes(d, (120, 110, 1120, 520), (min(ns) - 1, max(ns) + 1), (lo, hi),
              logy=True, ylabel="毫秒", xlabel="")
    ticks = [10.0 ** k for k in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)]
    ax.frame(ns, ticks, yfmt="{:.2f}")
    for pts, col, name in (((ns, t_milp), BLUE, "精确 MILP"),
                           ((ns, t_greedy), GREEN, "贪心"),
                           ((ns, t_sa), RED, "退火")):
        ax.line(list(zip(*pts)), col)
        for x, y in zip(*pts):
            ax.dot(x, y, col)
    d.rectangle([120, 578, 142, 590], fill=BLUE)
    d.text((150, 576), "精确 MILP", font=font(17), fill=INK)
    d.rectangle([320, 578, 342, 590], fill=GREEN)
    d.text((350, 576), "贪心", font=font(17), fill=INK)
    d.rectangle([470, 578, 492, 590], fill=RED)
    d.text((500, 576), "退火（模拟）", font=font(17), fill=INK)
    img.save(os.path.join(FIG, "fig2_time.png"))


def _plan_panel(d, origin, title, sc, placed, box_w):
    x0, y0 = origin
    d.text((x0, y0), title, font=font(20), fill=INK)
    cell_w, cell_h = 150, 56
    d.text((x0, y0 + 34), "步", font=font(15), fill=GREY)
    d.text((x0 + 60, y0 + 34), "物体", font=font(15), fill=GREY)
    d.text((x0 + 190, y0 + 34), "槽位", font=font(15), fill=GREY)
    d.text((x0 + 290, y0 + 34), "价值", font=font(15), fill=GREY)
    for k, (o, c, s) in enumerate(placed):
        y = y0 + 62 + k * cell_h
        d.rectangle([x0, y, x0 + box_w, y + cell_h - 10], fill=PALE)
        d.text((x0 + 14, y + 10), str(s), font=font(18), fill=INK)
        d.text((x0 + 60, y + 10), "obj%d" % o, font=font(18), fill=INK)
        d.text((x0 + 190, y + 10), "槽%d" % c, font=font(18), fill=INK)
        d.text((x0 + 290, y + 10), "%.2f" % sc.value(o, c), font=font(18), fill=INK)


def fig_case():
    sc = place.make_scene(n_obj=24, seed=0, cluster=3, n_slot=6)
    T = 6
    m = solvers.exact_milp(sc, T=T, time_limit=30.0)
    g = solvers.greedy(sc, T=T)
    pack = quboplace.build(sc, T=T)
    img = Image.new("RGB", (1180, 700), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 24), "同一个场景、同一个 QUBO：贪心少抓一个，少拿 %.1f%%" %
           (100.0 * (m.score - g.score) / m.score), font=font(27), fill=INK)
    d.text((40, 62), "24 个物体、6 个槽位、6 步预算；槽位宽窄不一，窄槽只接小物体；"
                     "被压住的物体必须先被移走。", font=font(16), fill=GREY)
    _plan_panel(d, (60, 110), "精确最优  总分 %.3f  （%d 个）" % (m.score, len(m.placed)), sc, m.placed, 420)
    _plan_panel(d, (620, 110), "贪心  总分 %.3f  （%d 个）" % (g.score, len(g.placed)), sc, g.placed, 420)
    miss = [o for o, c, s in m.placed if o not in {x[0] for x in g.placed}]
    d.text((620, 110 + 62 + len(g.placed) * 56 + 12),
           "没抓到的：" + "、".join("obj%d" % o for o in miss), font=font(17), fill=RED)
    d.text((40, 640), "物体数 %d，变量 %d，非零耦合 %d；精确解 %.0f ms，贪心 %.2f ms。" %
           (sc.n_obj, pack.n, pack.nnz, m.ms, g.ms), font=font(16), fill=GREY)
    img.save(os.path.join(FIG, "fig3_case.png"))
    return m, g, sc


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load("main")
    fig_gap(rows)
    fig_time(rows)
    fig_case()
    print("figures ok")


if __name__ == "__main__":
    main()
