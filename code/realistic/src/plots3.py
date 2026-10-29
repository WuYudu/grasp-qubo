"""Figures for the realistic closed-loop benchmark (Pillow only)."""

import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
FONT = "C:/Windows/Fonts/msyh.ttc"

INK = (32, 32, 32)
GREY = (120, 120, 120)
GRID = (226, 226, 226)
BG = (255, 255, 255)
COLORS = {"greedy": (44, 120, 180), "milp": (214, 39, 40), "sa": (44, 160, 44)}
NAMES = {"greedy": "贪心", "milp": "MILP 精确", "sa": "退火 SA"}


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc" if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


class Axes:
    def __init__(self, d, box, xr, yr, logy=False, ylabel="", xlabel=""):
        self.d, self.box, self.xr, self.yr, self.logy = d, box, xr, yr, logy
        x0, y0, x1, y1 = box
        self.d.rectangle(box, outline=GREY, width=1)
        for k in range(6):
            v = yr[0] + (yr[1] - yr[0]) * k / 5.0 if not logy else \
                10 ** (math.log10(yr[0]) + (math.log10(yr[1]) - math.log10(yr[0])) * k / 5.0)
            y = self.py(v)
            self.d.line([(x0, y), (x1, y)], fill=GRID, width=1)
            label = "{:g}".format(round(v, 3))
            self.d.text((x0 - 10 - self.d.textlength(label, font=font(15)), y - 9), label,
                        font=font(15), fill=GREY)
        if ylabel:
            self.d.text((x0 - 96, y0 - 34), ylabel, font=font(16), fill=INK)
        if xlabel:
            self.d.text((x0, y1 + 34), xlabel, font=font(16), fill=INK)

    def px(self, x):
        x0, _, x1, _ = self.box
        return x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (x1 - x0)

    def py(self, y):
        _, y0, _, y1 = self.box
        if self.logy:
            lo, hi = math.log10(self.yr[0]), math.log10(self.yr[1])
            return y1 - (math.log10(max(y, self.yr[0] * 1e-6)) - lo) / (hi - lo) * (y1 - y0)
        return y1 - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (y1 - y0)

    def bar(self, x, value, width, color, base=None):
        cx = self.px(x)
        w = width / (self.xr[1] - self.xr[0]) * (self.box[2] - self.box[0])
        y = self.py(value)
        yb = self.py(base if base is not None else self.yr[0])
        self.d.rectangle([cx - w / 2, y, cx + w / 2, yb], fill=color)


def legend(d, items, x, y, gap=250):
    for i, (col, name) in enumerate(items):
        cx = x + (i % 3) * gap
        cy = y + (i // 3) * 32
        d.rectangle([cx, cy + 6, cx + 26, cy + 20], fill=col)
        d.text((cx + 34, cy), name, font=font(17), fill=INK)


def fig_success(rows):
    sizes = sorted({int(r["n"]) for r in rows})
    planners = ["greedy", "milp", "sa"]
    img = Image.new("RGB", (1500, 700), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 26), "闭环：清单完成率与动作开销（感知有噪声、抓取会失败、可推、可清障）",
           font=font(28, True), fill=INK)
    ax1 = Axes(d, (110, 130, 720, 520), (0.5, len(sizes) + 0.5), (0, 105), ylabel="完成率 (%), 抓到/清单")
    ax2 = Axes(d, (880, 130, 1440, 520), (0.5, len(sizes) + 0.5), (0, 6), ylabel="每抓一个目标要几个动作")
    for _, ax in (("s", ax1), ("a", ax2)):
        for i, n in enumerate(sizes):
            x = i + 1
            ax.d.text((ax.px(x) - 12, ax.box[3] + 8), "N={}".format(n), font=font(16), fill=INK)

    for i, n in enumerate(sizes):
        for k, p in enumerate(planners):
            sel = [r for r in rows if int(r["n"]) == n and r["planner"] == p]
            if not sel:
                continue
            succ = 100.0 * mean(float(r["got"]) / max(1.0, float(r["targets"])) for r in sel)
            apt = mean(float(r["actions"]) / max(1.0, float(r["targets"])) for r in sel)
            x = i + 1 + (k - 1) * 0.24
            ax1.bar(x, succ, 0.2, COLORS[p])
            ax2.bar(x, apt, 0.2, COLORS[p])
            if i == 0:
                ax1.d.text((ax1.px(x) - 6, ax1.py(succ) - 22), "{:.0f}".format(succ),
                           font=font(14), fill=COLORS[p])
    legend(d, [(COLORS[p], NAMES[p]) for p in planners], 110, 585)
    d.text((40, 640), "每个规模 3 个种子、每个种子 1 次 episode；完成率 = 抓到的目标数 / 清单数；每个 episode 最多 90 个动作、8 次推。",
           font=font(16), fill=GREY)
    img.save(os.path.join(FIG, "fig1_success_actions.png"))
    print("fig1_success_actions.png")


def fig_latency(rows):
    sizes = sorted({int(r["n"]) for r in rows})
    planners = ["greedy", "milp", "sa"]
    img = Image.new("RGB", (1500, 700), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 26), "决策延迟：一次重规划的墙钟（对数轴）", font=font(28, True), fill=INK)
    ax = Axes(d, (120, 130, 1440, 520), (0.5, len(sizes) * 2 + 0.5), (0.01, 1000),
              logy=True, ylabel="毫秒")
    for i, n in enumerate(sizes):
        for j, key in enumerate(("p50", "p99")):
            x = i * 2 + 1 + j
            ax.d.text((ax.px(x) - 10, ax.box[3] + 8), "N={} {}".format(n, key),
                      font=font(13), fill=GREY)
    for i, n in enumerate(sizes):
        for j, key in enumerate(("p50", "p99")):
            for k, p in enumerate(planners):
                sel = [r for r in rows if int(r["n"]) == n and r["planner"] == p]
                if not sel:
                    continue
                v = mean(float(r["lat_" + key]) for r in sel)
                x = i * 2 + 1 + j + (k - 1) * 0.26
                ax.bar(x, max(v, 0.01), 0.22, COLORS[p])
    legend(d, [(COLORS[p], NAMES[p]) for p in planners], 120, 585)
    d.text((40, 640), "p50 是常见情况，p99 决定会不会卡住节拍；MILP 的尾部延迟是它最贵的地方。",
           font=font(16), fill=GREY)
    img.save(os.path.join(FIG, "fig2_latency.png"))
    print("fig2_latency.png")


def fig_scene(n_obj=40, seed=0, n_targets=9):
    import world2 as W
    import harness as H

    sc = W.make_scene(n_obj=n_obj, seed=seed, n_targets=n_targets)
    cv = sc.canvas
    scale = min(1240.0 / cv[0], 520.0 / cv[1])
    img = Image.new("RGB", (1400, 700), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 20), "场景：不规则物体、指杆扫掠通道、清单目标（每个框内是最近似位姿）", font=font(24, True), fill=INK)
    ox, oy = 60, 90

    def P(pt):
        return (ox + pt[0] * scale, oy + (cv[1] - pt[1]) * scale)

    for o, b in enumerate(sc.bodies):
        tgt = o in sc.targets
        fill = (222, 236, 222) if tgt else (234, 234, 234)
        edge = (40, 120, 40) if tgt else (170, 170, 170)
        d.polygon([P(p) for p in b.poly], fill=fill, outline=edge)
    for o in sc.targets:
        g = sc.best(o)
        if not g.blocked and g.score > 0:
            d.polygon([P(p) for p in g.corridor()], outline=(40, 120, 40))
        else:
            d.polygon([P(p) for p in g.corridor()], outline=(200, 120, 120))
    for o in range(sc.n):
        b = sc.bodies[o]
        c = P((b.cx, b.cy))
        if o in sc.targets:
            d.text((c[0] - 6, c[1] - 8), "T", font=font(16, True), fill=(20, 90, 20))
    d.text((40, 640),
           "绿描边 = 目标且当前可达；红描边 = 目标但通道被挡，得先清障或推一下。",
           font=font(16), fill=GREY)
    img.save(os.path.join(FIG, "fig3_scene.png"))
    print("fig3_scene.png")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(RES, "bench_{}.csv".format(args.tag))
    rows = load_rows(path)
    if args.only in ("", "success"):
        fig_success(rows)
    if args.only in ("", "latency"):
        fig_latency(rows)
    if args.only in ("", "scene"):
        fig_scene()


if __name__ == "__main__":
    main()
