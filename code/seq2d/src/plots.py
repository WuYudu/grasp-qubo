"""Figures for the sequential-grasp experiments. Pillow only, no matplotlib."""
from __future__ import annotations

import argparse
import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

import qubo
import scene as S
import solvers as V

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
FIG = os.path.join(os.path.dirname(HERE), "figures")
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

COLORS = {
    "milp": (31, 119, 180),
    "greedy": (127, 127, 127),
    "sa": (44, 160, 44),
    "sqa": (214, 39, 40),
}
LABELS = {
    "milp": "MILP/HiGHS(精确)",
    "greedy": "贪心剥离",
    "sa": "经典退火 SA",
    "sqa": "量子退火模拟 SQA",
}
PALETTE = [(44, 110, 190), (34, 150, 90), (200, 60, 60), (170, 110, 20),
           (120, 60, 170), (20, 140, 150), (190, 40, 130), (90, 90, 40)]


def font(size: int):
    return ImageFont.truetype(FONT_PATH, size)


class Axes:
    def __init__(self, d, box, xr, yr, logy=False, xlabel=""):
        self.d = d
        self.x0, self.y0, self.x1, self.y1 = box
        self.xr = xr
        self.yr = yr
        self.logy = logy
        self.xlabel = xlabel

    def px(self, x):
        a, b = self.xr
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        a, b = self.yr
        if self.logy:
            f = (math.log10(max(y, 1e-12)) - math.log10(a)) / (math.log10(b) - math.log10(a))
        else:
            f = (y - a) / (b - a)
        return self.y1 - f * (self.y1 - self.y0)

    def frame(self, xticks, yticks, yfmt=lambda v: "{:g}".format(v)):
        d = self.d
        f = font(16)
        for y in yticks:
            Y = self.py(y)
            d.line([(self.x0, Y), (self.x1, Y)], fill=(232, 232, 236))
            txt = yfmt(y)
            w = d.textlength(txt, font=f)
            d.text((self.x0 - 12 - w, Y - 10), txt, font=f, fill=(70, 70, 70))
        for x in xticks:
            X = self.px(x)
            d.line([(X, self.y1), (X, self.y1 + 6)], fill=(80, 80, 80))
            txt = "{:g}".format(x)
            w = d.textlength(txt, font=f)
            d.text((X - w / 2, self.y1 + 10), txt, font=f, fill=(70, 70, 70))
        d.line([(self.x0, self.y0), (self.x0, self.y1), (self.x1, self.y1)], fill=(80, 80, 80), width=2)
        if self.xlabel:
            w = d.textlength(self.xlabel, font=font(18))
            d.text(((self.x0 + self.x1 - w) / 2, self.y1 + 34), self.xlabel,
                   font=font(18), fill=(30, 30, 30))

    def series(self, pts, color, width=3, r=5):
        if not pts:
            return
        d = self.d
        xy = [(self.px(x), self.py(y)) for x, y in pts]
        if len(xy) > 1:
            d.line(xy, fill=color, width=width)
        for X, Y in xy:
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=color)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(fh)]


def _mean_by_nvar(rows, solver, key):
    bucket = {}
    for r in rows:
        if r["solver"] != solver:
            continue
        try:
            bucket.setdefault(int(r["n_var"]), []).append(float(r[key]))
        except (KeyError, ValueError):
            return []
    if not bucket:
        return []
    return [(n, sum(v) / len(v)) for n, v in sorted(bucket.items())]


def _legend(d, y, items):
    f = font(18)
    for i, key in enumerate(items):
        x = 110 + (i % 2) * 400
        yy = y + (i // 2) * 34
        d.line([(x, yy + 10), (x + 34, yy + 10)], fill=COLORS[key], width=4)
        d.text((x + 44, yy), LABELS[key], font=f, fill=(30, 30, 30))


def nice_ticks(top: float, n: int = 5):
    raw = top / n
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= raw:
            step = mult * mag
            break
    ymax = math.ceil(top / step) * step
    return ymax, [round(step * k, 4) for k in range(int(round(ymax / step)) + 1)]


def _panel(d, box, rows, key, yr, title, logy, yfmt,
           solvers=("milp", "greedy", "sa", "sqa"), yticks=None):
    ax = Axes(d, box, (25, 390), yr, logy=logy, xlabel="QUBO 变量数 N = M x (T+1)")
    d.text((box[0], box[1] - 74), title, font=font(22), fill=(20, 20, 20))
    d.text((box[0], box[1] - 40), "耗时 (ms, 对数)" if logy else "差距 (%)",
           font=font(18), fill=(40, 40, 40))
    if yticks is None:
        if logy:
            lo = int(math.floor(math.log10(yr[0])))
            hi = int(math.ceil(math.log10(yr[1])))
            yticks = [10.0 ** k for k in range(lo, hi + 1) if yr[0] <= 10.0 ** k <= yr[1]]
        else:
            yticks = [round(yr[1] * k / 4, 2) for k in range(5)]
    ax.frame([30, 60, 96, 150, 220, 300, 390], yticks, yfmt=yfmt)
    for key_ in solvers:
        ax.series(_mean_by_nvar(rows, key_, key), COLORS[key_])


def fig_time(rows, name="fig1_time_vs_nvar.png"):
    img = Image.new("RGB", (1560, 900), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 26), "求解耗时 vs QUBO 变量数（同一台机器、同一份代码）",
           font=font(28), fill=(20, 20, 20))
    piles = [r for r in rows if r["family"] == "piles"]
    clutter = [r for r in rows if r["family"] == "clutter"]
    _panel(d, (110, 170, 790, 610), piles, "time_ms", (0.005, 12000),
           "A. 堆叠料箱（有偏序）", True, lambda v: "{:g}".format(v))
    _panel(d, (860, 170, 1540, 610), clutter, "time_ms", (0.005, 12000),
           "B. 随机散乱（含互锁）", True, lambda v: "{:g}".format(v))
    _legend(d, 720, ["milp", "greedy", "sa", "sqa"])
    d.text((40, 838), "每个点是同一规模 3 个随机实例的均值；N = 物体数 x (步数预算 + 1)。",
           font=font(17), fill=(90, 90, 90))
    img.save(os.path.join(FIG, name))
    print(name)


def fig_gap(rows, name="fig2_gap_vs_nvar.png"):
    img = Image.new("RGB", (1560, 900), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 26), "与 MILP 精确最优的差距（分数低多少 %）", font=font(28), fill=(20, 20, 20))
    piles = [r for r in rows if r["family"] == "piles"]
    clutter = [r for r in rows if r["family"] == "clutter"]
    ymax, ticks = nice_ticks(max(1.0, max(float(r["gap_pct"]) for r in rows)) * 1.15, 5)
    _panel(d, (110, 170, 790, 610), piles, "gap_pct", (0, ymax),
           "A. 堆叠料箱（有偏序）", False, lambda v: "{:g}".format(v),
           solvers=("greedy", "sa", "sqa"), yticks=ticks)
    _panel(d, (860, 170, 1540, 610), clutter, "gap_pct", (0, ymax),
           "B. 随机散乱（含互锁）", False, lambda v: "{:g}".format(v),
           solvers=("greedy", "sa", "sqa"), yticks=ticks)
    _legend(d, 720, ["greedy", "sa", "sqa"])
    d.text((40, 838), "同一规模 3 个随机实例的均值；差距 = (最优分 - 该解分数) / 最优分。MILP 定义上恒为 0。",
           font=font(17), fill=(90, 90, 90))
    img.save(os.path.join(FIG, name))
    print(name)


def _rect_pts(cx, cy, w, h, deg):
    out = []
    r = math.radians(deg)
    for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append((cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)))
    return out


def _bbox(sc, pad: float = 10.0):
    xs, ys = [], []
    for b in sc.bodies:
        for x, y in _rect_pts(b.cx, b.cy, b.w, b.h, b.theta):
            xs.append(x)
            ys.append(y)
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _draw_plan(d, sc, res, org, scale, bbox, title):
    ox, oy = org
    bx0, by0, bx1, by1 = bbox

    def T(x, y):
        return (ox + (x - bx0) * scale, oy + (y - by0) * scale)

    d.rectangle([T(bx0, by0), T(bx1, by1)], outline=(200, 200, 205), fill=(252, 252, 253))
    d.text((ox, oy - 46), title, font=font(21), fill=(20, 20, 20))
    for b in sc.bodies:
        d.polygon([T(x, y) for x, y in _rect_pts(b.cx, b.cy, b.w, b.h, b.theta)],
                  outline=(120, 120, 128), fill=(228, 228, 234))
    for k, o in enumerate(res.seq):
        g = sc.grasp(o)
        col = PALETTE[k % len(PALETTE)]
        pp = [T(x, y) for x, y in _rect_pts(g.cx, g.cy, g.w, g.h, g.theta)]
        d.polygon(pp, outline=col)
        d.line(pp + [pp[0]], fill=col, width=3)
        cx, cy = T(g.cx, g.cy)
        d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=col, outline=(255, 255, 255))
        txt = str(k + 1)
        tw = d.textlength(txt, font=font(15))
        d.text((cx - tw / 2, cy - 9), txt, font=font(15), fill=(255, 255, 255))


def fig_scene(name="fig3_seq_scene.png"):
    sc = S.make_pile_scene(n_clusters=3, per_cluster=3, seed=0, budget=5)
    T = sc.budget
    pack = qubo.build(sc, T)
    milp = V.exact_milp(sc, T, 60.0)
    sa = V.anneal_classical(sc, pack, num_reads=40, num_sweeps=1000, seed=0)
    greedy = V.greedy(sc, T)
    bw, bh = 330, 300
    img = Image.new("RGB", (1320, 520), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 20), "同一场景的三种解：粗框 = 抓取矩形，数字 = 第几步抓",
           font=font(26), fill=(20, 20, 20))
    bbox = _bbox(sc)
    scale = min(bw / (bbox[2] - bbox[0]), bh / (bbox[3] - bbox[1]))
    fw, fh = (bbox[2] - bbox[0]) * scale, (bbox[3] - bbox[1]) * scale
    for k, (res, title) in enumerate(((milp, "MILP 精确最优"), (sa, "经典退火 SA"), (greedy, "贪心剥离"))):
        sub = "{} · 分数 {:.2f} · {:.0f} ms".format(title, res.score, res.time_ms)
        cx = 205 + k * 420
        _draw_plan(d, sc, res, (cx - fw / 2, 130), scale, bbox, sub)
    d.text((40, 462),
           "场景：3 堆 x 每堆 3 个物体（灰块），只能自上而下剥离，单次最多抓 {} 个；未编号 = 该解放弃了它。".format(T),
           font=font(16), fill=(90, 90, 90))
    img.save(os.path.join(FIG, name))
    print(name)


def fig_budget(name="fig4_quality_vs_budget.png"):
    cl, pc, seed = 4, 3, 0
    img = Image.new("RGB", (1200, 840), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 26), "给更多步数预算能拿到多少分（4 堆 x 3 个）", font=font(26), fill=(20, 20, 20))
    M = cl * pc
    xs = []
    series = {"milp": [], "greedy": [], "sa": [], "sqa": []}
    for T in range(2, M + 1):
        sc = S.make_pile_scene(n_clusters=cl, per_cluster=pc, seed=seed, budget=T)
        pack = qubo.build(sc, T)
        xs.append(T)
        series["milp"].append((T, V.exact_milp(sc, T, 60.0).score))
        series["greedy"].append((T, V.greedy(sc, T).score))
        series["sa"].append((T, V.anneal_classical(sc, pack, num_reads=40,
                                                   num_sweeps=1000, seed=0).score))
        series["sqa"].append((T, V.anneal_quantum(sc, pack, num_reads=16, num_sweeps=400,
                                                  replicas=6, seed=0).score))
    ymax, yticks = nice_ticks(max(y for k in series for _, y in series[k]) * 1.08, 5)
    ax = Axes(d, (110, 130, 1140, 600), (min(xs) - 0.3, max(xs) + 0.3), (0, ymax),
              xlabel="步数预算 T（最多能抓几个）")
    d.text((110, 100), "总抓取质量分", font=font(18), fill=(40, 40, 40))
    ax.frame(xs, yticks, yfmt=lambda v: "{:g}".format(v))
    for k, col in COLORS.items():
        ax.series(series[k], col)
    _legend(d, 690, ["milp", "greedy", "sa", "sqa"])
    d.text((40, 790), "同一场景（4 堆 x 3 个物体），只放宽步数预算；曲线贴在一起说明顺序决策不难。",
           font=font(17), fill=(90, 90, 90))
    img.save(os.path.join(FIG, name))
    print(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="time,gap,scene,budget")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
        if name.startswith("bench_") and name.endswith(".csv") and "quick" not in name:
            rows += load_rows(os.path.join(RES, name))
    if not rows:
        print("没有找到 results/bench_*.csv，先跑 bench.py")
        return
    if "time" in want:
        fig_time(rows)
    if "gap" in want:
        fig_gap(rows)
    if "scene" in want:
        fig_scene()
    if "budget" in want:
        fig_budget()


if __name__ == "__main__":
    main()
