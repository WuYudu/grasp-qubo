"""fig5: fixed best pose vs joint pose+order.  python src/plot_cand.py"""
from __future__ import annotations

import csv
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def main():
    with open(os.path.join(RES, "bench_cand.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ns = sorted({int(r["n_obj"]) for r in rows})
    fx, jt = [], []
    for n in ns:
        g = [r for r in rows if int(r["n_obj"]) == n]
        fx.append(sum(float(r["score_fixed"]) for r in g) / len(g))
        jt.append(sum(float(r["score_joint"]) for r in g) / len(g))

    img = Image.new("RGB", (1180, 620), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 24), "只优化顺序 vs 同时优化抓取位姿：分数随物体数", font=font(26), fill=(20, 20, 20))
    L, R, T_, B = 110, 1080, 120, 500
    ymax = max(max(fx), max(jt)) * 1.18
    d.line([(L, B), (R, B)], fill=(60, 60, 60), width=2)
    d.line([(L, T_), (L, B)], fill=(60, 60, 60), width=2)
    for k in range(5):
        y = B - (B - T_) * k / 4
        v = ymax * k / 4
        d.line([(L, y), (R, y)], fill=(232, 232, 232), width=1)
        d.text((L - 62, y - 10), "{:.1f}".format(v), font=font(16), fill=(90, 90, 90))
    x0, x1 = min(ns), max(ns)

    def px(n):
        return L + (R - L) * (n - x0) / max(1, (x1 - x0))

    def py(v):
        return B - (B - T_) * v / ymax

    for series, col, lab in ((fx, (110, 110, 110), "只调顺序（固定最高分位姿）"),
                             (jt, (31, 119, 180), "位姿 + 顺序一起调")):
        pts = [(px(n), py(v)) for n, v in zip(ns, series)]
        d.line(pts, fill=col, width=4)
        for x, y in pts:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col)
    for n, v in zip(ns, fx):
        d.text((px(n) - 26, py(v) + 12), "{:.2f}".format(v), font=font(15), fill=(110, 110, 110))
    for n, v in zip(ns, jt):
        d.text((px(n) - 26, py(v) - 30), "{:.2f}".format(v), font=font(15), fill=(31, 119, 180))

    d.text(((L + R) / 2 - 70, B + 14), "物体数 M", font=font(18), fill=(60, 60, 60))
    lx, ly = L, 540
    for i, (col, lab) in enumerate((((110, 110, 110), "只调顺序（固定最高分位姿）"),
                                    ((31, 119, 180), "位姿 + 顺序一起调"))):
        x = lx + i * 380
        d.line([(x, ly + 10), (x + 34, ly + 10)], fill=col, width=4)
        d.text((x + 44, ly), lab, font=font(17), fill=(40, 40, 40))
    d.text((40, 578), "每点 = 同规模 3 个种子的均值；拥挤场景（重叠上限 0.62），每个物体 4 个候选位姿。",
           font=font(16), fill=(90, 90, 90))
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig5_pose_vs_order.png")
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
