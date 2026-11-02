"""Two figures from results/bench_main.csv."""

import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
FONT = r"C:\Windows\Fonts\msyh.ttc"
INK, GREY = (32, 32, 32), (120, 120, 120)
COL = {"greedy": (31, 119, 180), "milp": (44, 160, 44), "sa": (214, 39, 40)}
NAME = {"greedy": "贪心", "milp": "精确 MILP", "sa": "退火（模拟）"}


def font(s):
    return ImageFont.truetype(FONT, s)


def load():
    with open(os.path.join(ROOT, "results", "bench_main3.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("n_obj", "placed", "actions", "cycles", "fails", "rounds"):
            r[k] = int(float(r[k]))
        for k in ("rate", "per_placed", "p50", "p99", "wall_s"):
            r[k] = float(r[k])
    return rows


def bars(rows, key, title, ylabel, fname, fmt="{:.0f}", logy=False, note=""):
    sizes = sorted({r["n_obj"] for r in rows})
    plans = ["greedy", "milp", "sa"]
    img = Image.new("RGB", (1180, 660), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 24), title, font=font(27), fill=INK)
    box = (110, 110, 1130, 520)
    d.rectangle(box, outline=(190, 196, 204))
    vals = [r[key] for r in rows]
    hi = max(vals) * 1.18
    lo = min(vals) * 0.6 if logy else 0.0
    def ypos(v):
        if logy:
            a, b = math.log10(max(lo, 1e-6)), math.log10(max(hi, 1e-6))
            return box[3] - (math.log10(max(v, 1e-6)) - a) / (b - a) * (box[3] - box[1])
        return box[3] - (v - lo) / (hi - lo) * (box[3] - box[1])
    ticks = [10 ** k for k in range(int(math.log10(max(lo, 1e-6))), int(math.ceil(math.log10(hi))) + 1)] if logy \
        else [hi * k / 4 for k in range(5)]
    for t in ticks:
        y = ypos(t)
        d.line([(box[0], y), (box[2], y)], fill=(228, 232, 238))
        d.text((box[0] - 10, y), fmt.format(t), font=font(15), fill=GREY, anchor="rm")
    d.text((box[0] - 74, box[1] - 26), ylabel, font=font(16), fill=GREY)
    gw = (box[2] - box[0]) / len(sizes)
    bw = gw / 4.2
    for si, n in enumerate(sizes):
        for pi, pl in enumerate(plans):
            sel = [r for r in rows if r["n_obj"] == n and r["planner"] == pl]
            if not sel:
                continue
            v = sum(r[key] for r in sel) / len(sel)
            x = box[0] + gw * (si + 0.5) + (pi - 1) * bw * 1.1
            y = ypos(v)
            d.rectangle([x - bw / 2, y, x + bw / 2, box[3]], fill=COL[pl])
            d.text((x, y - 6), fmt.format(v), font=font(15), fill=COL[pl], anchor="mb")
        d.text((box[0] + gw * (si + 0.5), box[3] + 10), "N=%d" % n, font=font(17), fill=GREY, anchor="ma")
    lx = 110
    for pl in plans:
        d.rectangle([lx, 574, lx + 22, 586], fill=COL[pl])
        d.text((lx + 30, 572), NAME[pl], font=font(17), fill=INK)
        lx += 190
    if note:
        d.text((40, 618), note, font=font(16), fill=GREY)
    img.save(os.path.join(FIG, fname))


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    bars(rows, "rate", "闭环里的清单完成率（放置会失败，托盘满即换）",
         "完成率 (%)", "fig1_rate.png", fmt="{:.0f}",
         note="每点是两个种子的均值；动作预算 4N，超出即停。")
    bars(rows, "p99", "单次重规划的尾部延迟", "毫秒", "fig2_latency.png",
         fmt="{:.0f}", note="p99 是决定节拍会不会卡住的那个数。")
    print("figures ok")


if __name__ == "__main__":
    main()
