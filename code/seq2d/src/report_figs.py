"""汇报用插图：结构、编码、以及"贪心为什么输"。

生成三张给导师看的图：
  report_A_structure.png  问题结构：堆叠 -> 前置链；互锁 -> 死锁
  report_B_encoding.png   编码：z[o,s] 矩阵 + skip 列 + 四项约束
  report_C_divergence.png 24 物体实例：贪心摊平 vs 全局打通（按堆的抓取顺序矩阵）
"""

import os

from PIL import Image, ImageDraw

import plots as P
import qubo
import scene as S
import solvers as V

FIG = P.FIG
font = P.font
_rect_pts = P._rect_pts
_bbox = P._bbox

GREEN = (44, 160, 44)
RED = (214, 39, 40)
BLUE = (31, 119, 180)
GRAY_TXT = (95, 95, 100)
INK = (25, 25, 28)


def _arrow(d, p0, p1, color, width=3, head=11):
    d.line([p0, p1], fill=color, width=width)
    import math
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (-0.45, 0.45):
        d.line([p1, (p1[0] - head * math.cos(ang + s), p1[1] - head * math.sin(ang + s))],
               fill=color, width=width)


def fig_structure(name="report_A_structure.png"):
    sc = S.make_pile_scene(n_clusters=2, per_cluster=2, seed=3, budget=2)
    img = Image.new("RGB", (1280, 560), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 22), "问题结构：堆叠让“选哪些”变成“先抓哪个”", font=font(27), fill=INK)

    # 左：三层堆，上面压住下面
    d.text((70, 112), "① 料箱里的一个堆（示意）", font=font(17), fill=GRAY_TXT)
    pile = [(215, 210, 168, 62, -7), (205, 300, 178, 66, 4), (200, 392, 186, 60, -3)]
    names = ["上层", "中层", "底层"]
    for (cx, cy, w, h, deg), nm in zip(pile, names):
        d.polygon([(x, y) for x, y in _rect_pts(cx, cy, w, h, deg)],
                  outline=(115, 115, 125), fill=(230, 230, 236))
        tw = d.textlength(nm, font=font(16))
        d.text((cx - tw / 2, cy - 10), nm, font=font(16), fill=(80, 80, 88))
    # 红虚线框 = 抓取矩形，压在中间物体上
    gr = _rect_pts(215, 214, 150, 40, -7)
    for i in range(4):
        d.line([gr[i], gr[(i + 1) % 4]], fill=RED, width=2)
    d.text((300, 186), "抓取矩形", font=font(14), fill=RED)
    d.text((300, 208), "压下层 →", font=font(14), fill=RED)
    # 中间那道向下箭头
    ax = 470
    for (y0, y1) in ((210, 300), (300, 392)):
        _arrow(d, (ax, y0 - 12), (ax, y1 + 12), BLUE, width=3)
    d.text((ax + 16, 236), "必须先", font=font(14), fill=BLUE)
    d.text((ax + 16, 258), "抓走", font=font(14), fill=BLUE)
    d.text((70, 470), "抓取矩形与下层实体重叠 > 阈值 →\n下层“被挡”，只能等上层被移走",
           font=font(15), fill=GRAY_TXT)

    # 中：链，有解
    gx, gy = 600, 170
    d.text((gx - 30, gy - 46), "② 前置图：链（有解）", font=font(17), fill=GRAY_TXT)
    nodes = {"上层": (gx + 130, gy + 30), "下层": (gx + 130, gy + 150)}
    for label, (x, y) in nodes.items():
        d.ellipse([x - 44, y - 26, x + 44, y + 26], outline=BLUE, width=3, fill=(240, 246, 252))
        tw = d.textlength(label, font=font(16))
        d.text((x - tw / 2, y - 10), label, font=font(16), fill=INK)
    _arrow(d, (gx + 130, gy + 58), (gx + 130, gy + 122), BLUE, width=3)
    d.text((gx + 186, gy + 74), "必须先抓", font=font(14), fill=BLUE)
    d.text((gx - 10, gy + 200), "顺序存在 → 可解，只是要选一条\n收益最高的序列", font=font(15), fill=GRAY_TXT)

    # 右：环，谁都抓不了
    cx0, cy0 = 1000, 255
    d.text((cx0 - 60, gy - 46), "③ 互锁：环（死锁）", font=font(17), fill=GRAY_TXT)
    pa = (cx0 - 60, cy0)
    pb = (cx0 + 70, cy0)
    for label, (x, y), col in (("A", pa, RED), ("B", pb, RED)):
        d.ellipse([x - 44, y - 26, x + 44, y + 26], outline=col, width=3, fill=(253, 242, 242))
        tw = d.textlength(label, font=font(16))
        d.text((x - tw / 2, y - 10), label, font=font(16), fill=INK)
    _arrow(d, (pa[0] + 46, pa[1] - 12), (pb[0] - 46, pb[1] - 12), RED, width=3)
    _arrow(d, (pb[0] - 46, pb[1] + 14), (pa[0] + 46, pa[1] + 14), RED, width=3)
    d.text((cx0 - 74, cy0 + 150), "A 挡 B 且 B 挡 A → 谁都抓不了：\n单次抓取的“选集合”在此处直接失效",
           font=font(15), fill=GRAY_TXT)

    img.save(os.path.join(FIG, name))
    print(name)


def fig_encoding(name="report_B_encoding.png"):
    sc = S.make_pile_scene(n_clusters=1, per_cluster=4, seed=1, budget=3)
    T = sc.budget
    res = V.exact_milp(sc, T, 60.0)
    img = Image.new("RGB", (1280, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 22), "编码：每个物体一行，每步一列，多一个“不抓”列", font=font(27), fill=INK)

    x0, y0, cw, ch = 250, 150, 104, 66
    cols = T + 1
    d.text((40, y0 - 40), "物体（分数 / 谁挡它）", font=font(15), fill=GRAY_TXT)
    for s in range(cols):
        label = "第{}步".format(s + 1) if s < T else "不抓"
        col = GRAY_TXT if s < T else RED
        tw = d.textlength(label, font=font(15))
        d.text((x0 + s * cw + (cw - 8) / 2 - tw / 2, y0 - 40), label, font=font(15), fill=col)

    pos = {}
    for k, o in enumerate(res.seq):
        pos[o] = k
    for r in range(sc.n_obj):
        y = y0 + r * ch
        bl = sorted(sc.blocked[r])
        head = "obj{}  {:.2f}".format(r, sc.score(r))
        if bl:
            head += "  被 {} 挡".format(",".join("obj{}".format(b) for b in bl))
        d.text((40, y + 22), head, font=font(15), fill=INK)
        for s in range(cols):
            x = x0 + s * cw
            got = pos.get(r) == s or (r not in pos and s == T)
            fill = (232, 246, 232) if got else (250, 250, 251)
            outline = GREEN if got else (215, 215, 220)
            d.rectangle([x, y, x + cw - 8, y + ch - 8], fill=fill, outline=outline, width=2)
            if got:
                txt = "1"
                tw = d.textlength(txt, font=font(20))
                d.text((x + (cw - 8) / 2 - tw / 2, y + 14), txt, font=font(20), fill=GREEN)

    # 四项罚，一行一句
    lx, ly = 800, 140
    d.text((lx, ly - 36), "能量函数 = 四段惩罚 + 一个目标", font=font(18), fill=INK)
    items = [
        (BLUE, "目标  −Σ score·z", "抓到高分物体就减能量"),
        (GREEN, "每行恰好一个 1", "每个物体只能被抓一次，或被放弃"),
        (RED, "每列最多一个 1", "一步只能抓一个（机械臂）"),
        ((150, 90, 160), "前置  z[o,q] 与 z[o',p≥q] 同时为 1 罚 λ", "含 p = 不抓：被挡却没先移走 → 一定被罚"),
    ]
    y = ly + 8
    for col, head, sub in items:
        d.rectangle([lx, y + 6, lx + 14, y + 20], fill=col)
        d.text((lx + 26, y), head, font=font(16), fill=INK)
        d.text((lx + 26, y + 26), sub, font=font(14), fill=GRAY_TXT)
        y += 78

    d.rectangle([lx, y + 10, lx + 430, y + 96], fill=(244, 248, 244), outline=(200, 220, 200))
    d.text((lx + 16, y + 22), "可行解上恒有   E(z) = − 总分", font=font(20), fill=GREEN)
    d.text((lx + 16, y + 58), "所以罚权重只要够大就行，不用精细调参", font=font(14), fill=GRAY_TXT)

    d.text((40, 552),
           "注：每物体只保留分数最高的 1 个候选（否则变量数 ×3）。这是为控规模的简化，见讲解文档第 6 节。",
           font=font(14), fill=GRAY_TXT)
    img.save(os.path.join(FIG, name))
    print(name)


def fig_divergence(name="report_C_divergence.png"):
    M_CL, PER, T = 6, 4, 14
    sc = S.make_pile_scene(n_clusters=M_CL, per_cluster=PER, seed=0, budget=T)
    greedy = V.greedy(sc, T)
    milp = V.exact_milp(sc, T, 120.0)

    img = Image.new("RGB", (1300, 680), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 20), "24 个物体、只能抓 14 个：贪心摊平，全局打通", font=font(27), fill=INK)
    d.text((40, 62),
           "同一场景、同一套 QUBO。格子 = 物体（上→下 = 堆的层），数字 = 第几步抓，灰 = 放弃。",
           font=font(15), fill=GRAY_TXT)

    def block(res, ox, oy, title, sub):
        d.text((ox, oy), title, font=font(19), fill=INK)
        d.text((ox, oy + 28), sub, font=font(14), fill=GRAY_TXT)
        pos = {o: k for k, o in enumerate(res.seq)}
        cw, chh = 92, 62
        gy = oy + 84
        for cl in range(M_CL):
            x = ox + cl * (cw + 6)
            for lv in range(PER - 1, -1, -1):
                o = cl * PER + lv
                y = gy + (PER - 1 - lv) * chh
                got = o in pos
                fill = GREEN if got else (238, 238, 240)
                d.rectangle([x, y, x + cw, y + chh - 8], fill=fill, outline=(210, 210, 215))
                if got:
                    txt = str(pos[o] + 1)
                    tw = d.textlength(txt, font=font(19))
                    d.text((x + cw / 2 - tw / 2, y + 13), txt, font=font(19), fill=(255, 255, 255))
            n = sum(1 for lv in range(PER) if cl * PER + lv in pos)
            tw = d.textlength("堆{}".format(cl + 1), font=font(15))
            d.text((x + cw / 2 - tw / 2, oy + 56), "堆{}".format(cl + 1), font=font(15), fill=INK)
            d.text((x + cw / 2 - 22, gy + PER * chh - 2), "{}/{}".format(n, PER),
                   font=font(15), fill=GREEN if n == PER else GRAY_TXT)

    block(greedy, 40, 118, "贪心剥离 · 分数 9.820",
          "每步都拿“当前可抓的最高分” → 槽位摊在六堆表面")
    block(milp, 680, 118, "QUBO + MILP 全局最优 · 分数 10.499  (+6.46%)",
          "先付一次低价开通费，把两堆铲到底")

    # MILP 第一步：obj19，第 5 堆最上层
    o = 19
    cl, lv = o // PER, o % PER
    x = 680 + cl * 98
    y = 118 + 84 + (PER - 1 - lv) * 62
    d.rectangle([x - 4, y - 4, x + 96, y + 50], outline=RED, width=3)
    d.text((40, 500), "贪心在第 1 步：抓 obj11(0.822)", font=font(16), fill=INK)
    d.text((40, 528), "→ 之后再也摸不到 obj4/5/6（第 2 堆底部，0.880/0.849/0.810）", font=font(16), fill=RED)
    d.text((40, 556), "→ 也摸不到 obj18/16（第 5 堆，0.825/0.819）", font=font(16), fill=RED)

    d.text((680, 500), "全局解在第 1 步：抓 obj19（只有 0.534，红框）", font=font(16), fill=INK)
    d.text((680, 528), "→ 用 0.534 换来 obj18(0.825)+obj17+obj16(0.819)", font=font(16), fill=GREEN)
    d.text((680, 556), "→ 以及第 2 堆底部 obj5/obj4（0.849/0.880）", font=font(16), fill=GREEN)
    d.text((40, 606),
           "这 6.46% 的差距全部来自一个取舍：预算有限时，要不要先花一步“开通”一条深堆。",
           font=font(17), fill=INK)
    img.save(os.path.join(FIG, name))
    print(name)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_structure()
    fig_encoding()
    fig_divergence()


if __name__ == "__main__":
    main()
