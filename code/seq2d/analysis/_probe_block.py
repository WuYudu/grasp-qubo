"""Probe: how dense is the blocking graph as a function of corridor size?"""
import geom
import scene as S


def stats(M, ov, finger, eps, seed):
    sc = S.make_scene(n_obj=M, seed=seed, max_overlap=ov)
    bodies = sc.bodies
    grasps = [S._candidates(b, o, 1.0) for o, b in enumerate(bodies)]
    bl = []
    for o in range(M):
        g = grasps[o][1]
        poly = geom.corners(g.cx, g.cy, g.w, finger, g.theta)
        bl.append({p for p, b in enumerate(bodies) if p != o and geom.ios(poly, b.poly) > eps})
    free = sum(1 for x in bl if not x)
    return free, sum(len(x) for x in bl) / M


for M, ov in ((10, 0.45), (12, 0.45), (10, 0.35), (16, 0.45)):
    for finger in (20.0, 26.0, 34.0, 46.0):
        for eps in (0.04, 0.12, 0.25):
            f, d = stats(M, ov, finger, eps, 0)
            print("M={} ov={} finger={} eps={} free={}/{} mean_deg={:.2f}".format(
                M, ov, finger, eps, f, M, d))
