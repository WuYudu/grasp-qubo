"""Scenes at shelf scale, sub-scene slicing, and the disturbance model."""
import math
import random
from dataclasses import replace

import geom
import scene as S


def big_clutter(n_obj, seed=0, density=1.0, max_overlap=0.35):
    """Same object density as the small scenes, canvas grows with sqrt(N)."""
    side = math.sqrt(n_obj / 12.0) * density
    canvas = (S.CANVAS[0] * side, S.CANVAS[1] * side)
    return S.make_scene(n_obj=n_obj, seed=seed, canvas=canvas,
                        max_overlap=max_overlap, margin=16.0)


def big_piles(n_clusters, per_cluster, seed=0, density=1.0):
    side = math.sqrt((n_clusters * per_cluster) / 9.0) * density
    canvas = (S.CANVAS[0] * side, S.CANVAS[1] * side)
    return S.make_pile_scene(n_clusters=n_clusters, per_cluster=per_cluster,
                             seed=seed, canvas=canvas)


def subset(sc, keep):
    """Scene containing only `keep` (global indices); returns (scene, keep_sorted)."""
    keep = sorted(keep)
    remap = {o: i for i, o in enumerate(keep)}
    bodies = [sc.bodies[o] for o in keep]
    grasps = [sc.grasps[o] for o in keep]
    chosen = [sc.chosen[o] for o in keep]
    blocked = [{remap[p] for p in sc.blocked[o] if p in remap} for o in keep]
    levels = [sc.levels[o] for o in keep] if sc.levels else [0] * len(keep)
    quality = [sc.quality[o] for o in keep] if sc.quality else [1.0] * len(keep)
    cand = None
    if sc.cand_blocked:
        cand = [[{remap[p] for p in cb if p in remap} for cb in sc.cand_blocked[o]]
                for o in keep]
    return S.Scene(bodies, grasps, chosen, blocked, sc.budget, levels,
                   sc.family, cand, quality), keep


def clusters(sc, touch=0.02):
    """Group bodies that overlap each other - one stack per group."""
    parent = list(range(sc.n_obj))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(sc.n_obj):
        for j in range(i + 1, sc.n_obj):
            if geom.ios(sc.bodies[i].poly, sc.bodies[j].poly) > touch:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for o in range(sc.n_obj):
        groups.setdefault(find(o), []).append(o)
    return list(groups.values())


def rebuild(sc, eps=0.10):
    """Recompute grasp candidates and blocking after the bodies moved.

    Inside a stack only the bodies above you block you (smaller cy = higher up).
    Across stacks, blocking is whatever your grasp rectangle physically runs into.
    """
    grasps = [S._candidates(b, o, sc.quality[o], 26.0, S.MAX_OPENING)
              for o, b in enumerate(sc.bodies)]
    chosen = [max(range(len(g)), key=lambda c: g[c].score) for g in grasps]
    group_of = {}
    for gi, grp in enumerate(clusters(sc)):
        for o in grp:
            group_of[o] = gi
    blocked = []
    for o in range(sc.n_obj):
        poly = grasps[o][chosen[o]].poly
        blk = {p for p in range(sc.n_obj)
               if p != o and geom.ios(poly, sc.bodies[p].poly) > eps
               and group_of[p] != group_of[o]}
        blk |= {p for p in range(sc.n_obj)
                if group_of.get(p) == group_of[o] and sc.bodies[p].cy < sc.bodies[o].cy}
        blocked.append(blk)
    return replace(sc, grasps=grasps, chosen=chosen, blocked=blocked)


def disturb(sc, rng, sigma=4.0, slip_p=0.10, slip_dist=(10.0, 26.0)):
    """Gaussian jitter on every body, plus an occasional slip of one body."""
    bodies = list(sc.bodies)
    for o, b in enumerate(bodies):
        bodies[o] = replace(b, cx=b.cx + rng.gauss(0.0, sigma),
                            cy=b.cy + rng.gauss(0.0, sigma))
    if rng.random() < slip_p:
        o = rng.randrange(len(bodies))
        bodies[o] = _slip(bodies[o], rng, slip_dist)
    return rebuild(replace(sc, bodies=bodies))


def _slip(b, rng, slip_dist=(10.0, 26.0)):
    r = rng.uniform(*slip_dist)
    a = rng.uniform(0.0, 2.0 * math.pi)
    return replace(b, cx=b.cx + r * math.cos(a), cy=b.cy + r * math.sin(a))


def nudge(sc, rng, objs, slip_dist=(12.0, 30.0)):
    """Push one of the remaining bodies aside - the rescue move when the bin locks up."""
    objs = list(objs)
    if not objs:
        return sc
    bodies = list(sc.bodies)
    o = objs[rng.randrange(len(objs))]
    bodies[o] = _slip(bodies[o], rng, slip_dist)
    return rebuild(replace(sc, bodies=bodies))
