"""Planar geometry: oriented rectangles, convex clipping, overlap ratios."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def corners(cx: float, cy: float, w: float, h: float, theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([cx, cy])


def area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def _cross(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_intersect(p, q, a, b):
    x1, y1 = p
    x2, y2 = q
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < EPS:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    out = [tuple(p) for p in subject]
    n = len(clipper)
    for i in range(n):
        a = tuple(clipper[i])
        b = tuple(clipper[(i + 1) % n])
        if not out:
            break
        inp, out = out, []
        for j in range(len(inp)):
            cur = inp[j]
            nxt = inp[(j + 1) % len(inp)]
            cur_in = _cross(a, b, cur) >= -EPS
            nxt_in = _cross(a, b, nxt) >= -EPS
            if cur_in:
                out.append(cur)
            if cur_in != nxt_in:
                out.append(_line_intersect(cur, nxt, a, b))
    return np.array(out) if out else np.zeros((0, 2))


def inter_area(a: np.ndarray, b: np.ndarray) -> float:
    return area(clip(a, b))


def ios(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over smaller area."""
    aa, ab = area(a), area(b)
    if aa < EPS or ab < EPS:
        return 0.0
    return inter_area(a, b) / min(aa, ab)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, ab = area(a), area(b)
    ia = inter_area(a, b)
    union = aa + ab - ia
    return ia / union if union > EPS else 0.0
