"""QUBO for pick-and-place ordering.

One block of variables per object: K*T assignment variables (slot, step) plus a
single skip variable. On any assignment that satisfies the four constraints the
energy equals minus the objective, so a solver's best feasible sample is directly
comparable with an exact solver's objective.
"""

import place

BIG = 1000.0


class Pack:
    def __init__(self, q, offset, n, n_obj, n_slot, T, lam):
        self.q = q
        self.offset = offset
        self.n = n
        self.n_obj = n_obj
        self.n_slot = n_slot
        self.T = T
        self.lam = lam

    @property
    def nnz(self):
        return len(self.q)

    def energy(self, x):
        e = self.offset
        for (i, j), v in self.q.items():
            if x[i] and x[j]:
                e += v
        return e


def layout(sc, T):
    """Variable ids: (o, c, s) for s < T, then one skip id per object."""
    block = sc.n_slot * T + 1
    def vid(o, c, s):
        return o * block + c * T + s
    def skip(o):
        return o * block + sc.n_slot * T
    return block, vid, skip


def build(sc, T=None, lam=None):
    N, K = sc.n_obj, sc.n_slot
    if T is None:
        T = max(1, min(K, N // 2))
    lam = 2.0 * T if lam is None else lam
    block, vid, skip = layout(sc, T)
    n = N * block
    q = {}
    offset = 0.0

    def add(i, j, v):
        if i > j:
            i, j = j, i
        q[(i, j)] = q.get((i, j), 0.0) + v

    # objective: reward a placement, weighted by how far the slot is
    for o in range(N):
        for c in range(K):
            if not sc.fit(o, c):
                for s in range(T):
                    add(vid(o, c, s), vid(o, c, s), BIG)
                continue
            for s in range(T):
                add(vid(o, c, s), vid(o, c, s), -sc.value(o, c))

    # one assignment (or the skip) per object
    for o in range(N):
        idx = [vid(o, c, s) for c in range(K) for s in range(T)] + [skip(o)]
        for i in idx:
            add(i, i, -lam)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], 2.0 * lam)
        offset += lam

    # at most one grasp per step
    for s in range(T):
        idx = [vid(o, c, s) for o in range(N) for c in range(K)]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], lam)

    # at most one object per tray slot
    for c in range(K):
        idx = [vid(o, c, s) for o in range(N) for s in range(T)]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], lam)

    # precedence: a blocked object cannot appear before its blockers, and a
    # skipped blocker means the blocked object can never be placed
    for o in range(N):
        for blk in sc.blocked[o]:
            for c in range(K):
                for s in range(T):
                    i = vid(o, c, s)
                    for c2 in range(K):
                        for t in range(s, T):
                            add(i, vid(blk, c2, t), lam)
                    add(i, skip(blk), lam)

    return Pack(q, offset, n, N, K, T, lam)


def decode(sc, pack, x):
    """Turn a bit vector into placements, or report why it is not feasible."""
    N, K, T = pack.n_obj, pack.n_slot, pack.T
    block, vid, skip = layout(sc, T)
    placed, skipped, used_slot, used_step = [], [], set(), set()
    for o in range(N):
        ones = [(c, s) for c in range(K) for s in range(T) if x[vid(o, c, s)]]
        if x[skip(o)]:
            if ones:
                return {"feasible": False, "reason": "skip plus placement"}
            skipped.append(o)
            continue
        if len(ones) != 1:
            return {"feasible": False, "reason": "object assigned %d times" % len(ones)}
        c, s = ones[0]
        if not sc.fit(o, c):
            return {"feasible": False, "reason": "object does not fit slot"}
        if c in used_slot:
            return {"feasible": False, "reason": "two objects in one slot"}
        if s in used_step:
            return {"feasible": False, "reason": "two grasps in one step"}
        used_slot.add(c)
        used_step.add(s)
        placed.append((o, c, s))

    placed.sort(key=lambda t: t[2])
    done = set()
    for o, c, s in placed:
        if sc.blocked[o] - done:
            return {"feasible": False, "reason": "blocked object placed too early"}
        done.add(o)
    score = sum(sc.value(o, c) for o, c, s in placed)
    return {
        "feasible": True,
        "placed": placed,
        "skipped": skipped,
        "score": score,
        "n_placed": len(placed),
        "travel": sum(sc.cost(o, c) for o, c, s in placed),
    }
