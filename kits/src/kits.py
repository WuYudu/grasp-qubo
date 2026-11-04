"""成套搬运：成对物品必须连着抓，否则先抓的那件会倒下。

场景：若干列，每列 2-3 件。列内上面的物体压住下面的（前置约束照旧）。
额外规则：预先给定若干"成套对"（同一套工具的两件），两件必须**连续两步**抓完；
如果抓了一件之后下一步没抓它的搭档，那件就倒塌损毁，整套报废。

收益：每抓一件给 0.1 的分；一对凑齐（连续两步）再给 0.9。整套收益不可加，
贪心按单件分数排会自然地把两件拆开，这是这个任务想测的东西。
"""

import random

import dimod
import numpy as np
import pulp
from dwave.samplers import SimulatedAnnealingSampler

W_ITEM = 0.10
W_PAIR = 0.90
LAM = 3.0


# ---------------------------------------------------------------- 场景

class World:
    def __init__(self, n_col=4, per_col=3, seed=0, n_pairs=None):
        rng = random.Random(seed)
        self.items = []          # dict(id, col, level, got, dead)
        self.cols = []
        oid = 0
        for c in range(n_col):
            h = rng.randint(2, per_col)
            col = []
            for lv in range(h):
                self.items.append({"id": oid, "col": c, "level": lv,
                                   "got": False, "dead": False})
                col.append(oid)
                oid += 1
            self.cols.append(col)
        self.n = oid
        tops = [c[-1] for c in self.cols]
        rng.shuffle(tops)
        k = n_pairs if n_pairs is not None else len(tops) // 2
        self.pairs = [(tops[2 * i], tops[2 * i + 1]) for i in range(k)]
        self.mate = {}
        for a, b in self.pairs:
            self.mate[a], self.mate[b] = b, a

    def alive(self):
        return [i for i in range(self.n) if not self.items[i]["got"] and not self.items[i]["dead"]]

    def blocked(self):
        """列内被上面物体压住即不可抓。"""
        out = {}
        for i in self.alive():
            it = self.items[i]
            above = [j for j in self.cols[it["col"]]
                     if self.items[j]["level"] > it["level"] and not self.items[j]["got"]
                     and not self.items[j]["dead"]]
            out[i] = above
        return out

    def grasp(self, i):
        self.items[i]["got"] = True

    def kill(self, i):
        if not self.items[i]["got"] and not self.items[i]["dead"]:
            self.items[i]["dead"] = True
            return True
        return False


# ---------------------------------------------------------------- QUBO

def build(world, T, horizon=None):
    """变量 z[i, s]：物体 i 在第 s 步被抓；s = T 表示不抓。"""
    n, S = world.n, T + 1
    Q = {}

    def add(a, b, v):
        key = (a, b) if a <= b else (b, a)
        Q[key] = Q.get(key, 0.0) + v

    for i in range(n):
        for s in range(T):
            add(i * S + s, i * S + s, -W_ITEM)
        for s in range(S):
            add(i * S + s, i * S + s, -LAM)
            for s2 in range(s + 1, S):
                add(i * S + s, i * S + s2, 2 * LAM)

    for s in range(T):
        for a in range(n):
            for b in range(a + 1, n):
                add(a * S + s, b * S + s, LAM)

    blk = world.blocked()
    for i, ups in blk.items():
        for j in ups:
            for q in range(T):
                for t in range(q, S):
                    add(i * S + q, j * S + t, LAM)

    # 成套对：连续两步各抓一件，给一次奖励
    for a, b in world.pairs:
        for s in range(T - 1):
            if horizon is not None and s + 1 >= horizon:
                break
            add(a * S + s, b * S + (s + 1), -W_PAIR)
            add(b * S + s, a * S + (s + 1), -W_PAIR)
    return {"Q": Q, "n": n * S, "S": S, "T": T, "n_obj": n, "offset": n * LAM}


def decode(pack, sample):
    S, T, n = pack["S"], pack["T"], pack["n_obj"]
    slot = {}
    for i in range(n):
        picked = [s for s in range(S) if sample.get(i * S + s, 0) == 1]
        if len(picked) != 1:
            return None
        slot[i] = picked[0]
    seq = sorted((s, i) for i, s in slot.items() if s < T)
    if len({s for s, _ in seq}) != len(seq):
        return None
    return [i for _, i in seq]


def to_bqm(pack):
    return dimod.BinaryQuadraticModel.from_qubo(dict(pack["Q"]), offset=pack["offset"])


# ---------------------------------------------------------------- 后端

def greedy(world, T):
    """按单件价值排序（不知道成套对），有前置检查。"""
    got, seq = set(), []
    for _ in range(T):
        cand = [i for i in world.alive() if i not in got
                and not any(j not in got and j in world.blocked().get(i, [])
                            for j in world.alive())]
        if not cand:
            break
        pick = max(cand, key=lambda i: (W_ITEM, -world.items[i]["level"]))
        seq.append(pick)
        got.add(pick)
    return seq


def greedy_pair(world, T):
    """知道成套对的贪心：抓一件后立刻抓搭档（可行的前提下）。"""
    got, seq = set(), []
    while len(seq) < T:
        blk = world.blocked()
        cand = [i for i in world.alive() if i not in got
                and not any(j not in got for j in blk.get(i, []))]
        if not cand:
            break
        if seq and world.mate.get(seq[-1]) in cand:
            pick = world.mate[seq[-1]]
        else:
            pick = max(cand, key=lambda i: (W_PAIR if world.mate.get(i) in cand else 0.0,
                                            W_ITEM))
        seq.append(pick)
        got.add(pick)
    return seq


def repair(world, pack, sample):
    """把违反约束的样本修成一条可执行序列：按槽位顺序取，跳过重复与前置未满足。"""
    S, T, n = pack["S"], pack["T"], pack["n_obj"]
    order = []
    for s in range(T):
        for i in range(n):
            if sample.get(i * S + s, 0) == 1 and i not in order:
                order.append(i)
    blk = world.blocked()
    seq, got = [], set()
    for i in order:
        if any(j not in got for j in blk.get(i, [])):
            continue
        seq.append(i)
        got.add(i)
    return seq


def seq_score(world, seq):
    sc = W_ITEM * len(seq)
    for k in range(len(seq) - 1):
        if world.mate.get(seq[k]) == seq[k + 1]:
            sc += W_PAIR
    return sc


def anneal(pack, reads=40, sweeps=800, seed=0, world=None):
    ss = SimulatedAnnealingSampler().sample(to_bqm(pack), num_reads=reads,
                                            num_sweeps=sweeps, seed=seed)
    best, best_score = [], -1e9
    for rec in ss.record:
        sample = dict(zip(ss.variables, rec.sample))
        seq = decode(pack, sample)
        if seq is None or world is None:
            cand = [seq] if seq else []
        else:
            cand = [seq, repair(world, pack, sample)]
        for c in cand:
            if not c:
                continue
            sc = seq_score(world, c) if world is not None else plan_score(c, pack)
            if sc > best_score:
                best, best_score = list(c), sc
    return best


def plan_score(seq, pack):
    """计划内部打分：单件 + 连续成套对。"""
    sc = W_ITEM * len(seq)
    pos = {i: k for k, i in enumerate(seq)}
    n, S = pack["n_obj"], pack["S"]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return sc


def evaluate(world, seq):
    """在真实世界里模拟执行 seq，返回（得分, 完成对数, 损毁数）。"""
    from copy import deepcopy
    w = deepcopy(world)
    last, score, done_pairs, killed = None, 0.0, 0, 0
    for i in seq:
        if w.items[i]["got"] or w.items[i]["dead"]:
            continue
        blk = w.blocked()
        if any(not w.items[j]["got"] for j in blk.get(i, [])):
            continue
        ok = w.grasp(i, last)
        killed += int(ok)
        score += W_ITEM
        if last is not None and w.mate.get(i) == last:
            score += W_PAIR
            done_pairs += 1
        last = i
    return score, done_pairs, killed


def milp(world, T, horizon=None):
    """精确上界：线性化 "相邻成套对" 的乘积项。"""
    n, S = world.n, T + 1
    prob = pulp.LpProblem("kits", pulp.LpMaximize)
    z = {(i, s): prob.add_variable(f"z_{i}_{s}", cat=pulp.LpBinary)
         for i in range(n) for s in range(S)}
    # 线性化 u[i,s] = z[i,s] * z[mate(i), s+1]
    u = {}
    for a, b in world.pairs:
        for s in range(T - 1):
            if horizon is not None and s + 1 >= horizon:
                break
            for (x, y) in ((a, b), (b, a)):
                v = prob.add_variable(f"u_{x}_{s}", cat=pulp.LpBinary)
                u[(x, s)] = v
                prob += v <= z[(x, s)]
                prob += v <= z[(y, s + 1)]
                prob += v >= z[(x, s)] + z[(y, s + 1)] - 1
    prob += pulp.lpSum(W_ITEM * z[(i, s)] for i in range(n) for s in range(T)) \
        + pulp.lpSum(W_PAIR * v for v in u.values())
    for i in range(n):
        prob += pulp.lpSum(z[(i, s)] for s in range(S)) == 1
    for s in range(T):
        prob += pulp.lpSum(z[(i, s)] for i in range(n)) <= 1
    blk = world.blocked()
    for i, ups in blk.items():
        for j in ups:
            for q in range(T):
                prob += z[(i, q)] <= pulp.lpSum(z[(j, t)] for t in range(q))
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
        prob.solve(pulp.HiGHS(msg=False, timeLimit=3))
    seq = []
    for s in range(T):
        for i in range(n):
            if z[(i, s)].value() and z[(i, s)].value() > 0.5:
                seq.append(i)
    return seq


# ---------------------------------------------------------------- 闭环

def run(world, planner, T=5, horizon=2, p_fail=0.15, seed=0, max_steps=40, reads=40, sweeps=800):
    w = type(world)(n_col=0)  # 占位，下面直接深拷贝原世界
    import copy
    w = copy.deepcopy(world)
    last, pending, score, killed, actions, retries = None, None, 0.0, 0, 0, 0
    done_pairs = 0
    stall = 0
    rng = random.Random(seed)
    plan, left = [], 0
    while actions < max_steps and w.alive():
        if left <= 0:
            blk_ok = True
            if planner == "greedy":
                plan = greedy(w, T)
            elif planner == "greedy_pair":
                plan = greedy_pair(w, T)
            elif planner == "sa":
                plan = anneal(build(w, T, horizon), reads=reads, sweeps=sweeps, seed=seed, world=w)
            else:
                plan = milp(w, T, horizon)
            if pending is not None and not w.items[pending]['got'] and not w.items[pending]['dead']:
                plan = [pending] + [p for p in plan if p != pending]
            left = horizon
        left -= 1
        if not plan:
            break
        # 沿计划找第一个在真实世界里确实能抓的物体；都不可行就重新规划
        i = None
        blk = w.blocked()
        while plan:
            cand = plan.pop(0)
            if cand is None or cand >= w.n or w.items[cand]["got"] or w.items[cand]["dead"]:
                continue
            if any(not w.items[j]["got"] for j in blk.get(cand, [])):
                continue
            i = cand
            break
        if i is None:
            left = 0
            stall += 1
            if stall > 2:
                break
            continue
        stall = 0
        actions += 1
        if rng.random() < p_fail:      # 抓取失败：位置不变，什么都得不到
            retries += 1
            left = 0
            continue
        if pending is not None and i != pending:
            killed += int(w.kill(pending))
        w.grasp(i)
        score += W_ITEM
        if pending is not None and i == pending:
            score += W_PAIR
            done_pairs += 1
        pending = w.mate.get(i)
        if pending is not None and (w.items[pending]["got"] or w.items[pending]["dead"]):
            pending = None
        last = i
    total_pairs = len(world.pairs)
    return {"pairs": total_pairs, "done": done_pairs, "score": score,
            "killed": killed, "actions": actions, "retries": retries}


if __name__ == "__main__":
    import itertools
    print(f"{'planner':10s} {'cols':>4s} {'N':>3s} {'T':>2s} {'h':>2s} {'pairs':>5s} {'done':>5s} {'rate':>6s} {'score':>6s} {'kill':>5s} {'act':>4s}")
    for planner in ("greedy", "greedy_pair", "milp", "sa"):
        for n_col, h in itertools.product((3, 4, 5), (2, 3)):
            rows = [run(World(n_col=n_col, per_col=3, seed=s), planner, T=5, horizon=h, seed=s)
                    for s in range(3)]
            k = len(rows)
            pairs = sum(r["pairs"] for r in rows) / k
            done = sum(r["done"] for r in rows) / k
            print(flush=True, *[], end="") or print(f"{planner:10s} {n_col:4d} {n_col*2:3d} {5:2d} {h:2d} {pairs:5.1f} {done:5.1f} "
                  f"{done/max(1e-9,pairs):6.0%} {sum(r['score'] for r in rows)/k:6.2f} "
                  f"{sum(r['killed'] for r in rows)/k:5.1f} {sum(r['actions'] for r in rows)/k:4.1f}")
