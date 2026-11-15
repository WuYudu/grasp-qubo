"""真机路径：提交前先算清规模和嵌入代价，再决定要不要花机时。

用法：
    python qpu.py --dry-run          只做体检，不需要 token
    python qpu.py --reads 100        真的提交（需要 DWAVE_API_TOKEN）
"""
import argparse
import os
import time

import dimod

import backend
import seq
import world


def describe(q):
    n = 1 + max(max(i, j) for i, j in q)
    deg = {}
    for (i, j) in q:
        if i == j:
            continue
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    dmax = max(deg.values()) if deg else 0
    n_int = sum(1 for (i, j) in q if i != j)
    return dict(variables=n, interactions=n_int, max_degree=dmax)


def embedding_estimate(info, degree_cap=15):
    """粗略估计：每条链至少 1 个物理比特，度数越高链越长。"""
    chain = max(1, -(-info["max_degree"] // degree_cap))
    return info["variables"] * chain * chain


def check(w, S=4, w_stale=0.5):
    built = seq.build(w, S=S, w_stale=w_stale)
    if not built:
        return None
    q = built[0]
    info = describe(q)
    info["physical_estimate"] = embedding_estimate(info)
    info["linear_range"] = [min(v for (i, j), v in q.items() if i == j),
                            max(v for (i, j), v in q.items() if i == j)]
    info["quad_range"] = [min(v for (i, j), v in q.items() if i != j),
                          max(v for (i, j), v in q.items() if i != j)]
    return info


def split_by_tool(w, S=4, w_stale=0.5):
    """按工具拆子问题：每个工具有自己的动作集，互不耦合，适合分别上机。"""
    import backend as bk
    out = {}
    for tool in world.TOOLS:
        pairs = [(pi, si) for pi, si, v, t, stale, ok in bk._pairs(w) if ok and t == tool]
        if not pairs:
            continue
        acts = []
        for pi, si in pairs:
            p, s = w.parts[pi], w.slots[si]
            acts.append((pi, si, p.tool, p.value - w_stale * (1.0 if s.stale else 0.0)))
        A = len(acts)
        q = {}
        def add(i, j, v):
            if i > j:
                i, j = j, i
            q[(i, j)] = q.get((i, j), 0.0) + v
        for a in range(A):
            for st in range(S):
                add(a * S + st, a * S + st, -acts[a][3])
        for a in range(A):
            for st in range(S):
                for tt in range(st + 1, S):
                    add(a * S + st, a * S + tt, 3.0)
        for st in range(S):
            for a in range(A):
                for b in range(a + 1, A):
                    share = acts[a][0] == acts[b][0] or acts[a][1] == acts[b][1]
                    add(a * S + st, b * S + st, 3.0 if share else 6.0)
        for st in range(S - 1):
            for a in range(A):
                for b in range(A):
                    add(a * S + st, b * S + st + 1, -world.W_STREAK)
        out[tool] = q
    return out


def solve(q, reads=100, annealing_time=20.0, label="assembly-qubo"):
    token = os.environ.get("DWAVE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "没有 DWAVE_API_TOKEN。先到 https://cloud.dwavesys.com/leap/ 复制 token，"
            "设置环境变量后再跑；本文所有数字在此之前都只是 CPU 模拟。")
    from dwave.system import DWaveSampler, EmbeddingComposite
    bqm = dimod.BinaryQuadraticModel.from_qubo(q)
    sampler = EmbeddingComposite(DWaveSampler(token=token))
    t0 = time.time()
    ss = sampler.sample(bqm, num_reads=reads, annealing_time=annealing_time, label=label)
    wall = time.time() - t0
    info = ss.info
    return ss, dict(
        wall_s=round(wall, 3),
        qpu_access_us=info.get("qpu_access_time"),
        chain_break_fraction=round(float(info.get("chain_break_fraction", -1)), 4)
        if info.get("chain_break_fraction") is not None else None,
        num_reads=reads,
        annealing_time_us=annealing_time,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split", action="store_true", help="按工具拆子问题，看能不能上机")
    ap.add_argument("--reads", type=int, default=100)
    ap.add_argument("--anneal", type=float, default=20.0)
    ap.add_argument("--part", type=int, default=12)
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    w = world.World(n_part=a.part, n_slot=a.slot, seed=a.seed,
                    p_stale=0.55, q_correct=0.55, obs_cost=0.25)
    info = check(w)
    print("场景 %d 件 / %d 槽，过期率 0.55" % (a.part, a.slot))
    print("变量 %d，耦合 %d，最大度数 %d" % (info["variables"], info["interactions"], info["max_degree"]))
    print("线性项 [%.2f, %.2f]，二次项 [%.2f, %.2f]"
          % (info["linear_range"][0], info["linear_range"][1],
             info["quad_range"][0], info["quad_range"][1]))
    print("按 Pegasus 度数 15 估算，物理比特约 %d" % info["physical_estimate"])
    if info["variables"] > 200 or info["physical_estimate"] > 5000:
        print("结论：这个规模不建议整块上机，先按列或按区拆成 100 变量以内的子问题。")
    else:
        print("结论：规模在可试范围内。")
    if a.split:
        for tool, q in split_by_tool(w).items():
            d = describe(q)
            d["physical_estimate"] = embedding_estimate(d)
            print("按工具 %s 拆开：变量 %d，耦合 %d，最大度数 %d，物理比特约 %d"
                  % (tool, d["variables"], d["interactions"], d["max_degree"], d["physical_estimate"]))
        return
    if a.dry_run:
        return
    q = seq.build(w, S=4, w_stale=0.5)[0]
    ss, rec = solve(q, reads=a.reads, annealing_time=a.anneal)
    print("真机返回：", rec)
    best = ss.first.sample
    print("最优样本能量", round(ss.first.energy, 4))


if __name__ == "__main__":
    main()