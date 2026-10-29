"""诊断：在堆叠场景里，贪心究竟在哪一步、因为什么输给全局解。"""
import sys

import scene as S
import qubo as Q
import solvers as V

M_CL, PER = 6, 4
T = 14
sc = S.make_pile_scene(n_clusters=M_CL, per_cluster=PER, seed=0, budget=T)
res_g = V.greedy(sc, T)
res_e = V.exact_milp(sc, T, time_limit=120.0)
print("n_obj={}  T={}  greedy={:.4f}  milp={:.4f}  gap={:.2f}%".format(
    sc.n_obj, T, res_g.score, res_e.score, 100 * (res_e.score - res_g.score) / res_e.score))
print("levels:", [b for b in sc.levels])
print("greedy seq:", [(o, round(sc.score(o), 3)) for o in res_g.seq])
print("milp   seq:", [(o, round(sc.score(o), 3)) for o in res_e.seq])

# 每个物体被谁挡住
for o in range(sc.n_obj):
    bl = sorted(sc.blocked[o])
    if bl:
        print("obj{:2d} level={} score={:.3f}  blocked_by={}".format(
            o, sc.levels[o], sc.score(o), bl))

# 逐步对比：第一处分歧
a, b = res_g.seq, res_e.seq
k = 0
while k < min(len(a), len(b)) and a[k] == b[k]:
    k += 1
print("first divergence at step", k + 1)
print("  greedy took obj{} (score {:.3f}), milp took obj{} (score {:.3f})".format(
    a[k], sc.score(a[k]), b[k], sc.score(b[k])))
# 分歧之后两边各拿到什么
print("  greedy tail:", [(o, round(sc.score(o), 3)) for o in a[k:]])
print("  milp   tail:", [(o, round(sc.score(o), 3)) for o in b[k:]])
print("  greedy total tail score {:.3f} vs milp {:.3f}".format(
    sum(sc.score(o) for o in a[k:]), sum(sc.score(o) for o in b[k:])))
