"""把 6.46% 的差距拆到"到底选了哪 14 个"上。"""
import scene as S
import solvers as V

T = 14
sc = S.make_pile_scene(n_clusters=6, per_cluster=4, seed=0, budget=T)
g = V.greedy(sc, T)
m = V.exact_milp(sc, T, 120.0)
gs, ms = set(g.seq), set(m.seq)

print("全部 24 个物体的分数（优化前就固定，谁也没改）：")
for o in range(sc.n_obj):
    print("  obj{:2d} lv={} score={:.3f}".format(o, sc.levels[o], sc.score(o)))

print("\n贪心 {} 个: 总分 {:.4f}  平均 {:.4f}".format(len(gs), g.score, g.score / len(gs)))
print("全局 {} 个: 总分 {:.4f}  平均 {:.4f}".format(len(ms), m.score, m.score / len(ms)))
print("共有:", sorted(gs & ms), "小计 {:.4f}".format(sum(sc.score(o) for o in gs & ms)))
only_g = sorted(gs - ms)
only_m = sorted(ms - gs)
print("只有贪心选的:", [(o, round(sc.score(o), 3)) for o in only_g],
      "小计 {:.4f}".format(sum(sc.score(o) for o in only_g)))
print("只有全局选的:", [(o, round(sc.score(o), 3)) for o in only_m],
      "小计 {:.4f}".format(sum(sc.score(o) for o in only_m)))
print("差 = {:.4f}".format(sum(sc.score(o) for o in only_m) - sum(sc.score(o) for o in only_g)))
