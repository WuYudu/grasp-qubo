"""Probe: piled scenes - does the ordering decision bite?"""
import solvers as V
import scene as S

if __name__ == "__main__":
    for cl, pc in ((2, 4), (3, 3), (4, 3), (3, 4), (5, 3)):
        M = cl * pc
        for frac in (0.4, 0.6, 0.8):
            T = max(1, int(round(frac * M)))
            gaps = []
            scores = []
            for seed in range(6):
                sc = S.make_pile_scene(n_clusters=cl, per_cluster=pc, seed=seed, budget=T)
                m = V.exact_milp(sc, T, 60.0)
                g = V.greedy(sc, T)
                gaps.append((m.score - g.score) / m.score * 100 if m.score > 1e-9 else 0.0)
                scores.append(m.score)
            print("clusters={} per={} M={} T={} milp_mean={:.2f} gaps={}".format(
                cl, pc, M, T, sum(scores) / len(scores),
                " ".join("{:.0f}%".format(x) for x in gaps)))
