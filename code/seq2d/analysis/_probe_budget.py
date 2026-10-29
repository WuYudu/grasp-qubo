"""Probe: budget sweep - when does the greedy peel start losing?"""
import solvers as V
import scene as S

if __name__ == "__main__":
    for M in (10, 14, 18):
        for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
            T = max(1, int(round(frac * M)))
            gaps = []
            for seed in range(6):
                sc = S.make_scene(n_obj=M, seed=seed, max_overlap=0.45, budget=T)
                m = V.exact_milp(sc, T, 30.0)
                g = V.greedy(sc, T)
                gap = (m.score - g.score) / m.score * 100 if m.score > 1e-9 else 0.0
                gaps.append(gap)
            print("M={} T={} (frac={}) gaps={}".format(
                M, T, frac, " ".join("{:.0f}%".format(x) for x in gaps)))
