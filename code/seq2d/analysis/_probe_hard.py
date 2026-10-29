"""Probe: where does the peeling planner stop being trivial?"""
import solvers as V
import scene as S


def depth_of(sc, T):
    """Longest chain in the feasible plan produced by the optimal solution."""
    res = V.exact_milp(sc, T, 30.0)
    return res


if __name__ == "__main__":
    for M in (8, 10, 12, 16):
        for ov in (0.35, 0.45, 0.55):
            line = []
            for seed in range(4):
                sc = S.make_scene(n_obj=M, seed=seed, max_overlap=ov)
                T = sc.budget
                m = V.exact_milp(sc, T, 30.0)
                g = V.greedy(sc, T)
                free = sum(1 for o in range(M) if not sc.blocked[o])
                gap = (m.score - g.score) / m.score * 100 if m.score > 1e-9 else 0.0
                line.append("s{}:free{}|m{:.2f}|g{:.2f}|{:.0f}%|ms{:.0f}".format(
                    seed, free, m.score, g.score, gap, m.time_ms))
            print("M={} ov={} T={}  ".format(M, ov, sc.budget), "  ".join(line))
