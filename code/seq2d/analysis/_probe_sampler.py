"""Probe: sampler behaviour (feasibility rate) on the one-hot encoding."""
import solvers as V
import scene as S
import qubo

if __name__ == "__main__":
    for cl, pc, frac in ((2, 3, 0.6), (3, 3, 0.6), (4, 3, 0.6), (3, 4, 0.6)):
        M = cl * pc
        T = max(1, int(round(frac * M)))
        sc = S.make_pile_scene(n_clusters=cl, per_cluster=pc, seed=0, budget=T)
        pack = qubo.build(sc, T)
        m = V.exact_milp(sc, T, 60.0)
        g = V.greedy(sc, T)
        sa = V.anneal_classical(sc, pack, num_reads=40, num_sweeps=1000)
        sqa = V.anneal_quantum(sc, pack, num_reads=16, num_sweeps=400, replicas=6)
        print("M={} T={} n_var={} nnz={} lam={:.0f} | milp {:.2f} ({:.0f} ms) | greedy {:.2f} | "
              "sa {:.2f} gap {:.1f}% inf {:.0%} ({:.0f} ms) | sqa {:.2f} gap {:.1f}% inf {:.0%} ({:.0f} ms)".format(
                  M, T, pack.n, pack.nnz, pack.lam, m.score, m.time_ms, g.score,
                  sa.score, (m.score - sa.score) / m.score * 100, sa.extra["infeasible_rate"], sa.time_ms,
                  sqa.score, (m.score - sqa.score) / m.score * 100, sqa.extra["infeasible_rate"], sqa.time_ms))
