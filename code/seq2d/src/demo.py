"""Tiny end-to-end demo. numpy only, no solver deps:  python src/demo.py"""
import numpy as np

import qubo
import scene as S
import solvers as V

N_CLUSTERS, PER_CLUSTER, T, SEED = 3, 3, 5, 0


def main() -> None:
    sc = S.make_pile_scene(n_clusters=N_CLUSTERS, per_cluster=PER_CLUSTER,
                           seed=SEED, budget=T)
    pack = qubo.build(sc, T)

    print("objects   : {} = {} piles x {}".format(sc.n_obj, N_CLUSTERS, PER_CLUSTER))
    print("variables : M x (T+1) = {} x {} = {}".format(sc.n_obj, T + 1, pack.n))
    print("Q entries : {}, lam = {:.1f}, offset = {:.1f}".format(
        pack.nnz, pack.lam, pack.offset))

    res = V.greedy(sc, T)
    print("\ngreedy    : score {:.4f}, feasible {}, {:.0f} us".format(
        res.score, res.feasible, res.time_ms * 1e3))
    for step, o in enumerate(res.seq, 1):
        print("  {}. obj{:>2}  s={:.3f}  blocked by {}".format(
            step, o, sc.score(o), sorted(sc.blocked[o])))

    sz = T + 1
    x = np.zeros(pack.n, dtype=np.int64)
    for step, o in enumerate(res.seq):
        x[qubo.vid(o, step, sz)] = 1
    for o in range(sc.n_obj):
        if o not in res.seq:
            x[qubo.vid(o, T, sz)] = 1

    e = pack.energy(x)
    print("\nE(x) = {:.6f}   -score = {:.6f}   diff = {:.1e}".format(
        e, -res.score, abs(e + res.score)))
    print("decode    :", qubo.decode(sc, x, T))


if __name__ == "__main__":
    main()
