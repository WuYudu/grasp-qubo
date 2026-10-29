import scene as sc, solvers
for canvas, ov in [((280.,215.),0.45), ((340.,260.),0.35), ((380.,290.),0.30), ((420.,320.),0.25)]:
    print(f"\n--- canvas={canvas} overlap={ov}")
    for M, T in [(4,3),(6,4),(8,5),(10,7),(12,8),(16,10)]:
        s = sc.make_scene(M, seed=0, canvas=canvas, max_overlap=ov, budget=T)
        mut = sum(1 for i,b in enumerate(s.blockers) if any(i in s.blockers[j] for j in b if j < len(s.blockers)))
        free0 = sum(1 for b in s.blockers if not b)
        m = solvers.exact_milp(s, T, time_limit=30)
        g = solvers.greedy(s, T)
        ub = solvers.relaxed_upper(s, T)
        edges = sum(len(b) for b in s.blockers)
        print(f" M={M:2d} T={T} edges={edges:3d} free0={free0:2d}/{M} milp={m.score:5.2f}({m.time_ms:6.1f}ms) greedy={g.score:5.2f} ub={ub:4.1f}")
