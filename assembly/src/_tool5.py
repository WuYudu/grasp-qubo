"""换工具变贵之后，顺序是否第一次咬合。"""
import statistics
import world
import backend
import sim
import seq

world.T_TOOL = 5.0

NAMES = ["greedy_safe", "greedy_tool", "risk", "seq"]
for h in (1, 2, 3):
    out = []
    for n in NAMES:
        sc, sw, ck = [], [], []
        for s in range(5):
            w = world.World(n_part=12, n_slot=8, seed=s, p_stale=0.55,
                            q_correct=0.55, obs_cost=0.25)
            b = seq.SeqAnneal() if n == "seq" else backend.make(n)
            r = sim.episode(w, b, h=h, budget_s=12.0, seed=100 + s)
            sc.append(r["score"]); sw.append(r.get("switches", 0)); ck.append(r["clock"])
        out.append("%-12s %5.2f" % (n, statistics.mean(sc)))
    print("h=%d  " % h + " | ".join(out))