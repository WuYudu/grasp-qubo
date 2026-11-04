import sys

from kits import World, run

PLANNERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["greedy", "greedy_pair", "sa"]
H = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2"])]
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

print("planner       cols  N  h  pairs  done  rate  score  kill  act", flush=True)
for p in PLANNERS:
    for n_col in (3, 4, 5):
        for h in H:
            rows = [run(World(n_col=n_col, per_col=3, seed=s), p, T=5, horizon=h, seed=s)
                    for s in range(SEEDS)]
            k = len(rows)
            pairs = sum(r["pairs"] for r in rows) / k
            done = sum(r["done"] for r in rows) / k
            print(f"{p:12s} {n_col:4d} {n_col*2:3d} {h:2d} {pairs:6.1f} {done:5.1f} "
                  f"{done/max(1e-9, pairs):5.0%} {sum(r['score'] for r in rows)/k:6.2f} "
                  f"{sum(r['killed'] for r in rows)/k:5.1f} "
                  f"{sum(r['actions'] for r in rows)/k:4.1f}", flush=True)
