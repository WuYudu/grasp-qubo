import time
from kits import World, run

LADDER = [(20, 200), (40, 400), (80, 800), (160, 1200), (320, 2000), (640, 3000)]
print("n_col  reads  sweeps  rate  score  sec", flush=True)
for n_col in (4, 5):
    for reads, sweeps in LADDER:
        t0 = time.time()
        rows = [run(World(n_col=n_col, per_col=3, seed=s), "sa", T=5, horizon=2,
                    seed=s, reads=reads, sweeps=sweeps) for s in range(3)]
        sec = (time.time() - t0) / 3.0
        pairs = sum(r["pairs"] for r in rows)
        done = sum(r["done"] for r in rows)
        print(f"{n_col:5d} {reads:6d} {sweeps:7d} {done/max(1,pairs):5.0%} "
              f"{sum(r['score'] for r in rows)/3:6.2f} {sec:5.2f}", flush=True)
