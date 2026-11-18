"""固定前缀 h=3，改观测成本，看风险策略还有没有活路。"""
import csv
import statistics

import backend
import sim
import world

OBS = [0.10, 0.25, 0.60]
HS = [2, 3]
NAMES = ["greedy_safe", "risk", "milp"]


def main():
    rows = []
    for ob in OBS:
        for h in HS:
            for n in NAMES:
                sc, sp, rp = [], [], []
                for s in range(5):
                    w = world.World(n_part=12, n_slot=8, seed=s, p_stale=0.55,
                                   q_correct=0.55, obs_cost=ob)
                    r = sim.episode(w, backend.make(n), h=h, budget_s=18.0, seed=100 + s)
                    sc.append(r["score"])
                    sp.append(r["spoiled"])
                    rp.append(r["replans"])
                row = dict(obs=ob, h=h, name=n, score=round(statistics.mean(sc), 3),
                           spoiled=round(statistics.mean(sp), 2), replans=round(statistics.mean(rp), 1))
                rows.append(row)
                print("obs=%.2f h=%d %-12s score=%6.3f spoiled=%4.2f replans=%5.1f"
                      % (ob, h, n, row["score"], row["spoiled"], row["replans"]))
    with open("../results/calib_obs.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["obs", "h", "name", "score", "spoiled", "replans"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


if __name__ == "__main__":
    main()