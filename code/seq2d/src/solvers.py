"""Reference solvers for the ordered grasping QUBO."""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass, field

import numpy as np

import qubo
from scene import Scene


@dataclass
class Result:
    solver: str
    score: float
    feasible: bool
    seq: list[int]
    time_ms: float
    extra: dict = field(default_factory=dict)


def relaxed_upper(scene: Scene, T: int) -> float:
    scores = sorted((scene.score(o) for o in range(scene.n_obj)), reverse=True)
    return float(sum(scores[:T]))


def exact_milp(scene: Scene, T: int | None = None, time_limit: float = 180.0) -> Result:
    import pulp

    T = scene.budget if T is None else T
    S = T + 1
    t0 = time.perf_counter()
    prob = pulp.LpProblem("seq_grasp", pulp.LpMaximize)
    z = {}
    for o in range(scene.n_obj):
        for s in range(S):
            z[o, s] = prob.add_variable(f"z_{o}_{s}", cat=pulp.LpBinary)
    prob += pulp.lpSum(scene.score(o) * z[o, s] for o in range(scene.n_obj) for s in range(T))
    for o in range(scene.n_obj):
        prob += pulp.lpSum(z[o, s] for s in range(S)) == 1
    for s in range(T):
        prob += pulp.lpSum(z[o, s] for o in range(scene.n_obj)) <= 1
    for o in range(scene.n_obj):
        for blocker in scene.blocked[o]:
            for q in range(T):
                prob += z[o, q] + pulp.lpSum(z[blocker, p] for p in range(q, S)) <= 1
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    ms = (time.perf_counter() - t0) * 1e3
    ok = stats is not None and getattr(stats, "has_solution", False)
    seq = []
    if ok:
        for s in range(T):
            for o in range(scene.n_obj):
                if pulp.value(z[o, s]) and pulp.value(z[o, s]) > 0.5:
                    seq.append(o)
    out = qubo.evaluate(scene, seq, T)
    return Result("milp", float(pulp.value(prob.objective)) if ok else 0.0, out["feasible"],
                  seq, ms, {"status": str(getattr(stats, "status", "")), "reason": out["reason"]})


def greedy(scene: Scene, T: int | None = None, unblock_weight: float = 0.0) -> Result:
    T = scene.budget if T is None else T
    t0 = time.perf_counter()
    remaining = set(range(scene.n_obj))
    seq: list[int] = []
    score = 0.0
    while remaining and len(seq) < T:
        best, best_val = None, -1e9
        for o in remaining:
            if scene.blocked[o] & remaining:
                continue
            val = scene.score(o)
            if unblock_weight:
                freed = sum(1 for p in remaining
                            if p != o and o in scene.blocked[p]
                            and not (scene.blocked[p] & (remaining - {o})))
                val += unblock_weight * freed
            if val > best_val:
                best, best_val = o, val
        if best is None:
            break
        seq.append(best)
        score += scene.score(best)
        remaining.discard(best)
    ms = (time.perf_counter() - t0) * 1e3
    out = qubo.evaluate(scene, seq, T)
    return Result("greedy", out["score"], out["feasible"], seq, ms, {"reason": out["reason"]})


def _to_bqm(pack: qubo.Pack):
    import dimod

    Q = {}
    for (a, b), v in pack.Q.items():
        Q[(a, b)] = v
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=pack.offset)


def _pick(scene: Scene, pack: qubo.Pack, sampleset, name: str, ms: float) -> Result:
    best = None
    n_inf = 0
    reads = len(sampleset)
    for sample in sampleset.samples():
        x = np.zeros(pack.n, dtype=np.int8)
        for k, v in sample.items():
            x[k] = 1 if v else 0
        out = qubo.decode(scene, x, pack.T)
        if not out["feasible"]:
            n_inf += 1
            continue
        if best is None or out["score"] > best["score"]:
            best = out
    if best is None:
        return Result(name, 0.0, False, [], ms, {"reads": reads, "infeasible_reads": reads,
                                                 "infeasible_rate": 1.0})
    return Result(name, best["score"], True, best["seq"], ms,
                  {"reads": reads, "infeasible_reads": n_inf, "infeasible_rate": n_inf / max(1, reads)})


def anneal_classical(scene: Scene, pack: qubo.Pack, num_reads: int = 40,
                     num_sweeps: int = 1000, seed: int = 0) -> Result:
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = _to_bqm(pack)
    t0 = time.perf_counter()
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    ms = (time.perf_counter() - t0) * 1e3
    return _pick(scene, pack, ss, "sa", ms)


def anneal_quantum(scene: Scene, pack: qubo.Pack, num_reads: int = 16,
                   num_sweeps: int = 400, replicas: int = 6, seed: int = 0) -> Result:
    from dwave.samplers import PathIntegralAnnealingSampler

    bqm = _to_bqm(pack)
    t0 = time.perf_counter()
    ss = PathIntegralAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps,
                                              replicas=replicas, seed=seed)
    ms = (time.perf_counter() - t0) * 1e3
    return _pick(scene, pack, ss, "sqa", ms)


def anneal_hardware(scene: Scene, pack: qubo.Pack, num_reads: int = 200,
                    annealing_time: float = 20.0) -> Result:
    import os

    if not os.environ.get("DWAVE_API_TOKEN"):
        raise RuntimeError(
            "DWAVE_API_TOKEN not set. Register at https://cloud.dwavesys.com/leap/, "
            "copy the token, then run  $env:DWAVE_API_TOKEN='...'  and retry with --hw "
            "(needs  pip install dwave-ocean-sdk)."
        )
    from dwave.system import DWaveSampler, EmbeddingComposite

    bqm = _to_bqm(pack)
    t0 = time.perf_counter()
    ss = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=num_reads,
                                                   annealing_time=annealing_time, label="seq2d")
    ms = (time.perf_counter() - t0) * 1e3
    res = _pick(scene, pack, ss, "qpu", ms)
    info = ss.info
    res.extra.update({
        "qpu_access_us": info.get("timing", {}).get("qpu_access_time"),
        "chain_break_fraction": float(np.mean(ss.record.chain_break_fraction)) if "chain_break_fraction" in ss.record else None,
        "physical_qubits": info.get("embedding_context", {}).get("embedding") and
        sum(len(v) for v in info["embedding_context"]["embedding"].values()),
    })
    return res
