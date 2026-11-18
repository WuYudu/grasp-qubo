import io
p = r"C:\Users\Administrator\Documents\deepseek-harness\default-workspace\qc-embodied-research\assembly_qubo\src\world.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace(
    "    def __init__(self, n_part=12, n_slot=8, seed=0, p_stale=0.35):",
    "    def __init__(self, n_part=12, n_slot=8, seed=0, p_stale=0.35, q_correct=0.55):")
old = """            stale = rng.random() < p_stale
            belief = kind if not stale else rng.choice([k for k in KINDS if k != kind])"""
new = """            stale = rng.random() < p_stale
            if not stale or rng.random() < q_correct:
                belief = kind
            else:
                belief = rng.choice([k for k in KINDS if k != kind])"""
assert old in s
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8").write(s)
print("world patched")