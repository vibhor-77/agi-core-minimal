#!/usr/bin/env python3
"""4-pillar wake-sleep loop for ARC-AGI-1 in under 100 lines.
Pillars: (1) Primitives, (2) Composition, (3) Search/Wake, (4) Abstraction/Sleep.
Usage: python solve.py [data_path] [max_tasks]
"""
import json, os, sys
from itertools import product

# ── Pillar 1: Primitives (atomic grid → grid transforms) ─────────────────────
def _crop(g):
    rs = [i for i, r in enumerate(g) if any(r)]
    cs = [j for j in range(len(g[0])) if any(g[i][j] for i in rs)] if rs else []
    return [[g[i][j] for j in cs] for i in rs] if rs and cs else g

PRIMS = {
    "rot90":   lambda g: [list(r) for r in zip(*g[::-1])],
    "rot180":  lambda g: [r[::-1] for r in g[::-1]],
    "flip_h":  lambda g: [r[::-1] for r in g],
    "flip_v":  lambda g: g[::-1],
    "transp":  lambda g: [list(r) for r in zip(*g)],
    "crop":    _crop,
    "tile_h":  lambda g: [r + r for r in g],
    "tile_v":  lambda g: g + g,
    "mir_h":   lambda g: [r + r[::-1] for r in g],
    "mir_v":   lambda g: g + g[::-1],
    "up2":     lambda g: [sum(([c,c] for c in r),[]) for r in g for _ in (0,1)],
    "shrink2": lambda g: [[g[r][c] for c in range(0,len(g[0]),2)]
                           for r in range(0,len(g),2)] if len(g)>1 else g,
}

# ── Pillar 2: Composition (programs = chains of primitive names) ─────────────
def run(chain, g):
    for p in chain: g = PRIMS[p](g)
    return g

def check(chain, pairs):
    try:    return all(run(chain, i) == o for i, o in pairs)
    except: return False

# ── Pillar 3: Search / Wake (enumerate compositions, find solutions) ─────────
def wake(tasks, max_depth, solved):
    found, names = {}, list(PRIMS)
    for tid, (train, test) in tasks.items():
        if tid in solved: continue
        for d in range(1, max_depth + 1):
            hit = next((list(p) for p in product(names, repeat=d)
                        if check(list(p), train + test)), None)
            if hit: found[tid] = hit; break
    return found

# ── Pillar 4: Abstraction / Sleep (extract reusable sub-programs) ────────────
_promoted = set()

def sleep(solved, min_reuse=1):
    subs = {}
    for tid, prog in solved.items():
        for ln in range(2, len(prog) + 1):
            for i in range(len(prog) - ln + 1):
                subs.setdefault(tuple(prog[i:i + ln]), set()).add(tid)
    new = []
    for chain, tids in sorted(subs.items(), key=lambda x: -len(x[1])):
        if len(tids) < min_reuse or chain in _promoted: continue
        _promoted.add(chain)
        name = f"L{len(PRIMS)}"
        PRIMS[name] = (lambda c: lambda g: run(c, g))(list(chain))
        new.append((name, chain, len(tids)))
    return new

# ── Data loading ─────────────────────────────────────────────────────────────
def load(path, n=400):
    tasks = {}
    for f in sorted(os.listdir(path))[:n]:
        if f.endswith(".json"):
            d = json.load(open(os.path.join(path, f)))
            tasks[f[:-5]] = ([(e["input"], e["output"]) for e in d["train"]],
                             [(e["input"], e["output"]) for e in d["test"]])
    return tasks

# ── Compounding loop ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    tasks = load(path, int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    n_base = len(PRIMS)
    print(f"Loaded {len(tasks)} tasks, {n_base} primitives\n")

    solved = {}
    for rd in range(1, 6):
        new = wake(tasks, max_depth=2, solved=solved)
        solved.update(new)
        lib = sleep(solved)
        print(f"Round {rd}: {len(solved)}/{len(tasks)} solved "
              f"(+{len(new)} new), {len(PRIMS) - n_base} learned")
        for name, chain, cnt in lib:
            print(f"  {name} = {' → '.join(chain)} ({cnt} tasks)")
        if not new and not lib:
            print("  converged"); break

    print(f"\nFinal: {len(solved)}/{len(tasks)} solved, "
          f"{len(PRIMS)} primitives ({len(PRIMS) - n_base} learned)")
