"""
Minimal AGI scaffolding built on 5 pillars:

  Composition      — programs are sequences of primitives; execute by chaining
  Feedback         — continuous error signal: how close is a program to solving a task?
  Exploration      — search the space of programs by composing primitives
  Approximability  — near-miss programs are valuable; closer is better even if imperfect
  Abstraction      — extract recurring sub-programs as new primitives, weighted by quality

The learning loop ties them together: explore → feedback → abstract → repeat.
Even unsuccessful attempts compound — a near-miss promoted as a primitive
becomes a stepping stone for the next round.
"""
import json, sys, os, itertools, numpy as np
from pathlib import Path

# ── Domain: ARC-AGI-1 ────────────────────────────────────────────────
primitives = {
    "rotate_cw":  lambda g: np.rot90(g, k=-1).tolist(),
    "rotate_ccw": lambda g: np.rot90(g, k=1).tolist(),
    "flip_h":     lambda g: np.fliplr(g).tolist(),
    "flip_v":     lambda g: np.flipud(g).tolist(),
    "transpose":  lambda g: np.transpose(g).tolist(),
}

def load(path):
    """Load ARC tasks: {task_id: [(input, output), ...]}."""
    tasks = {}
    for f in sorted(Path(path).glob("*.json")):
        t = json.loads(f.read_text())
        tasks[f.stem] = [(e["input"], e["output"]) for e in t["train"]]
    return tasks

# ── Composition ──────────────────────────────────────────────────────
def execute(program, grid):
    """Run a program (list of primitive names) on a grid by chaining."""
    for name in program:
        grid = primitives[name](np.array(grid))
    return grid

# ── Feedback ─────────────────────────────────────────────────────────
def error(program, examples):
    """Score: 0.0 = solves all examples, 1.0 = total failure."""
    wrong = sum(1 for inp, out in examples if execute(program, inp) != out)
    return wrong / len(examples)

# ── Exploration ──────────────────────────────────────────────────────
def candidates(max_depth):
    """Yield all programs up to max_depth primitives."""
    names = list(primitives)
    for depth in range(1, max_depth + 1):
        for combo in itertools.product(names, repeat=depth):
            yield list(combo)

# ── Abstraction ──────────────────────────────────────────────────────
def abstract(scored_programs):
    """Promote recurring length-2 sub-programs, weighted by quality (1 - error).
    Near-misses contribute proportionally, so good approximations compound."""
    weights = {}
    for prog, quality in scored_programs:
        for i in range(len(prog) - 1):
            pair = tuple(prog[i : i + 2])
            weights[pair] = weights.get(pair, 0.0) + quality
    new = []
    for pair, weight in weights.items():
        if weight >= 1.5 and pair not in primitives:
            name = "_".join(pair)
            primitives[name] = (lambda p: lambda g: execute(list(p), g))(pair)
            new.append(name)
    return new

# ── The Loop ─────────────────────────────────────────────────────────
def learn(tasks, rounds):
    """Wake-sleep compounding: explore → feedback → abstract → repeat."""
    for r in range(1, rounds + 1):
        solved, best_programs = 0, []
        for tid, examples in tasks.items():
            best_err, best_prog = 1.0, None
            for prog in candidates(max_depth=2):
                err = error(prog, examples)
                if err < best_err:
                    best_err, best_prog = err, prog
                    if err == 0.0:
                        solved += 1
                        break
            if best_prog:
                best_programs.append((best_prog, 1.0 - best_err))
        new = abstract(best_programs)
        approx = sum(1 for _, q in best_programs if 0 < q < 1)
        print(f"Round {r}: {solved}/{len(tasks)} solved, {approx} near-misses, "
              f"{len(primitives)} primitives (+{len(new)} new)")

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if not os.path.isdir(path):
        sys.exit(f"Data not found: {path}")
    tasks = load(path)
    print(f"Loaded {len(tasks)} tasks, {len(primitives)} primitives")
    learn(tasks, rounds)
