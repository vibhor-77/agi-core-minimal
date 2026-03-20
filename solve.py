"""
Minimal AGI scaffolding built on 4 pillars:

  1. Feedback Loops       — closed-loop error signal drives the learning cycle
  2. Approximability      — near-misses are valuable; closer is better even if imperfect
  3. Abstraction/Compose  — compose programs from primitives; abstract sub-programs back
  4. Exploration          — search the space of programs by composing primitives

The learning loop: explore → score → keep best (even near-misses) → abstract → repeat.
Compounding emerges because approximations feed abstraction, and abstractions
widen what exploration can reach next round.
"""
import json, sys, os, numpy as np
from pathlib import Path

# ── Domain: ARC-AGI-1 ────────────────────────────────────────────────
def _grav_down(g):
    """Slide non-zero cells to the bottom of each column."""
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        vals = g[:, c][g[:, c] != 0]
        if len(vals) > 0:
            out[-len(vals):, c] = vals
    return out.tolist()

primitives = {
    "rotate_cw":  lambda g: np.rot90(g, k=-1).tolist(),
    "rotate_ccw": lambda g: np.rot90(g, k=1).tolist(),
    "flip_h":     lambda g: np.fliplr(g).tolist(),
    "flip_v":     lambda g: np.flipud(g).tolist(),
    "transpose":  lambda g: np.transpose(g).tolist(),
    "grav_down":  lambda g: _grav_down(np.array(g)),
}

_prim_score = {}  # primitive name -> cumulative usefulness

def load(path):
    """Load ARC tasks: {task_id: {"train": [...], "test": [...]}}."""
    tasks = {}
    for f in sorted(Path(path).glob("*.json")):
        t = json.loads(f.read_text())
        tasks[f.stem] = {
            "train": [(e["input"], e["output"]) for e in t["train"]],
            "test":  [(e["input"], e["output"]) for e in t.get("test", [])],
        }
    return tasks

# ── Pillar 3a: Composition (programs = chains of primitives) ─────────
def execute(program, grid):
    """Run a program (list of primitive names) on a grid by chaining."""
    for name in program:
        grid = primitives[name](np.array(grid))
    return grid

# ── Pillar 1: Feedback Loops (continuous error signal) ───────────────
def error(program, examples):
    """Score on cells that actually change (input != output). Ignore background."""
    total = 0.0
    for inp, out in examples:
        got = execute(program, inp)
        inp_a, out_a, got_a = np.array(inp), np.array(out), np.array(got)
        if out_a.shape != got_a.shape:
            continue
        changed = inp_a != out_a
        n_changed = changed.sum()
        if n_changed == 0:
            total += float(np.array_equal(out_a, got_a))
        else:
            total += np.mean(got_a[changed] == out_a[changed])
    return 1.0 - total / len(examples)

# ── Pillar 4: Exploration (search by composing primitives) ───────────
def candidates(max_depth, budget=200):
    """Sample programs weighted by primitive usefulness (explore/exploit)."""
    names = list(primitives)
    scores = np.array([_prim_score.get(n, 1.0) for n in names])
    weights = scores / scores.sum()
    for depth in range(1, max_depth + 1):
        for _ in range(budget):
            idxs = np.random.choice(len(names), size=depth, p=weights)
            yield [names[i] for i in idxs]

# ── Pillar 3b: Abstraction (promote sub-programs as new primitives) ──
def abstract(scored_programs):
    """Promote recurring length-2 sub-programs, weighted by quality.
    Quality = 1 - error, so near-misses (Pillar 2: Approximability)
    contribute proportionally — good approximations compound."""
    weights = {}
    for prog, quality in scored_programs:
        for i in range(len(prog) - 1):
            pair = tuple(prog[i : i + 2])
            weights[pair] = weights.get(pair, 0.0) + quality
    new = []
    probe = np.arange(12).reshape(3, 4).tolist()  # asymmetric grid
    for pair, weight in weights.items():
        name = "_".join(pair)
        if weight >= 1.5 and name not in primitives:
            fn = (lambda p: lambda g: execute(list(p), g))(pair)
            if np.array_equal(fn(probe), probe):
                continue  # skip identity-equivalent compositions
            primitives[name] = fn
            new.append(name)
            print(f"  ++ promoted '{name}' (weight {weight:.2f})")
    return new

# ── The Loop (all 4 pillars working together) ────────────────────────
def learn(tasks, rounds):
    """The compounding cycle: explore → feedback → abstract → repeat.
    Pillar 2 (Approximability) lives in the loop: we keep the best program
    per task regardless of whether it solved, so near-misses feed abstraction.
    Cross-round extension: each round tries extending the previous best program
    by prepending/appending one primitive — this is the compounding engine."""
    best_so_far = {}  # tid -> (prog, err) — persists across rounds
    for r in range(1, rounds + 1):
        solved, best_programs = 0, []
        for tid, task in tasks.items():
            examples = task["train"]
            best_err = best_so_far[tid][1] if tid in best_so_far else 1.0
            best_prog = best_so_far[tid][0] if tid in best_so_far else None
            # Pillar 4: Exploration — try all compositions
            for prog in candidates(max_depth=2):
                # Pillar 1: Feedback — score each candidate
                err = error(prog, examples)
                if err < best_err:
                    best_err, best_prog = err, prog
                    if err == 0.0:
                        solved += 1
                        print(f"  ✓ {tid} solved by {prog}")
                        break
            # Extension: try prepending/appending one primitive to previous best
            if best_err > 0.0 and tid in best_so_far:
                prev = best_so_far[tid][0]
                for name in primitives:
                    for ext in ([name] + prev, prev + [name]):
                        err = error(ext, examples)
                        if err < best_err:
                            best_err, best_prog = err, ext
                            if err == 0.0:
                                solved += 1
                                print(f"  ✓ {tid} solved by {ext} (extended)")
                                break
                    if best_err == 0.0:
                        break
            # Pillar 2: Approximability — keep best even if imperfect
            if best_prog:
                best_so_far[tid] = (best_prog, best_err)
                best_programs.append((best_prog, 1.0 - best_err))
        # Update primitive scores based on usefulness
        for prog, quality in best_programs:
            for name in prog:
                _prim_score[name] = _prim_score.get(name, 1.0) + quality
        # Pillar 3: Abstraction — promote recurring sub-programs
        new = abstract(best_programs)
        approx = sum(1 for _, q in best_programs if 0 < q < 1)
        print(f"Round {r}: {solved}/{len(tasks)} solved, {approx} near-misses, "
              f"{len(primitives)} primitives (+{len(new)} new)")

# ── Evaluation (frozen primitives, held-out test) ─────────────────────
def evaluate(tasks):
    """Find best program per task using train examples, score on test examples.
    Primitives are frozen — no learning, no abstraction."""
    solved_train, solved_test, total = 0, 0, 0
    for tid, task in tasks.items():
        train_ex, test_ex = task["train"], task["test"]
        if not test_ex:
            continue
        total += 1
        # Find best program using train examples
        best_err, best_prog = 1.0, None
        for prog in candidates(max_depth=2, budget=200):
            err = error(prog, train_ex)
            if err < best_err:
                best_err, best_prog = err, prog
                if err == 0.0:
                    break
        # Also try extending depth-1 programs to depth-3
        if best_err > 0.0:
            for prog in candidates(max_depth=1, budget=50):
                for name in primitives:
                    for ext in ([name] + prog, prog + [name]):
                        err = error(ext, train_ex)
                        if err < best_err:
                            best_err, best_prog = err, ext
                            if err == 0.0:
                                break
                    if best_err == 0.0:
                        break
                if best_err == 0.0:
                    break
        if best_err == 0.0:
            solved_train += 1
            # Test on held-out examples
            test_err = error(best_prog, test_ex)
            if test_err == 0.0:
                solved_test += 1
                print(f"  ✓ {tid} eval pass: {best_prog}")
            else:
                print(f"  ~ {tid} train pass, eval fail: {best_prog}")
    print(f"Eval: {solved_test}/{total} test solved "
          f"({solved_train} solved on train examples)")
    return solved_test

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "data/ARC-AGI/data/evaluation"
    rounds     = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if not os.path.isdir(train_path):
        sys.exit(f"Training data not found: {train_path}")

    # Phase 1: Train — learn primitives and abstractions
    train_tasks = load(train_path)
    print(f"Loaded {len(train_tasks)} training tasks, {len(primitives)} primitives")
    learn(train_tasks, rounds)

    # Phase 2: Evaluate — frozen primitives on held-out eval set
    if os.path.isdir(eval_path):
        eval_tasks = load(eval_path)
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(eval_tasks)} held-out tasks "
              f"({len(primitives)} learned primitives)")
        print(f"{'='*60}")
        evaluate(eval_tasks)
