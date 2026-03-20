"""
Minimal AGI scaffolding built on 4 pillars:

  1. Feedback Loops       — closed-loop error signal drives the learning cycle
  2. Approximability      — near-misses are valuable; closer is better even if imperfect
  3. Abstraction/Compose  — compose programs from primitives; abstract sub-programs back
  4. Exploration          — population-based evolutionary search over program space

The learning loop: evolve population → fitness selection → crossover/mutate → abstract → repeat.
Compounding emerges because fit programs breed, and successful compositions become
new primitives that widen what evolution can reach next round.
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
    "shift_r":    lambda g: np.roll(np.array(g), 1, axis=1).tolist(),
    "shift_d":    lambda g: np.roll(np.array(g), 1, axis=0).tolist(),
}

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
        fn = primitives.get(name)
        if fn is None:
            return None  # primitive was pruned; program is invalid
        try:
            grid = fn(np.array(grid))
        except Exception:
            return None
        if grid is None:
            return None
    return grid

# ── Pillar 1: Feedback Loops (continuous error signal) ───────────────
def error(program, examples):
    """Score on cells that actually change (input != output). Ignore background."""
    total = 0.0
    for inp, out in examples:
        got = execute(program, inp)
        if got is None:
            continue
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

# ── Pillar 4: Exploration (evolutionary search) ──────────────────────
def random_program(max_depth=4):
    """Seed population with random programs."""
    names = list(primitives)
    depth = np.random.randint(1, max_depth + 1)
    return [names[i] for i in np.random.choice(len(names), size=depth)]

def tournament_select(population, fitnesses, k=3):
    """Pick k random programs, return the fittest."""
    idxs = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]

def crossover(p1, p2, max_len=6):
    """Single-point crossover: prefix of p1 + suffix of p2."""
    c1 = np.random.randint(1, len(p1) + 1)
    c2 = np.random.randint(0, len(p2))
    child = p1[:c1] + p2[c2:]
    return child[:max_len]

def mutate(prog):
    """Insert, delete, or swap one primitive with equal probability."""
    prog = list(prog)
    names = list(primitives)
    r = np.random.random()
    if r < 0.33 and len(prog) < 6:          # insert
        prog.insert(np.random.randint(len(prog) + 1), np.random.choice(names))
    elif r < 0.66 and len(prog) > 1:         # delete
        del prog[np.random.randint(len(prog))]
    else:                                     # swap
        prog[np.random.randint(len(prog))] = np.random.choice(names)
    return prog

def fitness(prog, tasks):
    """Aggregate fitness: sum of (1 - error) across all tasks."""
    total = 0.0
    for tid, task in tasks.items():
        err = error(prog, task["train"])
        total += 1.0 - err
    return total

# ── Pillar 3b: Abstraction (promote solvers as new primitives) ───────
def abstract(population, tasks):
    """Promote programs that solve tasks as new primitives."""
    probe = np.arange(12).reshape(3, 4).tolist()
    new = []
    for prog in population:
        if len(prog) < 2:
            continue
        name = "abs_" + "_".join(prog)
        if name in primitives:
            continue
        solves_any = any(error(prog, task["train"]) == 0.0
                         for task in tasks.values())
        if not solves_any:
            continue
        fn = (lambda p: lambda g: execute(list(p), g))(tuple(prog))
        if np.array_equal(fn(probe), probe):
            continue
        primitives[name] = fn
        new.append(name)
        print(f"  ++ abstracted '{name}'")
    return new

def prune_primitives(population):
    """Remove abstracted primitives unused by top-50% of population."""
    base = {"rotate_cw","rotate_ccw","flip_h","flip_v","transpose","grav_down","shift_r","shift_d"}
    used = set()
    for prog in population:
        used.update(prog)
    to_remove = [n for n in primitives if n not in base and n not in used]
    for n in to_remove:
        del primitives[n]
        print(f"  -- pruned '{n}'")

# ── The Loop (evolutionary) ──────────────────────────────────────────
def evolve(tasks, rounds, pop_size=200):
    """Population-based evolutionary search with behavioral abstraction."""
    population = [random_program() for _ in range(pop_size)]

    for r in range(1, rounds + 1):
        # Evaluate fitness
        fitnesses = [fitness(prog, tasks) for prog in population]

        # Track solved tasks
        solved = set()
        best_per_task = {}
        for prog, fit in zip(population, fitnesses):
            for tid, task in tasks.items():
                err = error(prog, task["train"])
                if err == 0.0:
                    solved.add(tid)
                if tid not in best_per_task or err < best_per_task[tid][1]:
                    best_per_task[tid] = (prog, err)

        for tid in solved:
            prog = best_per_task[tid][0]
            print(f"  ✓ {tid} solved by {prog}")

        # Behavioral abstraction: promote solvers as new primitives
        abstract(population, tasks)

        # Evolve next generation
        ranked = sorted(zip(fitnesses, population), reverse=True)
        elite_n = pop_size // 5
        new_pop = [prog for _, prog in ranked[:elite_n]]
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

        # Prune unused primitives every 3 rounds
        if r % 3 == 0:
            prune_primitives(population)

        approx = sum(1 for _, (_, e) in best_per_task.items() if 0 < e < 1)
        print(f"Round {r}: {len(solved)}/{len(tasks)} solved, "
              f"{approx} near-misses, {len(primitives)} primitives")

# ── Evaluation (frozen primitives, held-out test) ─────────────────────
def evaluate(tasks, generations=5, pop_size=100):
    """Short evolutionary search per eval task with frozen primitives."""
    solved_train, solved_test, total = 0, 0, 0
    for tid, task in tasks.items():
        train_ex, test_ex = task["train"], task["test"]
        if not test_ex:
            continue
        total += 1
        # Mini evolution on this task's train examples
        pop = [random_program() for _ in range(pop_size)]
        best_err, best_prog = 1.0, None
        for _ in range(generations):
            fits = [1.0 - error(p, train_ex) for p in pop]
            for p, f in zip(pop, fits):
                if 1.0 - f < best_err:
                    best_err, best_prog = 1.0 - f, p
            if best_err == 0.0:
                break
            ranked = sorted(zip(fits, pop), reverse=True)
            elite = [p for _, p in ranked[:pop_size // 5]]
            new_pop = list(elite)
            while len(new_pop) < pop_size:
                p1 = tournament_select(pop, fits)
                p2 = tournament_select(pop, fits)
                new_pop.append(mutate(crossover(p1, p2)))
            pop = new_pop
        if best_err == 0.0:
            solved_train += 1
            if error(best_prog, test_ex) == 0.0:
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

    # Phase 1: Train — evolve programs and learn abstractions
    train_tasks = load(train_path)
    print(f"Loaded {len(train_tasks)} training tasks, {len(primitives)} primitives")
    evolve(train_tasks, rounds)

    # Phase 2: Evaluate — frozen primitives on held-out eval set
    if os.path.isdir(eval_path):
        eval_tasks = load(eval_path)
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(eval_tasks)} held-out tasks "
              f"({len(primitives)} learned primitives)")
        print(f"{'='*60}")
        evaluate(eval_tasks)
