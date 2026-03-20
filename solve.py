"""
Evolutionary program synthesis for ARC-AGI.

Uses genetic programming to evolve cell-level programs from atomic primitives.
Programs compute f(grid, r, c) → color for each output cell. Evolution discovers
compositions like flip, rotate, and recolor. Useful subtrees are abstracted into
a reusable library, enabling compounding across tasks.
"""

import json, sys, os, numpy as np
from pathlib import Path
import gp

# ── Data ─────────────────────────────────────────────────────────────

def load(path):
    """Load ARC tasks from a directory of JSON files."""
    tasks = {}
    for f in sorted(Path(path).glob("*.json")):
        t = json.loads(f.read_text())
        tasks[f.stem] = {
            "train": [(e["input"], e["output"]) for e in t["train"]],
            "test":  [(e["input"], e["output"]) for e in t.get("test", [])],
        }
    return tasks


def output_shape(inp, out):
    """Get the output shape for a single example."""
    return np.asarray(out).shape


def predict_test_shape(train_examples):
    """Infer the input→output shape rule from training examples."""
    rules = []
    for inp, out in train_examples:
        si, so = np.asarray(inp).shape, np.asarray(out).shape
        if si == so:
            rules.append("same")
        elif (si[1], si[0]) == so:
            rules.append("transpose")
        else:
            rules.append(so)  # fixed shape
    if len(set(rules)) == 1:
        return rules[0]
    return "same"  # default fallback


def test_output_shape(rule, inp):
    """Apply a shape rule to predict output shape for a test input."""
    si = np.asarray(inp).shape
    if rule == "same":
        return si
    if rule == "transpose":
        return (si[1], si[0])
    return rule  # fixed shape tuple


# ── Scoring ──────────────────────────────────────────────────────────

IDENTITY = ("get", ("r",), ("c",))


def fitness(tree, examples, library=None):
    """Weighted cell accuracy: changed cells worth 3x, with parsimony pressure."""
    total = 0.0
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library)
        if got is None or got.shape != out_a.shape:
            continue
        inp_a = np.asarray(inp)
        # Handle different-size tasks: only compare output cells
        if inp_a.shape == out_a.shape:
            changed = inp_a != out_a
        else:
            changed = np.ones(out_a.shape, dtype=bool)  # all cells "changed"
        n_ch, n_un = int(changed.sum()), int((~changed).sum())
        score = 0.0
        if n_ch > 0:
            score += 3.0 * np.sum(got[changed] == out_a[changed])
        if n_un > 0:
            score += np.sum(got[~changed] == out_a[~changed])
        max_score = 3.0 * n_ch + n_un
        total += score / max_score if max_score > 0 else 0.0
    accuracy = total / len(examples)
    return max(0.0, accuracy - 0.002 * gp.size(tree))


def solves(tree, examples, library=None):
    """Check if tree produces exact output for all examples."""
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library)
        if got is None or not np.array_equal(got, out_a):
            return False
    return True


def simplify(tree, examples, library=None):
    """Greedily replace subtrees with simpler equivalents."""
    if not solves(tree, examples, library):
        return tree
    R, C = ("r",), ("c",)
    replacements = [
        R, C, ("max_r",), ("max_c",), IDENTITY, ("get", C, R),
        ("sub", ("sub", ("max_r",), ("const", 1)), R),
        ("sub", ("sub", ("max_c",), ("const", 1)), C),
    ] + [("const", i) for i in range(10)]
    changed = True
    while changed:
        changed = False
        subs = gp.subtrees(tree)
        subs.sort(key=lambda ps: -gp.size(ps[1]))
        for path, sub in subs:
            if not path or sub[0] not in gp.ARITY:
                continue
            for repl in replacements:
                if gp.size(repl) >= gp.size(sub):
                    continue
                candidate = gp.replace(tree, path, repl)
                if solves(candidate, examples, library):
                    tree = candidate
                    changed = True
                    break
            if changed:
                break
    return tree


# ── Seeds ────────────────────────────────────────────────────────────

def make_seeds():
    """Generate initial program templates from the atomic primitives."""
    R, C, MR, MC, ONE = ("r",), ("c",), ("max_r",), ("max_c",), ("const", 1)
    flip_r = ("sub", ("sub", MR, ONE), R)   # max_r - 1 - r
    flip_c = ("sub", ("sub", MC, ONE), C)   # max_c - 1 - c
    seeds = [
        IDENTITY,                                           # identity
        ("get", C, R),                                      # transpose
        ("get", flip_r, C),                                 # flip vertical
        ("get", R, flip_c),                                 # flip horizontal
        ("get", flip_r, flip_c),                            # rotate 180
        ("get", C, flip_r),                                 # rotate cw
        ("get", flip_c, R),                                 # rotate ccw
    ]
    # Shifts by 1–4
    for k in range(1, 5):
        K = ("const", k)
        seeds.append(("get", ("mod", ("add", R, K), MR), C))   # shift down
        seeds.append(("get", R, ("mod", ("add", C, K), MC)))   # shift right
    # Recolors: if cell == x, change to y
    for x in range(10):
        for y in range(10):
            if x != y:
                seeds.append(("if", ("eq", ("get", R, C), ("const", x)),
                              ("const", y), ("get", R, C)))
    return seeds


# ── Search ───────────────────────────────────────────────────────────

def search(examples, seed=None, library=None, pop_size=80, gens=20):
    """Evolve a program for a single task.

    Starts from seed templates, evolves via tournament selection + crossover/mutation.
    Returns (best_tree, best_fitness).
    """
    seeds = make_seeds()
    pop_size = max(pop_size, len(seeds) + 20)
    pop = list(seeds)

    if seed is not None and seed != IDENTITY:
        pop.append(seed)
        for _ in range(pop_size // 6):
            pop.append(gp.mutate(seed, library=library))
    for _ in range(pop_size // 6):
        pop.append(gp.mutate(IDENTITY, library=library))
    while len(pop) < pop_size:
        pop.append(gp.random_tree(max_depth=4, library=library))

    best_t, best_f = IDENTITY, fitness(IDENTITY, examples, library)
    if seed is not None:
        sf = fitness(seed, examples, library)
        if sf > best_f:
            best_t, best_f = seed, sf

    stale = 0
    for _ in range(gens):
        fits = [fitness(p, examples, library) for p in pop]
        for p, f in zip(pop, fits):
            if f > best_f:
                best_t, best_f, stale = p, f, 0
        if solves(best_t, examples, library):
            return simplify(best_t, examples, library), 1.0
        stale += 1
        if stale > 7:
            break
        # Next generation
        ranked = sorted(zip(fits, pop), reverse=True)
        elite = [p for _, p in ranked[:pop_size // 5]]
        new_pop = list(elite) + [IDENTITY, best_t]
        for _ in range(pop_size // 10):
            new_pop.append(gp.mutate(best_t, library=library))
        while len(new_pop) < pop_size:
            p1 = gp.tournament(pop, fits)
            p2 = gp.tournament(pop, fits)
            child = gp.crossover(p1, p2)
            if np.random.random() < 0.3:
                child = gp.mutate(child, library=library)
            new_pop.append(child)
        pop = new_pop

    # Neighborhood refinement: try all single-node edits
    if best_f > 0.7 and not solves(best_t, examples, library):
        best_t, best_f = refine(best_t, examples, library)
    return best_t, best_f


def refine(tree, examples, library=None):
    """Try all single-node edits of tree. Returns (best_tree, best_fitness)."""
    best_t, best_f = tree, fitness(tree, examples, library)
    terminals = [("r",), ("c",), ("max_r",), ("max_c",)]
    terminals += [("const", i) for i in range(10)]
    for name in (library or {}):
        terminals.append(("lib", name))
    binary_ops = [k for k, v in gp.ARITY.items() if v == 2]

    for path, node in gp.subtrees(tree):
        op = node[0]
        if op not in gp.ARITY:
            for term in terminals:
                if term == node:
                    continue
                c = gp.replace(tree, path, term)
                if solves(c, examples, library):
                    return simplify(c, examples, library), 1.0
                f = fitness(c, examples, library)
                if f > best_f:
                    best_t, best_f = c, f
        elif gp.ARITY[op] == 2:
            for new_op in binary_ops:
                if new_op == op:
                    continue
                c = gp.replace(tree, path, (new_op, node[1], node[2]))
                if solves(c, examples, library):
                    return simplify(c, examples, library), 1.0
                f = fitness(c, examples, library)
                if f > best_f:
                    best_t, best_f = c, f
    return best_t, best_f


# ── Abstraction ──────────────────────────────────────────────────────

def abstract(solvers, library):
    """Find subtrees shared by 2+ solvers and add to library."""
    if len(solvers) < 2:
        return
    sub_sources = {}
    for i, tree in enumerate(solvers):
        seen = set()
        for _, sub in gp.subtrees(tree):
            sz = gp.size(sub)
            if sz < 3 or sz > 20 or sub in seen:
                continue
            seen.add(sub)
            sub_sources.setdefault(sub, set()).add(i)
    existing = set(library.values())
    for sub, sources in sub_sources.items():
        if len(sources) >= 2 and sub not in existing:
            name = f"L{len(library)}"
            library[name] = sub
            existing.add(sub)
            print(f"  ++ lib '{name}' = {gp.to_str(sub)} "
                  f"(size={gp.size(sub)}, shared by {len(sources)})")


# ── Evolution loop ───────────────────────────────────────────────────

def evolve(tasks, rounds, library):
    """Per-task evolution with cross-task abstraction.

    Each round: evolve a program for each unsolved task, then abstract
    common subtrees from solvers into the library.
    """
    solved = {}       # tid → tree
    best_fit = {}     # tid → (tree, fitness)

    for rnd in range(1, rounds + 1):
        new_count = 0
        tids = [t for t in tasks if t not in solved]
        tids.sort(key=lambda t: -(best_fit[t][1] if t in best_fit else 0))

        for tid in tids:
            prev = best_fit[tid][0] if tid in best_fit else None
            prev_f = best_fit[tid][1] if tid in best_fit else 0
            ps = 120 if prev_f > 0.85 else 80
            gs = 30 if prev_f > 0.85 else 20
            tree, f = search(tasks[tid]["train"], seed=prev, library=library,
                             pop_size=ps, gens=gs)
            if f > best_fit.get(tid, (None, 0))[1]:
                best_fit[tid] = (tree, f)
            if solves(tree, tasks[tid]["train"], library):
                solved[tid] = tree
                new_count += 1
                print(f"  ✓ {tid}: {gp.to_str(tree)}")

        if len(solved) >= 2:
            abstract(list(solved.values()), library)

        n_near = sum(1 for t in tids if best_fit.get(t, (None, 0))[1] > 0.8)
        print(f"Round {rnd}: {len(solved)}/{len(tasks)} solved "
              f"(+{new_count}), {n_near} near, lib={len(library)}")
    return solved


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(tasks, library, train_solvers=None):
    """Score on held-out test examples.

    First tries transferring training solvers directly, then runs per-task search.
    """
    transfer = list({id(t): t for t in (train_solvers or {}).values()}.values())
    solved_train, solved_test, total = 0, 0, 0

    for tid, task in tasks.items():
        train_ex, test_ex = task["train"], task["test"]
        if not test_ex:
            continue
        total += 1

        # Try direct transfer from training solvers
        found = False
        for t in transfer:
            if solves(t, train_ex, library):
                solved_train += 1
                if solves(t, test_ex, library):
                    solved_test += 1
                    print(f"  ✓ {tid} (transfer): {gp.to_str(t)}")
                found = True
                break
        if found:
            continue

        # Per-task search
        tree, _ = search(train_ex, library=library, pop_size=100, gens=25)
        if solves(tree, train_ex, library):
            solved_train += 1
            if solves(tree, test_ex, library):
                solved_test += 1
                print(f"  ✓ {tid}: {gp.to_str(tree)}")

    print(f"Eval: {solved_test}/{total} test ({solved_train} train)")
    return solved_test


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "data/ARC-AGI/data/evaluation"
    rounds     = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    if not os.path.isdir(train_path):
        sys.exit(f"Training data not found: {train_path}")

    library = {}
    train_tasks = load(train_path)
    print(f"Loaded {len(train_tasks)} training tasks")
    train_solved = evolve(train_tasks, rounds, library)

    if os.path.isdir(eval_path):
        eval_tasks = load(eval_path)
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(eval_tasks)} eval tasks (lib={len(library)})")
        print(f"{'='*60}")
        evaluate(eval_tasks, library, train_solvers=train_solved)
