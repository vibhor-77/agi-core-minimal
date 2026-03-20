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

# ── Constants ────────────────────────────────────────────────────────

# Search budget (base values — actual budget scales continuously with fitness)
POP_SIZE_BASE = 80              # population at fitness=0; doubles at fitness=1
GENERATIONS_BASE = 15           # generations at fitness=0; doubles at fitness=1
ELITE_FRACTION = 5              # keep top 1/N of population as elite

# Fitness function
# Changed-cell weight: derived from typical ARC tasks where ~10-20% of cells
# change. Weight of 3 means fixing a changed cell is worth 3 unchanged cells,
# so a program must get ~75% of changed cells right to beat identity.
CHANGED_CELL_WEIGHT = 3.0
# Parsimony: penalty per node. Set so a 50-node tree loses ~0.1 fitness,
# enough to prefer simpler programs among near-equals but not enough to
# prevent finding complex solutions.
PARSIMONY_PENALTY = 0.002

# Abstraction: promote subtrees shared across solvers
MIN_SUBTREE_SIZE = 3            # below this, subtrees are too trivial to abstract
MAX_SUBTREE_SIZE = 20           # above this, subtrees are too specific to reuse
MIN_SHARED_SOLVERS = 2          # must appear in at least this many solver programs


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


# ── Scoring ──────────────────────────────────────────────────────────

IDENTITY = ("get", ("r",), ("c",))


def fitness(tree, examples, library=None):
    """Weighted cell accuracy with parsimony pressure.

    Changed cells are weighted higher than unchanged cells so that
    evolution is rewarded for fixing the cells that actually differ
    between input and output, rather than just preserving the background.

    For different-size tasks (where input and output shapes differ),
    all output cells are treated as "changed" since there is no
    cell-to-cell correspondence with the input.
    """
    total = 0.0
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library)
        if got is None or got.shape != out_a.shape:
            continue
        inp_a = np.asarray(inp)
        if inp_a.shape == out_a.shape:
            changed = inp_a != out_a
        else:
            changed = np.ones(out_a.shape, dtype=bool)
        n_ch, n_un = int(changed.sum()), int((~changed).sum())
        score = 0.0
        if n_ch > 0:
            score += CHANGED_CELL_WEIGHT * np.sum(got[changed] == out_a[changed])
        if n_un > 0:
            score += np.sum(got[~changed] == out_a[~changed])
        max_score = CHANGED_CELL_WEIGHT * n_ch + n_un
        total += score / max_score if max_score > 0 else 0.0
    accuracy = total / len(examples)
    return max(0.0, accuracy - PARSIMONY_PENALTY * gp.size(tree))


def solves(tree, examples, library=None):
    """Check if tree produces exact output for all examples."""
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library)
        if got is None or not np.array_equal(got, out_a):
            return False
    return True


def simplify(tree, examples, library=None):
    """Greedily replace subtrees with simpler equivalents.

    Tries replacing each non-terminal subtree (largest first) with a
    set of canonical simple programs. If the simplified version still
    solves all examples, keep it. This removes overfitting junk and
    improves generalization.
    """
    if not solves(tree, examples, library):
        return tree
    R, C = ("r",), ("c",)
    replacements = [
        R, C, ("max_r",), ("max_c",), IDENTITY, ("get", C, R),
        ("sub", ("sub", ("max_r",), ("const", 1)), R),
        ("sub", ("sub", ("max_c",), ("const", 1)), C),
    ] + [("const", i) for i in range(gp.NUM_COLORS)]
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
    """Generate initial program templates from the atomic primitives.

    These are programs that the system COULD discover through evolution,
    expressed using only the atomic operations. Including them as seeds
    ensures the system doesn't waste generations rediscovering well-known
    geometric transforms, and instead spends its budget on novel compositions.

    Categories:
      - Geometric remaps (7): identity, transpose, flips, rotations
      - Shifts (8): circular shifts by 1-4 in each axis
      - Recolors (90): if cell==X, change to Y (all color pairs)
    """
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
    for k in range(1, 5):
        K = ("const", k)
        seeds.append(("get", ("mod", ("add", R, K), MR), C))   # shift down by k
        seeds.append(("get", R, ("mod", ("add", C, K), MC)))   # shift right by k
    for x in range(gp.NUM_COLORS):
        for y in range(gp.NUM_COLORS):
            if x != y:
                seeds.append(("if", ("eq", ("get", R, C), ("const", x)),
                              ("const", y), ("get", R, C)))
    return seeds


# ── Search ───────────────────────────────────────────────────────────

def search_budget(prev_fitness):
    """Compute search effort that scales continuously with fitness.

    Higher fitness = closer to solving = more value in additional effort.
    Budget scales linearly: fitness=0 gets base, fitness=1 gets 2x base.
    """
    scale = 1.0 + prev_fitness
    return int(POP_SIZE_BASE * scale), int(GENERATIONS_BASE * scale)


def search(examples, seed=None, library=None, pop_size=None, gens=None):
    """Evolve a program for a single task.

    Starts from seed templates + mutations, evolves via tournament selection,
    crossover, and mutation. Stops early when improvement rate drops below
    threshold. Returns (best_tree, best_fitness).
    """
    if pop_size is None or gens is None:
        pop_size, gens = POP_SIZE_BASE, GENERATIONS_BASE

    seeds = make_seeds()
    pop_size = max(pop_size, len(seeds) + 20)
    pop = list(seeds)

    n_mutants = pop_size // ELITE_FRACTION
    if seed is not None and seed != IDENTITY:
        pop.append(seed)
        for _ in range(n_mutants):
            pop.append(gp.mutate(seed, library=library))
    for _ in range(n_mutants):
        pop.append(gp.mutate(IDENTITY, library=library))
    while len(pop) < pop_size:
        pop.append(gp.random_tree(max_depth=4, library=library))

    best_t, best_f = IDENTITY, fitness(IDENTITY, examples, library)
    if seed is not None:
        sf = fitness(seed, examples, library)
        if sf > best_f:
            best_t, best_f = seed, sf

    # Patience scales with fitness: high fitness = worth trying longer.
    # At fitness=0: patience=3. At fitness=0.9: patience=8.
    patience = int(3 + best_f * 6)
    stale = 0
    for gen in range(gens):
        fits = [fitness(p, examples, library) for p in pop]
        improved = False
        for p, f in zip(pop, fits):
            if f > best_f:
                best_t, best_f = p, f
                improved = True
        if improved:
            stale = 0
            patience = int(3 + best_f * 6)
        else:
            stale += 1
        if solves(best_t, examples, library):
            return simplify(best_t, examples, library), 1.0
        if stale > patience:
            break
        # Next generation
        ranked = sorted(zip(fits, pop), reverse=True)
        elite = [p for _, p in ranked[:pop_size // ELITE_FRACTION]]
        new_pop = list(elite) + [IDENTITY, best_t]
        n_best_mutants = pop_size // (ELITE_FRACTION * 2)
        for _ in range(n_best_mutants):
            new_pop.append(gp.mutate(best_t, library=library))
        while len(new_pop) < pop_size:
            p1 = gp.tournament(pop, fits)
            p2 = gp.tournament(pop, fits)
            child = gp.crossover(p1, p2)
            # Mutation probability after crossover: always mutate.
            # Crossover alone is too conservative for this search space.
            child = gp.mutate(child, library=library)
            new_pop.append(child)
        pop = new_pop

    # Neighborhood refinement: probability scales with fitness.
    # Low fitness → don't bother. High fitness → always refine.
    if np.random.random() < best_f:
        best_t, best_f = refine(best_t, examples, library)
    return best_t, best_f


def refine(tree, examples, library=None):
    """Try all single-node edits of tree.

    Exhaustively tests every alternative terminal at every terminal position,
    and every alternative binary op at every binary position. This is cheap
    (typically ~150 evaluations) and converts near-misses into solutions by
    fixing the one wrong constant or operator.
    """
    best_t, best_f = tree, fitness(tree, examples, library)
    terminals = [("r",), ("c",), ("max_r",), ("max_c",)]
    terminals += [("const", i) for i in range(gp.NUM_COLORS)]
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
    """Find subtrees shared by 2+ solvers and add to library.

    This is the compounding mechanism: useful compositions discovered for
    one task become single-node building blocks for future tasks. A subtree
    of size 10 that took many generations to evolve becomes a single "lib"
    terminal that can appear in any random tree.
    """
    if len(solvers) < MIN_SHARED_SOLVERS:
        return
    sub_sources = {}
    for i, tree in enumerate(solvers):
        seen = set()
        for _, sub in gp.subtrees(tree):
            sz = gp.size(sub)
            if sz < MIN_SUBTREE_SIZE or sz > MAX_SUBTREE_SIZE or sub in seen:
                continue
            seen.add(sub)
            sub_sources.setdefault(sub, set()).add(i)
    existing = set(library.values())
    for sub, sources in sub_sources.items():
        if len(sources) >= MIN_SHARED_SOLVERS and sub not in existing:
            name = f"L{len(library)}"
            library[name] = sub
            existing.add(sub)
            print(f"  ++ lib '{name}' = {gp.to_str(sub)} "
                  f"(size={gp.size(sub)}, shared by {len(sources)})")


# ── Evolution loop ───────────────────────────────────────────────────

def evolve(tasks, rounds, library):
    """Per-task evolution with cross-task abstraction.

    Each round: evolve a program for each unsolved task (budget proportional
    to previous fitness), then abstract common subtrees from solvers into the
    library for use in future rounds.
    """
    solved = {}       # tid → tree
    best_fit = {}     # tid → (tree, fitness)

    for rnd in range(1, rounds + 1):
        new_count = 0
        tids = [t for t in tasks if t not in solved]
        # Process higher-fitness tasks first (most likely to solve next)
        tids.sort(key=lambda t: -(best_fit[t][1] if t in best_fit else 0))

        for tid in tids:
            prev = best_fit[tid][0] if tid in best_fit else None
            prev_f = best_fit[tid][1] if tid in best_fit else 0
            ps, gs = search_budget(prev_f)
            tree, f = search(tasks[tid]["train"], seed=prev, library=library,
                             pop_size=ps, gens=gs)
            if f > best_fit.get(tid, (None, 0))[1]:
                best_fit[tid] = (tree, f)
            if solves(tree, tasks[tid]["train"], library):
                solved[tid] = tree
                new_count += 1
                print(f"  ✓ {tid}: {gp.to_str(tree)}")

        if len(solved) >= MIN_SHARED_SOLVERS:
            abstract(list(solved.values()), library)

        n_near = sum(1 for t in tids if best_fit.get(t, (None, 0))[1] > 0.8)
        print(f"Round {rnd}: {len(solved)}/{len(tasks)} solved "
              f"(+{new_count}), {n_near} near, lib={len(library)}")
    return solved


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(tasks, library, train_solvers=None):
    """Score on held-out test examples.

    First tries transferring training solvers directly (free generalization
    test), then runs per-task search with the learned library.
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
        tree, _ = search(train_ex, library=library)
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
