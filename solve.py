"""
Genetic Programming for ARC-AGI: cell-level program synthesis.

Programs are expression trees computing f(grid, r, c) → color for each output
cell. Primitives are truly atomic: read cells, arithmetic, comparison, conditional.
Evolution discovers compositions (flip, rotate, recolor, pattern detection).
Compounding: useful subtrees get abstracted into library primitives, collapsing
depth and enabling deeper compositions in future generations.
"""
import json, sys, os, numpy as np
from pathlib import Path

# ── Data ─────────────────────────────────────────────────────────────
def load(path):
    tasks = {}
    for f in sorted(Path(path).glob("*.json")):
        t = json.loads(f.read_text())
        tasks[f.stem] = {
            "train": [(e["input"], e["output"]) for e in t["train"]],
            "test":  [(e["input"], e["output"]) for e in t.get("test", [])],
        }
    return tasks

def filter_same_size(tasks):
    """Keep tasks where all train examples have input.shape == output.shape."""
    out = {}
    for tid, task in tasks.items():
        if all(np.array(i).shape == np.array(o).shape for i, o in task["train"]):
            out[tid] = task
    return out

# ── Tree representation ──────────────────────────────────────────────
# Terminals: ("r",) ("c",) ("max_r",) ("max_c",) ("const",v) ("lib",name)
# Functions: ("add",l,r) ("sub",l,r) ("mod",l,r) ("eq",l,r) ("gt",l,r)
#            ("get",row,col) ("if",cond,then,else)
ARITY = {"add": 2, "sub": 2, "mod": 2, "eq": 2, "gt": 2, "get": 2, "if": 3}
library = {}  # name → tree (abstracted subtrees)

def random_tree(max_depth=4, depth=0):
    p_term = 0.4 if depth < max_depth - 1 else 1.0
    if depth >= max_depth or np.random.random() < p_term:
        # Occasionally use library entries as terminals
        if library and np.random.random() < 0.15:
            name = list(library)[np.random.randint(len(library))]
            return ("lib", name)
        ch = np.random.randint(14)
        if ch == 0: return ("r",)
        if ch == 1: return ("c",)
        if ch == 2: return ("max_r",)
        if ch == 3: return ("max_c",)
        return ("const", ch - 4)  # 0-9
    ops = list(ARITY)
    op = ops[np.random.randint(len(ops))]
    children = [random_tree(max_depth, depth + 1) for _ in range(ARITY[op])]
    return (op, *children)

def tree_size(tree):
    op = tree[0]
    if op not in ARITY:
        return 1
    return 1 + sum(tree_size(tree[i]) for i in range(1, 1 + ARITY[op]))

def tree_depth(tree):
    op = tree[0]
    if op not in ARITY:
        return 0
    return 1 + max(tree_depth(tree[i]) for i in range(1, 1 + ARITY[op]))

def tree_str(tree):
    op = tree[0]
    if op == "const": return str(tree[1])
    if op == "lib": return tree[1]
    if op not in ARITY: return op
    args = ", ".join(tree_str(tree[i]) for i in range(1, 1 + ARITY[op]))
    return f"{op}({args})"

def all_subtrees(tree):
    """Return [(path, subtree), ...] for every node."""
    result = [((), tree)]
    op = tree[0]
    if op not in ARITY:
        return result
    for i in range(1, 1 + ARITY[op]):
        for path, node in all_subtrees(tree[i]):
            result.append(((i,) + path, node))
    return result

def replace_at(tree, path, new):
    if not path:
        return new
    lst = list(tree)
    lst[path[0]] = replace_at(tree[path[0]], path[1:], new)
    return tuple(lst)

# ── Vectorized tree evaluation ───────────────────────────────────────
def eval_tree(tree, grid):
    """Evaluate tree for all output cells simultaneously. Returns 2D int array."""
    g = np.asarray(grid)
    rows, cols = g.shape
    r_arr = np.broadcast_to(np.arange(rows)[:, None], (rows, cols))
    c_arr = np.broadcast_to(np.arange(cols)[None, :], (rows, cols))
    MR, MC = np.int64(rows), np.int64(cols)
    _depth = [0]

    def _e(node):
        _depth[0] += 1
        if _depth[0] > 200:
            raise RuntimeError("too deep")
        op = node[0]
        if op == "r": return r_arr
        if op == "c": return c_arr
        if op == "max_r": return MR
        if op == "max_c": return MC
        if op == "const": return np.int64(node[1])
        if op == "lib":
            t = library.get(node[1])
            return _e(t) if t is not None else np.int64(0)
        a = _e(node[1])
        b = _e(node[2])
        if op == "add": return a + b
        if op == "sub": return a - b
        if op == "mod":
            bs = np.where(b == 0, 1, b)
            return np.mod(a, bs)
        if op == "eq": return (a == b).astype(np.int64)
        if op == "gt": return (a > b).astype(np.int64)
        if op == "get":
            ri = np.clip(np.broadcast_to(np.asarray(a), (rows, cols)), 0, rows - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (rows, cols)), 0, cols - 1).astype(int)
            return g[ri, ci]
        if op == "if":
            c = _e(node[3])
            return np.where(a != 0, b, c)
        return np.int64(0)

    try:
        result = _e(tree)
        out = np.broadcast_to(np.asarray(result), (rows, cols)).copy()
        return np.clip(out, 0, 9).astype(int)
    except Exception:
        return None

# ── Fitness ──────────────────────────────────────────────────────────
def fitness(tree, examples):
    """Weighted cell accuracy with parsimony pressure.
    Changed cells worth 3x. Small penalty for tree size (Occam's razor)."""
    total = 0.0
    for inp, out in examples:
        got = eval_tree(tree, inp)
        inp_a, out_a = np.asarray(inp), np.asarray(out)
        if got is None or got.shape != out_a.shape:
            continue
        changed = inp_a != out_a
        n_ch, n_un = int(changed.sum()), int((~changed).sum())
        score = 0.0
        if n_ch > 0:
            score += 3.0 * np.sum(got[changed] == out_a[changed])
        if n_un > 0:
            score += np.sum(got[~changed] == out_a[~changed])
        max_score = 3.0 * n_ch + n_un
        total += score / max_score if max_score > 0 else 0.0
    accuracy = total / len(examples)
    # Parsimony: prefer simpler programs (penalty ~0.02 at size 10, ~0.1 at size 50)
    size_penalty = 0.002 * tree_size(tree)
    return max(0.0, accuracy - size_penalty)

def solves(tree, examples):
    """Exact match on all examples."""
    for inp, out in examples:
        got = eval_tree(tree, inp)
        if got is None or not np.array_equal(got, np.asarray(out)):
            return False
    return True

def simplify(tree, examples):
    """Greedily replace subtrees with simpler equivalents while preserving correctness."""
    if not solves(tree, examples):
        return tree
    # Try terminals and small useful subtrees as replacements
    R, C = ("r",), ("c",)
    replacements = [R, C, ("max_r",), ("max_c",)]
    replacements += [("const", i) for i in range(10)]
    replacements += [IDENTITY, ("get", C, R)]  # identity and transpose
    replacements += [("sub", ("sub", ("max_r",), ("const", 1)), R)]  # max_r-1-r
    replacements += [("sub", ("sub", ("max_c",), ("const", 1)), C)]  # max_c-1-c
    changed = True
    while changed:
        changed = False
        subs = all_subtrees(tree)
        # Try largest subtrees first (biggest simplification wins)
        subs.sort(key=lambda ps: -tree_size(ps[1]))
        for path, sub in subs:
            if not path:
                continue
            if sub[0] not in ARITY:
                continue
            for repl in replacements:
                if tree_size(repl) >= tree_size(sub):
                    continue  # only replace with something smaller
                candidate = replace_at(tree, path, repl)
                if solves(candidate, examples):
                    tree = candidate
                    changed = True
                    break
            if changed:
                break
    return tree

# ── GP operators ─────────────────────────────────────────────────────
def gp_crossover(p1, p2, max_d=7):
    subs1 = all_subtrees(p1)
    subs2 = all_subtrees(p2)
    path1, _ = subs1[np.random.randint(len(subs1))]
    _, donor = subs2[np.random.randint(len(subs2))]
    child = replace_at(p1, path1, donor)
    return child if tree_depth(child) <= max_d else p1

def gp_subtree_mutate(tree, max_d=7):
    subs = all_subtrees(tree)
    path, _ = subs[np.random.randint(len(subs))]
    remaining = max(1, max_d - len(path))
    return replace_at(tree, path, random_tree(max_depth=remaining))

def gp_point_mutate(tree):
    subs = all_subtrees(tree)
    path, node = subs[np.random.randint(len(subs))]
    op = node[0]
    if op == "const":
        return replace_at(tree, path, ("const", np.random.randint(10)))
    if op in ("r", "c", "max_r", "max_c"):
        terms = [("r",), ("c",), ("max_r",), ("max_c",), ("const", np.random.randint(10))]
        return replace_at(tree, path, terms[np.random.randint(len(terms))])
    if op in ARITY and ARITY[op] == 2:
        bin_ops = [k for k, v in ARITY.items() if v == 2]
        new_op = bin_ops[np.random.randint(len(bin_ops))]
        return replace_at(tree, path, (new_op, node[1], node[2]))
    return tree

def gp_hoist(tree):
    """Replace tree with one of its subtrees (simplification)."""
    subs = all_subtrees(tree)
    if len(subs) <= 1:
        return tree
    _, sub = subs[np.random.randint(1, len(subs))]
    return sub

def breed(p1, p2):
    """Apply one random GP operator."""
    r = np.random.random()
    if r < 0.50: return gp_crossover(p1, p2)
    if r < 0.70: return gp_subtree_mutate(p1)
    if r < 0.85: return gp_point_mutate(p1)
    return gp_hoist(p1)

def tournament(pop, fits, k=4):
    idxs = np.random.choice(len(pop), size=min(k, len(pop)), replace=False)
    return pop[max(idxs, key=lambda i: fits[i])]

# ── Abstraction ──────────────────────────────────────────────────────
def abstract(solvers):
    """Find subtrees shared by 2+ solvers → promote to library."""
    if len(solvers) < 2:
        return []
    # Collect non-trivial subtrees with their source solver index
    sub_sources = {}
    for i, tree in enumerate(solvers):
        seen_in_tree = set()
        for _, sub in all_subtrees(tree):
            sz = tree_size(sub)
            if sz < 3 or sz > 20:
                continue
            if sub in seen_in_tree:
                continue
            seen_in_tree.add(sub)
            if sub not in sub_sources:
                sub_sources[sub] = set()
            sub_sources[sub].add(i)
    new = []
    existing = set(library.values())
    for sub, sources in sub_sources.items():
        if len(sources) >= 2 and sub not in existing:
            name = f"L{len(library)}"
            library[name] = sub
            existing.add(sub)
            new.append(name)
            print(f"  ++ lib '{name}' = {tree_str(sub)} "
                  f"(size={tree_size(sub)}, in {len(sources)} solvers)")
    return new

# ── Evolution ────────────────────────────────────────────────────────
IDENTITY = ("get", ("r",), ("c",))  # the "do nothing" program

def _make_seeds():
    """Generate useful program templates from existing primitives."""
    R, C, MR, MC = ("r",), ("c",), ("max_r",), ("max_c",)
    ONE = ("const", 1)
    seeds = [IDENTITY]
    # Geometric remaps: get(f(r,c), g(r,c))
    comp_r = ("sub", ("sub", MR, ONE), R)  # max_r - 1 - r
    comp_c = ("sub", ("sub", MC, ONE), C)  # max_c - 1 - c
    seeds += [
        ("get", C, R),                                     # transpose
        ("get", comp_r, C),                                # flip_v
        ("get", R, comp_c),                                # flip_h
        ("get", comp_r, comp_c),                           # rotate 180
        ("get", C, comp_r),                                # rotate_cw
        ("get", comp_c, R),                                # rotate_ccw
        ("get", ("mod", ("add", R, ONE), MR), C),          # shift_d
        ("get", R, ("mod", ("add", C, ONE), MC)),          # shift_r
    ]
    # Shifts by 2-4
    for k in range(2, 5):
        K = ("const", k)
        seeds.append(("get", ("mod", ("add", R, K), MR), C))
        seeds.append(("get", R, ("mod", ("add", C, K), MC)))
    # Conditional recolors: if(eq(get(r,c), X), Y, get(r,c))
    for x in range(10):
        for y in range(10):
            if x != y:
                seeds.append(("if", ("eq", ("get", R, C), ("const", x)),
                              ("const", y), ("get", R, C)))
    return seeds

_SEEDS = None

def get_seeds():
    global _SEEDS
    if _SEEDS is None:
        _SEEDS = _make_seeds()
    return _SEEDS

def search_task(task_examples, seed=None, pop_size=80, gens=20):
    """Mini-evolution for a single task, building on previous best (seed)."""
    # Seed population: templates + previous best + mutations + random
    seeds = get_seeds()
    pop_size = max(pop_size, len(seeds) + 20)
    pop = list(seeds)  # all geometric + recolor templates
    if seed is not None and seed != IDENTITY:
        pop.append(seed)
        for _ in range(pop_size // 6):
            pop.append(gp_subtree_mutate(seed, max_d=6))
        for _ in range(pop_size // 6):
            pop.append(breed(seed, IDENTITY))
    for _ in range(pop_size // 6):
        pop.append(gp_subtree_mutate(IDENTITY, max_d=5))
    while len(pop) < pop_size:
        pop.append(random_tree(max_depth=4))
    best_f = fitness(IDENTITY, task_examples)
    best_t = IDENTITY
    if seed is not None:
        sf = fitness(seed, task_examples)
        if sf > best_f:
            best_f, best_t = sf, seed
    stale = 0
    for _ in range(gens):
        fits = [fitness(p, task_examples) for p in pop]
        for p, f in zip(pop, fits):
            if f > best_f:
                best_f, best_t = f, p
                stale = 0
        if solves(best_t, task_examples):
            best_t = simplify(best_t, task_examples)
            return best_t, 1.0
        stale += 1
        if stale > 7:
            break
        # Next gen
        ranked = sorted(zip(fits, pop), reverse=True)
        elite = [p for _, p in ranked[:pop_size // 5]]
        new_pop = list(elite) + [IDENTITY, best_t]
        for _ in range(pop_size // 10):
            new_pop.append(gp_subtree_mutate(best_t, max_d=6))
        for _ in range(pop_size // 10):
            new_pop.append(gp_subtree_mutate(IDENTITY, max_d=5))
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            new_pop.append(breed(p1, p2))
        pop = new_pop
    return best_t, best_f

def evolve(tasks, rounds):
    """Per-task evolution + library abstraction for cross-task transfer."""
    all_solved = {}   # tid → tree
    best_fit = {}     # tid → (tree, fitness)

    for rnd in range(1, rounds + 1):
        new_this_round = 0
        tids = [t for t in tasks if t not in all_solved]
        # Sort: near-misses first (more search budget for promising tasks)
        tids.sort(key=lambda t: -best_fit[t][1] if t in best_fit else 0)
        for tid in tids:
            prev_best = best_fit[tid][0] if tid in best_fit else None
            prev_f = best_fit[tid][1] if tid in best_fit else 0
            # More budget for near-misses
            ps = 120 if prev_f > 0.85 else 80
            gs = 30 if prev_f > 0.85 else 20
            tree, f = search_task(tasks[tid]["train"], seed=prev_best,
                                  pop_size=ps, gens=gs)
            if f > best_fit.get(tid, (None, 0))[1]:
                best_fit[tid] = (tree, f)
            if solves(tree, tasks[tid]["train"]):
                all_solved[tid] = tree
                new_this_round += 1
                print(f"  ✓ {tid}: {tree_str(tree)}")

        # Abstraction: promote common subtrees from solvers
        if len(all_solved) >= 2:
            abstract(list(all_solved.values()))

        n_near = sum(1 for t in tids if best_fit.get(t, (None, 0))[1] > 0.8)
        print(f"Round {rnd}: {len(all_solved)}/{len(tasks)} solved "
              f"(+{new_this_round}), {n_near} near, lib={len(library)}")
    return all_solved

# ── Evaluation ───────────────────────────────────────────────────────
def evaluate(tasks, train_solvers=None):
    """Per-task evolutionary search with frozen library.
    Also tries known training solvers directly (transfer)."""
    solved_train, solved_test, total = 0, 0, 0
    # Collect unique solver trees from training for direct transfer
    transfer = list({id(t): t for t in (train_solvers or {}).values()}.values())
    for tid, task in tasks.items():
        train_ex, test_ex = task["train"], task["test"]
        if not test_ex:
            continue
        if not all(np.array(i).shape == np.array(o).shape for i, o in test_ex):
            continue
        total += 1
        # First: try all training solvers directly (fast transfer)
        found = False
        for t in transfer:
            if solves(t, train_ex):
                solved_train += 1
                if solves(t, test_ex):
                    solved_test += 1
                    print(f"  ✓ {tid} eval pass (transfer): {tree_str(t)}")
                else:
                    print(f"  ~ {tid} train pass (transfer), eval fail")
                found = True
                break
        if found:
            continue
        # Then: per-task evolution
        tree, _ = search_task(train_ex, pop_size=100, gens=25)
        if solves(tree, train_ex):
            solved_train += 1
            if solves(tree, test_ex):
                solved_test += 1
                print(f"  ✓ {tid} eval pass: {tree_str(tree)}")
            else:
                print(f"  ~ {tid} train pass, eval fail")
    print(f"Eval: {solved_test}/{total} test solved ({solved_train} train solved)")
    return solved_test

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "data/ARC-AGI/data/evaluation"
    rounds     = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    if not os.path.isdir(train_path):
        sys.exit(f"Training data not found: {train_path}")

    train_tasks = filter_same_size(load(train_path))
    print(f"Loaded {len(train_tasks)} same-size training tasks")
    train_solved = evolve(train_tasks, rounds)

    if os.path.isdir(eval_path):
        eval_tasks = filter_same_size(load(eval_path))
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(eval_tasks)} same-size eval tasks "
              f"(library={len(library)})")
        print(f"{'='*60}")
        evaluate(eval_tasks, train_solvers=train_solved)
