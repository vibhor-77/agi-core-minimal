"""
Tree-based genetic programming engine.

Programs are expression trees that compute f(grid, r, c) → color for each
cell (r, c) of the output grid, optionally applied for multiple passes.

All primitives are truly atomic:

  Position:    r, c, max_r, max_c     (output coordinates and dimensions)
  Perception:  get(row, col)           (read CURRENT grid — changes each pass)
               inp(row, col)           (read ORIGINAL input — constant across passes)
  Constants:   0–9                     (ARC color values)
  Arithmetic:  add, sub, mod           (coordinate math)
  Logic:       eq, gt, if              (comparison and branching)

Multi-pass: when passes > 1, the tree is applied repeatedly. Each pass reads
its own output from the previous pass via `get`, while `inp` always reads the
original input. This enables information propagation (cellular automata),
flood-fill, and iterative refinement — all from the same atomic primitives.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

# Domain-determined (not tunable — these follow from the problem definition)
NUM_COLORS = 10                 # ARC uses colors 0–9
NUM_TERMINALS = 4 + NUM_COLORS  # r, c, max_r, max_c + one per color
MAX_TREE_DEPTH = 7              # limits composition depth; 7 allows ~128 nodes
MAX_PASSES = 5                  # maximum multi-pass iterations

# Tree generation probabilities
# P_TERMINAL ≈ 1/(1+mean_arity) keeps expected tree size finite.
# Mean arity across our ops is ~2.1, so 1/3.1 ≈ 0.32. We use 0.35.
P_TERMINAL = 0.35

# Safety: max node evaluations per pass = 2^(MAX_TREE_DEPTH+1)
MAX_EVAL_NODES = 2 ** (MAX_TREE_DEPTH + 1)

# Tournament: k=4 gives ~75% chance of selecting the best quartile.
TOURNAMENT_K = 4

# Mutation operator weights (relative, normalized internally).
MUTATE_WEIGHTS = {"subtree": 5, "point": 3, "hoist": 2}
_MUTATE_TOTAL = sum(MUTATE_WEIGHTS.values())


# ── Tree structure ───────────────────────────────────────────────────
# A node is a tuple: (op, child1, child2, ...)
# Terminals: ("r",), ("c",), ("max_r",), ("max_c",), ("const", v), ("lib", name)
# Functions:
ARITY = {
    "add": 2, "sub": 2, "mod": 2,    # arithmetic
    "eq": 2, "gt": 2,                 # comparison
    "get": 2,                          # read current grid (changes each pass)
    "inp": 2,                          # read original input (constant across passes)
    "if": 3,                           # conditional
}


def random_tree(max_depth=4, depth=0, library=None):
    """Generate a random expression tree."""
    p_term = P_TERMINAL if depth < max_depth - 1 else 1.0
    if depth >= max_depth or np.random.random() < p_term:
        if library:
            lib_frac = len(library) / (len(library) + NUM_TERMINALS)
            if np.random.random() < lib_frac:
                return ("lib", list(library)[np.random.randint(len(library))])
        ch = np.random.randint(NUM_TERMINALS)
        if ch == 0: return ("r",)
        if ch == 1: return ("c",)
        if ch == 2: return ("max_r",)
        if ch == 3: return ("max_c",)
        return ("const", ch - 4)
    ops = list(ARITY)
    op = ops[np.random.randint(len(ops))]
    children = [random_tree(max_depth, depth + 1, library) for _ in range(ARITY[op])]
    return (op, *children)


def size(tree):
    """Number of nodes in tree."""
    if tree[0] not in ARITY:
        return 1
    return 1 + sum(size(tree[i]) for i in range(1, 1 + ARITY[tree[0]]))


def depth(tree):
    """Maximum depth of tree."""
    if tree[0] not in ARITY:
        return 0
    return 1 + max(depth(tree[i]) for i in range(1, 1 + ARITY[tree[0]]))


def to_str(tree):
    """Human-readable string representation."""
    op = tree[0]
    if op == "const": return str(tree[1])
    if op == "lib": return tree[1]
    if op not in ARITY: return op
    args = ", ".join(to_str(tree[i]) for i in range(1, 1 + ARITY[op]))
    return f"{op}({args})"


# ── Tree manipulation ────────────────────────────────────────────────

def subtrees(tree):
    """All (path, subtree) pairs. Path is a tuple of child indices."""
    result = [((), tree)]
    if tree[0] not in ARITY:
        return result
    for i in range(1, 1 + ARITY[tree[0]]):
        for path, node in subtrees(tree[i]):
            result.append(((i,) + path, node))
    return result


def replace(tree, path, new):
    """Return tree with subtree at path replaced by new."""
    if not path:
        return new
    lst = list(tree)
    lst[path[0]] = replace(tree[path[0]], path[1:], new)
    return tuple(lst)


# ── Evaluation ───────────────────────────────────────────────────────

def _evaluate_once(tree, g_current, g_original, out_shape, library):
    """Evaluate tree once for all output cells. Returns 2D int array or None."""
    cur_r, cur_c = g_current.shape
    orig_r, orig_c = g_original.shape
    o_r, o_c = out_shape

    r_arr = np.broadcast_to(np.arange(o_r)[:, None], (o_r, o_c))
    c_arr = np.broadcast_to(np.arange(o_c)[None, :], (o_r, o_c))
    MR, MC = np.int64(o_r), np.int64(o_c)
    node_count = [0]

    def _eval(node):
        node_count[0] += 1
        if node_count[0] > MAX_EVAL_NODES:
            raise RuntimeError("expression too large")
        op = node[0]
        # Terminals
        if op == "r": return r_arr
        if op == "c": return c_arr
        if op == "max_r": return MR
        if op == "max_c": return MC
        if op == "const": return np.int64(node[1])
        if op == "lib":
            sub = (library or {}).get(node[1])
            return _eval(sub) if sub is not None else np.int64(0)
        # Functions
        a = _eval(node[1])
        b = _eval(node[2])
        if op == "add": return a + b
        if op == "sub": return a - b
        if op == "mod": return np.mod(a, np.where(b == 0, 1, b))
        if op == "eq": return (a == b).astype(np.int64)
        if op == "gt": return (a > b).astype(np.int64)
        if op == "get":
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, cur_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, cur_c - 1).astype(int)
            return g_current[ri, ci]
        if op == "inp":
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, orig_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, orig_c - 1).astype(int)
            return g_original[ri, ci]
        if op == "if":
            return np.where(a != 0, b, _eval(node[3]))
        return np.int64(0)

    try:
        result = _eval(tree)
        out = np.broadcast_to(np.asarray(result), (o_r, o_c)).copy()
        return np.clip(out, 0, 9).astype(int)
    except Exception:
        return None


def evaluate(tree, input_grid, out_shape=None, library=None, passes=1):
    """Evaluate tree, optionally applying it for multiple passes.

    Pass 1: get reads from input, inp reads from input (equivalent).
    Pass 2+: get reads from previous pass output, inp reads from original input.

    This enables cellular-automata-like computation: local rules that through
    iteration produce global behavior (flood fill, object detection, etc.).
    """
    g_original = np.asarray(input_grid)
    o_shape = out_shape if out_shape is not None else g_original.shape
    g_current = g_original

    for _ in range(passes):
        result = _evaluate_once(tree, g_current, g_original, o_shape, library)
        if result is None:
            return None
        g_current = result

    return g_current


# ── GP operators ─────────────────────────────────────────────────────

def crossover(p1, p2):
    """Swap a random subtree of p1 with a random subtree of p2."""
    s1, s2 = subtrees(p1), subtrees(p2)
    path, _ = s1[np.random.randint(len(s1))]
    _, donor = s2[np.random.randint(len(s2))]
    child = replace(p1, path, donor)
    return child if depth(child) <= MAX_TREE_DEPTH else p1


def mutate(tree, library=None):
    """Apply one random mutation: subtree, point, or hoist."""
    r = np.random.random() * _MUTATE_TOTAL
    subs = subtrees(tree)
    path, node = subs[np.random.randint(len(subs))]

    if r < MUTATE_WEIGHTS["subtree"]:
        remaining = max(1, MAX_TREE_DEPTH - len(path))
        return replace(tree, path, random_tree(remaining, library=library))

    if r < MUTATE_WEIGHTS["subtree"] + MUTATE_WEIGHTS["point"]:
        op = node[0]
        if op == "const":
            return replace(tree, path, ("const", np.random.randint(NUM_COLORS)))
        if op in ("r", "c", "max_r", "max_c"):
            terms = [("r",), ("c",), ("max_r",), ("max_c",),
                     ("const", np.random.randint(NUM_COLORS))]
            return replace(tree, path, terms[np.random.randint(len(terms))])
        if op in ARITY and ARITY[op] == 2:
            bin_ops = [k for k, v in ARITY.items() if v == 2]
            new_op = bin_ops[np.random.randint(len(bin_ops))]
            return replace(tree, path, (new_op, node[1], node[2]))
        return tree

    # hoist: replace tree with one of its subtrees
    if len(subs) > 1:
        _, sub = subs[np.random.randint(1, len(subs))]
        return sub
    return tree


def tournament(population, fitnesses, k=TOURNAMENT_K):
    """Select the fittest of k random individuals."""
    idxs = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    return population[max(idxs, key=lambda i: fitnesses[i])]
