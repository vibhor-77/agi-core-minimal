"""
Tree-based genetic programming engine.

Programs are expression trees that compute f(input_grid, r, c) → color
for each cell (r, c) of the output grid. All primitives are truly atomic:

  Position:    r, c, max_r, max_c     (output coordinates and dimensions)
  Perception:  get(row, col)           (read input grid at computed position)
  Constants:   0–9                     (ARC color values)
  Arithmetic:  add, sub, mod           (coordinate math)
  Logic:       eq, gt, if              (comparison and branching)
"""

import numpy as np

# ── Tree structure ───────────────────────────────────────────────────
# A node is a tuple: (op, child1, child2, ...)
# Terminals have no children: ("r",), ("c",), ("max_r",), ("max_c",),
#   ("const", v), ("lib", name)
# Functions have children per ARITY:
ARITY = {"add": 2, "sub": 2, "mod": 2, "eq": 2, "gt": 2, "get": 2, "if": 3}


def random_tree(max_depth=4, depth=0, library=None):
    """Generate a random expression tree."""
    p_term = 0.4 if depth < max_depth - 1 else 1.0
    if depth >= max_depth or np.random.random() < p_term:
        if library and np.random.random() < 0.15:
            return ("lib", list(library)[np.random.randint(len(library))])
        ch = np.random.randint(14)  # 4 vars + 10 constants
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
def evaluate(tree, input_grid, out_shape=None, library=None):
    """Evaluate tree for every cell of the output grid.

    Args:
        tree: expression tree
        input_grid: 2D list or array (the input)
        out_shape: (rows, cols) of output; defaults to input shape
        library: dict mapping names to subtrees

    Returns:
        2D int array (values 0–9), or None on error.
    """
    g = np.asarray(input_grid)
    in_r, in_c = g.shape
    o_r, o_c = out_shape if out_shape is not None else g.shape

    r_arr = np.broadcast_to(np.arange(o_r)[:, None], (o_r, o_c))
    c_arr = np.broadcast_to(np.arange(o_c)[None, :], (o_r, o_c))
    MR, MC = np.int64(o_r), np.int64(o_c)
    node_count = [0]

    def _eval(node):
        node_count[0] += 1
        if node_count[0] > 300:
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
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, in_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, in_c - 1).astype(int)
            return g[ri, ci]
        if op == "if":
            return np.where(a != 0, b, _eval(node[3]))
        return np.int64(0)

    try:
        result = _eval(tree)
        out = np.broadcast_to(np.asarray(result), (o_r, o_c)).copy()
        return np.clip(out, 0, 9).astype(int)
    except Exception:
        return None


# ── GP operators ─────────────────────────────────────────────────────
def crossover(p1, p2, max_depth=7):
    """Swap a random subtree of p1 with a random subtree of p2."""
    s1, s2 = subtrees(p1), subtrees(p2)
    path, _ = s1[np.random.randint(len(s1))]
    _, donor = s2[np.random.randint(len(s2))]
    child = replace(p1, path, donor)
    return child if depth(child) <= max_depth else p1


def mutate(tree, max_depth=7, library=None):
    """Apply one random mutation: subtree, point, or hoist."""
    r = np.random.random()
    subs = subtrees(tree)
    path, node = subs[np.random.randint(len(subs))]

    if r < 0.5:  # subtree: replace with random tree
        remaining = max(1, max_depth - len(path))
        return replace(tree, path, random_tree(remaining, library=library))

    if r < 0.8:  # point: change node type or constant
        op = node[0]
        if op == "const":
            return replace(tree, path, ("const", np.random.randint(10)))
        if op in ("r", "c", "max_r", "max_c"):
            terms = [("r",), ("c",), ("max_r",), ("max_c",), ("const", np.random.randint(10))]
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


def tournament(population, fitnesses, k=4):
    """Select the fittest of k random individuals."""
    idxs = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    return population[max(idxs, key=lambda i: fitnesses[i])]
