"""
Tree-based genetic programming engine.

Programs are expression trees that compute f(grid, r, c) → color for each
cell (r, c) of the output grid, optionally applied for multiple passes.

All primitives are truly atomic:

  Position:    r, c, max_r, max_c     (output coordinates and dimensions)
               inp_r, inp_c           (input grid dimensions)
  Perception:  get(row, col)           (read CURRENT grid — changes each pass)
               inp(row, col)           (read ORIGINAL input — constant across passes)
  Constants:   0–9                     (ARC color values)
  Arithmetic:  add, sub, mod           (coordinate math)
  Logic:       eq, gt, if              (comparison and branching)

Derived primitives (provably equivalent to compositions of atomics):
  Neighborhood: n_count(r, c, v)       (count of 4-neighbors with color v)
  Row/column:   row_count(r, v)        (count of color v in row r)
                col_count(c, v)        (count of color v in column c)
  Global:       total_count(v)         (count of color v in input)
                mode_color             (most common color in input)
  Object:       obj_id(r, c)           (connected component ID, 0=background)
                obj_size(r, c)         (cell count of component at (r,c))
                obj_color(r, c)        (color of component at (r,c))
                obj_top/left/bottom/right(r, c)  (bounding box of component)
                obj_count              (number of non-background components)
                max_obj_size           (size of largest component)

Multi-pass: when passes > 1, the tree is applied repeatedly. Each pass reads
its own output from the previous pass via `get`, while `inp` always reads the
original input. This enables information propagation (cellular automata),
flood-fill, and iterative refinement — all from the same atomic primitives.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

# Domain-determined (not tunable — these follow from the problem definition)
NUM_COLORS = 10                 # ARC uses colors 0–9
NUM_TERMINALS = 11 + NUM_COLORS  # r, c, max_r, max_c, inp_r, inp_c, pass_num, mode_color, obj_count, max_obj_size + colors
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
# Terminals: ("r",), ("c",), ("max_r",), ("max_c",), ("pass_num",),
#            ("const", v), ("lib", name)
# Functions:
ARITY = {
    "add": 2, "sub": 2, "mod": 2,    # arithmetic
    "eq": 2, "gt": 2,                 # comparison
    "get": 2,                          # read current grid (changes each pass)
    "inp": 2,                          # read original input (constant across passes)
    "if": 3,                           # conditional
    # Derived primitives (provably equivalent to atomic compositions)
    "n_count": 3,                      # count of 4-neighbors with color v at (r, c)
    "row_count": 2,                    # count of cells in row r with color v
    "col_count": 2,                    # count of cells in col c with color v
    "total_count": 1,                  # count of all cells with color v in input
    # Object (connected component) primitives — computed on g_original via BFS
    "obj_id": 2, "obj_size": 2, "obj_color": 2,       # component identity
    "obj_top": 2, "obj_left": 2, "obj_bottom": 2, "obj_right": 2,  # bounding box
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
        if ch == 4: return ("inp_r",)
        if ch == 5: return ("inp_c",)
        if ch == 6: return ("pass_num",)
        if ch == 7: return ("mode_color",)
        if ch == 8: return ("obj_count",)
        if ch == 9: return ("max_obj_size",)
        return ("const", ch - 10)
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

def _evaluate_once(tree, g_current, g_original, out_shape, library, pass_num=0):
    """Evaluate tree once for all output cells. Returns 2D int array or None."""
    cur_r, cur_c = g_current.shape
    orig_r, orig_c = g_original.shape
    o_r, o_c = out_shape

    r_arr = np.broadcast_to(np.arange(o_r)[:, None], (o_r, o_c))
    c_arr = np.broadcast_to(np.arange(o_c)[None, :], (o_r, o_c))
    MR, MC = np.int64(o_r), np.int64(o_c)
    IR, IC = np.int64(orig_r), np.int64(orig_c)
    PN = np.int64(pass_num)
    node_count = [0]

    # Pre-compute derived aggregations on g_current
    # Neighbor count: for each (r, c, v), count of 4-neighbors equal to v
    # We pre-compute per-color neighbor count maps
    _n_count_cache = {}
    def _get_n_count(v):
        v = int(v) % NUM_COLORS
        if v not in _n_count_cache:
            mask = (g_current == v).astype(np.int64)
            count = np.zeros_like(mask)
            if cur_r > 1:
                count[1:, :] += mask[:-1, :]   # neighbor above
                count[:-1, :] += mask[1:, :]   # neighbor below
            if cur_c > 1:
                count[:, 1:] += mask[:, :-1]   # neighbor left
                count[:, :-1] += mask[:, 1:]   # neighbor right
            _n_count_cache[v] = count
        return _n_count_cache[v]

    # Row/column color counts: row_hist[r, v] = count of color v in row r
    row_hist = np.zeros((cur_r, NUM_COLORS), dtype=np.int64)
    col_hist = np.zeros((cur_c, NUM_COLORS), dtype=np.int64)
    for v in range(NUM_COLORS):
        row_hist[:, v] = (g_current == v).sum(axis=1)
        col_hist[:, v] = (g_current == v).sum(axis=0)

    # Global aggregation on original input
    global_hist = np.zeros(NUM_COLORS, dtype=np.int64)
    for v in range(NUM_COLORS):
        global_hist[v] = (g_original == v).sum()
    MODE_COLOR = np.int64(np.argmax(global_hist))

    # ── Connected component labeling (BFS on g_original, same-color 4-connectivity) ──
    _label_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _comp_size = {}   # label -> cell count
    _comp_color = {}  # label -> color
    _comp_top = {}; _comp_left = {}; _comp_bottom = {}; _comp_right = {}
    _next_label = 1
    for _sr in range(orig_r):
        for _sc in range(orig_c):
            if _label_map[_sr, _sc] != 0 or g_original[_sr, _sc] == 0:
                continue
            _color = int(g_original[_sr, _sc])
            _lbl = _next_label; _next_label += 1
            _queue = [(_sr, _sc)]
            _label_map[_sr, _sc] = _lbl
            _sz = 0; _t = _sr; _l = _sc; _b = _sr; _ri = _sc
            _qi = 0
            while _qi < len(_queue):
                _qr, _qc = _queue[_qi]; _qi += 1
                _sz += 1
                if _qr < _t: _t = _qr
                if _qr > _b: _b = _qr
                if _qc < _l: _l = _qc
                if _qc > _ri: _ri = _qc
                for _dr, _dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    _nr, _nc = _qr+_dr, _qc+_dc
                    if 0 <= _nr < orig_r and 0 <= _nc < orig_c and _label_map[_nr, _nc] == 0 and g_original[_nr, _nc] == _color:
                        _label_map[_nr, _nc] = _lbl
                        _queue.append((_nr, _nc))
            _comp_size[_lbl] = _sz
            _comp_color[_lbl] = _color
            _comp_top[_lbl] = _t; _comp_left[_lbl] = _l
            _comp_bottom[_lbl] = _b; _comp_right[_lbl] = _ri

    # Build lookup maps
    _obj_size_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _obj_color_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _obj_top_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _obj_left_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _obj_bottom_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    _obj_right_map = np.zeros((orig_r, orig_c), dtype=np.int64)
    for _lbl in _comp_size:
        _mask = _label_map == _lbl
        _obj_size_map[_mask] = _comp_size[_lbl]
        _obj_color_map[_mask] = _comp_color[_lbl]
        _obj_top_map[_mask] = _comp_top[_lbl]
        _obj_left_map[_mask] = _comp_left[_lbl]
        _obj_bottom_map[_mask] = _comp_bottom[_lbl]
        _obj_right_map[_mask] = _comp_right[_lbl]

    OBJ_COUNT = np.int64(len(_comp_size))
    MAX_OBJ_SIZE = np.int64(max(_comp_size.values())) if _comp_size else np.int64(0)
    _obj_maps = {
        "obj_id": _label_map, "obj_size": _obj_size_map, "obj_color": _obj_color_map,
        "obj_top": _obj_top_map, "obj_left": _obj_left_map,
        "obj_bottom": _obj_bottom_map, "obj_right": _obj_right_map,
    }

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
        if op == "inp_r": return IR
        if op == "inp_c": return IC
        if op == "pass_num": return PN
        if op == "const": return np.int64(node[1])
        if op == "mode_color": return MODE_COLOR
        if op == "obj_count": return OBJ_COUNT
        if op == "max_obj_size": return MAX_OBJ_SIZE
        if op == "lib":
            sub = (library or {}).get(node[1])
            return _eval(sub) if sub is not None else np.int64(0)
        # Functions
        if op == "total_count":
            v = _eval(node[1])
            v_int = np.clip(np.asarray(v).flat[0], 0, NUM_COLORS - 1).astype(int)
            return np.int64(global_hist[v_int])
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
        if op == "n_count":
            c_val = _eval(node[3])
            v_int = int(np.clip(np.asarray(c_val).flat[0], 0, NUM_COLORS - 1))
            nc = _get_n_count(v_int)
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, cur_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, cur_c - 1).astype(int)
            return nc[ri, ci]
        if op == "row_count":
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, cur_r - 1).astype(int)
            vi = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, NUM_COLORS - 1).astype(int)
            return row_hist[ri.ravel(), vi.ravel()].reshape(o_r, o_c)
        if op == "col_count":
            ci = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, cur_c - 1).astype(int)
            vi = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, NUM_COLORS - 1).astype(int)
            return col_hist[ci.ravel(), vi.ravel()].reshape(o_r, o_c)
        if op in _obj_maps:
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, orig_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, orig_c - 1).astype(int)
            return _obj_maps[op][ri, ci]
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

    for p in range(passes):
        result = _evaluate_once(tree, g_current, g_original, o_shape, library, pass_num=p)
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
        if op in ("r", "c", "max_r", "max_c", "inp_r", "inp_c", "pass_num", "mode_color",
                  "obj_count", "max_obj_size"):
            terms = [("r",), ("c",), ("max_r",), ("max_c",), ("inp_r",), ("inp_c",),
                     ("pass_num",), ("mode_color",), ("obj_count",), ("max_obj_size",),
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
