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
                n_count8(r, c, v)      (count of 8-neighbors with color v)
  Row/column:   row_count(r, v)        (count of color v in row r)
                col_count(c, v)        (count of color v in column c)
  Global:       total_count(v)         (count of color v in input)
                mode_color             (most common color in input)
  Object (4-conn): obj_id/size/color/top/left/bottom/right(r, c)
                obj_count, max_obj_size
  Object (8-conn): obj8_id/size/color/top/left/bottom/right(r, c)
                obj8_count, max_obj8_size

Multi-pass: when passes > 1, the tree is applied repeatedly. Each pass reads
its own output from the previous pass via `get`, while `inp` always reads the
original input. This enables information propagation (cellular automata),
flood-fill, and iterative refinement — all from the same atomic primitives.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

# Domain-determined (not tunable — these follow from the problem definition)
NUM_COLORS = 10                 # ARC uses colors 0–9
NUM_TERMINALS = 13 + NUM_COLORS  # r, c, max_r, max_c, inp_r, inp_c, pass_num, mode_color, obj_count, max_obj_size, obj8_count, max_obj8_size + colors
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
MUTATE_WEIGHTS = {"subtree": 4, "point": 3, "hoist": 2, "graft": 3}
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
    "n_count8": 3,                     # count of 8-neighbors with color v at (r, c)
    "row_count": 2,                    # count of cells in row r with color v
    "col_count": 2,                    # count of cells in col c with color v
    "total_count": 1,                  # count of all cells with color v in input
    # Object (connected component) primitives — computed on g_original via BFS
    "obj_id": 2, "obj_size": 2, "obj_color": 2,       # 4-connectivity
    "obj_top": 2, "obj_left": 2, "obj_bottom": 2, "obj_right": 2,
    "obj8_id": 2, "obj8_size": 2, "obj8_color": 2,    # 8-connectivity
    "obj8_top": 2, "obj8_left": 2, "obj8_bottom": 2, "obj8_right": 2,
}

_OBJ_OPS = {"obj_id", "obj_size", "obj_color", "obj_top", "obj_left", "obj_bottom", "obj_right"}
_OBJ8_OPS = {"obj8_id", "obj8_size", "obj8_color", "obj8_top", "obj8_left", "obj8_bottom", "obj8_right"}
_OBJ_TERMINALS = {"obj_count", "max_obj_size"}
_OBJ8_TERMINALS = {"obj8_count", "max_obj8_size"}

def uses_obj(tree):
    """Check if tree references any obj_* op or terminal (4-conn)."""
    op = tree[0]
    if op in _OBJ_OPS or op in _OBJ_TERMINALS:
        return True
    if op == "lib":
        return True  # conservative: library entries might use obj
    if op in ARITY:
        return any(uses_obj(tree[i]) for i in range(1, 1 + ARITY[op]))
    return False

def uses_obj8(tree):
    """Check if tree references any obj8_* op or terminal (8-conn)."""
    op = tree[0]
    if op in _OBJ8_OPS or op in _OBJ8_TERMINALS:
        return True
    if op == "lib":
        return True
    if op in ARITY:
        return any(uses_obj8(tree[i]) for i in range(1, 1 + ARITY[op]))
    return False


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
        if ch == 10: return ("obj8_count",)
        if ch == 11: return ("max_obj8_size",)
        return ("const", ch - 12)
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

_ccl_cache = {}   # (grid_bytes, conn) -> (label_map, obj_maps, OBJ_COUNT, MAX_OBJ_SIZE)

_NEIGHBORS_4 = ((-1,0),(1,0),(0,-1),(0,1))
_NEIGHBORS_8 = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

def _get_ccl(g_original, connectivity=4):
    """Compute and cache connected component labeling for a grid.

    connectivity: 4 for cardinal-only, 8 for cardinal+diagonal.
    """
    key = (g_original.tobytes(), connectivity)
    if key not in _ccl_cache:
        neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
        prefix = "obj" if connectivity == 4 else "obj8"
        orig_r, orig_c = g_original.shape
        _label_map = np.zeros((orig_r, orig_c), dtype=np.int64)
        _comp_size = {}
        _comp_color = {}
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
                    for _dr, _dc in neighbors:
                        _nr, _nc = _qr+_dr, _qc+_dc
                        if 0 <= _nr < orig_r and 0 <= _nc < orig_c and _label_map[_nr, _nc] == 0 and g_original[_nr, _nc] == _color:
                            _label_map[_nr, _nc] = _lbl
                            _queue.append((_nr, _nc))
                _comp_size[_lbl] = _sz
                _comp_color[_lbl] = _color
                _comp_top[_lbl] = _t; _comp_left[_lbl] = _l
                _comp_bottom[_lbl] = _b; _comp_right[_lbl] = _ri

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
            f"{prefix}_id": _label_map, f"{prefix}_size": _obj_size_map,
            f"{prefix}_color": _obj_color_map,
            f"{prefix}_top": _obj_top_map, f"{prefix}_left": _obj_left_map,
            f"{prefix}_bottom": _obj_bottom_map, f"{prefix}_right": _obj_right_map,
        }
        _ccl_cache[key] = (_label_map, _obj_maps, OBJ_COUNT, MAX_OBJ_SIZE)
        if len(_ccl_cache) > 500:
            _ccl_cache.pop(next(iter(_ccl_cache)))
    return _ccl_cache[key]


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

    _n_count8_cache = {}
    def _get_n_count8(v):
        v = int(v) % NUM_COLORS
        if v not in _n_count8_cache:
            mask = (g_current == v).astype(np.int64)
            count = np.zeros_like(mask)
            if cur_r > 1:
                count[1:, :] += mask[:-1, :]
                count[:-1, :] += mask[1:, :]
            if cur_c > 1:
                count[:, 1:] += mask[:, :-1]
                count[:, :-1] += mask[:, 1:]
            # Diagonals
            if cur_r > 1 and cur_c > 1:
                count[1:, 1:] += mask[:-1, :-1]    # top-left
                count[1:, :-1] += mask[:-1, 1:]    # top-right
                count[:-1, 1:] += mask[1:, :-1]    # bottom-left
                count[:-1, :-1] += mask[1:, 1:]    # bottom-right
            _n_count8_cache[v] = count
        return _n_count8_cache[v]

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

    # ── Connected component labeling (cached, only if tree uses obj_* ops) ──
    _obj_maps = {}
    OBJ_COUNT = np.int64(0)
    MAX_OBJ_SIZE = np.int64(0)
    OBJ8_COUNT = np.int64(0)
    MAX_OBJ8_SIZE = np.int64(0)
    if uses_obj(tree):
        _label_map, _maps4, OBJ_COUNT, MAX_OBJ_SIZE = _get_ccl(g_original, 4)
        _obj_maps.update(_maps4)
    if uses_obj8(tree):
        _label_map8, _maps8, OBJ8_COUNT, MAX_OBJ8_SIZE = _get_ccl(g_original, 8)
        _obj_maps.update(_maps8)

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
        if op == "obj8_count": return OBJ8_COUNT
        if op == "max_obj8_size": return MAX_OBJ8_SIZE
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
        if op == "n_count8":
            c_val = _eval(node[3])
            v_int = int(np.clip(np.asarray(c_val).flat[0], 0, NUM_COLORS - 1))
            nc8 = _get_n_count8(v_int)
            ri = np.clip(np.broadcast_to(np.asarray(a), (o_r, o_c)), 0, cur_r - 1).astype(int)
            ci = np.clip(np.broadcast_to(np.asarray(b), (o_r, o_c)), 0, cur_c - 1).astype(int)
            return nc8[ri, ci]
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


def _random_condition():
    """Generate a biased-random condition for graft mutation."""
    R, C = ("r",), ("c",)
    k = np.random.randint(NUM_COLORS)
    K = ("const", k)
    k2 = np.random.randint(1, 5)
    K2 = ("const", k2)
    v = np.random.randint(NUM_COLORS)
    V = ("const", v)

    if np.random.random() < 0.6:
        # Template-based condition
        templates = [
            # Position
            ("gt", R, K),
            ("gt", C, K),
            ("eq", ("mod", R, ("const", 2)), ("const", 0)),
            ("eq", ("mod", C, ("const", 2)), ("const", 0)),
            ("eq", ("mod", ("add", R, C), ("const", 2)), ("const", 0)),
            # Value
            ("eq", ("get", R, C), K),
            ("gt", ("get", R, C), ("const", 0)),
            ("eq", ("inp", R, C), K),
            # Neighbor
            ("gt", ("n_count", R, C, V), K2),
            ("eq", ("n_count", R, C, V), ("const", 0)),
            ("gt", ("n_count8", R, C, V), K2),
            ("eq", ("n_count8", R, C, V), ("const", 0)),
            # Object (4-conn)
            ("gt", ("obj_size", R, C), K2),
            ("eq", ("obj_color", R, C), K),
            ("eq", ("obj_size", R, C), K2),
            # Object (8-conn)
            ("gt", ("obj8_size", R, C), K2),
            ("eq", ("obj8_color", R, C), K),
            ("eq", ("obj8_size", R, C), K2),
            # Row/col
            ("gt", ("row_count", R, V), ("const", 0)),
            ("gt", ("col_count", C, V), ("const", 0)),
        ]
        return templates[np.random.randint(len(templates))]
    else:
        return random_tree(max_depth=3)


def mutate(tree, library=None):
    """Apply one random mutation: subtree, point, hoist, or graft."""
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
                  "obj_count", "max_obj_size", "obj8_count", "max_obj8_size"):
            terms = [("r",), ("c",), ("max_r",), ("max_c",), ("inp_r",), ("inp_c",),
                     ("pass_num",), ("mode_color",), ("obj_count",), ("max_obj_size",),
                     ("obj8_count",), ("max_obj8_size",),
                     ("const", np.random.randint(NUM_COLORS))]
            return replace(tree, path, terms[np.random.randint(len(terms))])
        if op in ARITY and ARITY[op] == 2:
            bin_ops = [k for k, v in ARITY.items() if v == 2]
            new_op = bin_ops[np.random.randint(len(bin_ops))]
            return replace(tree, path, (new_op, node[1], node[2]))
        return tree

    if r < MUTATE_WEIGHTS["subtree"] + MUTATE_WEIGHTS["point"] + MUTATE_WEIGHTS["hoist"]:
        # hoist: replace tree with one of its subtrees
        if len(subs) > 1:
            _, sub = subs[np.random.randint(1, len(subs))]
            return sub
        return tree

    # graft: wrap a random subtree in a conditional
    remaining_depth = MAX_TREE_DEPTH - len(path) - 2
    if remaining_depth < 1:
        return tree
    condition = _random_condition()
    alternative = random_tree(max_depth=min(remaining_depth, 3), library=library)
    if np.random.random() < 0.5:
        grafted = ("if", condition, node, alternative)
    else:
        grafted = ("if", condition, alternative, node)
    result = replace(tree, path, grafted)
    return result if depth(result) <= MAX_TREE_DEPTH else tree


def tournament(population, fitnesses, k=TOURNAMENT_K):
    """Select the fittest of k random individuals."""
    idxs = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    return population[max(idxs, key=lambda i: fitnesses[i])]
