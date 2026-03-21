"""
Evolutionary program synthesis for ARC-AGI.

Uses genetic programming to evolve cell-level programs from atomic primitives.
Programs compute f(grid, r, c) → color for each output cell, optionally applied
for multiple passes (enabling cellular-automata-like information propagation).

Evolution discovers compositions like flip, rotate, recolor, flood-fill, and
conditional logic. Useful subtrees are abstracted into a reusable library,
enabling compounding across tasks.
"""

import json, sys, os, numpy as np
from pathlib import Path
import gp

# ── Constants ────────────────────────────────────────────────────────

# Search budget (base values — actual budget scales continuously with fitness)
POP_SIZE_BASE = 120             # population at fitness=0; doubles at fitness=1
GENERATIONS_BASE = 20           # generations at fitness=0; doubles at fitness=1
ELITE_FRACTION = 5              # keep top 1/N of population as elite

# Fitness function
CHANGED_CELL_WEIGHT = 3.0       # changed cells weighted higher to beat identity plateau
PARSIMONY_PENALTY = 0.002       # fitness cost per tree node (prefer simpler programs)

# Abstraction: promote subtrees shared across solvers
MIN_SUBTREE_SIZE = 3            # below this, subtrees are too trivial to abstract
MAX_SUBTREE_SIZE = 20           # above this, subtrees are too specific to reuse
MIN_SHARED_SOLVERS = 2          # must appear in at least this many solver programs

# Multi-pass: fraction of population initialized with passes > 1
MULTIPASS_FRACTION = 0.2        # 20% of population starts with 2-3 passes


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


def fitness(tree, examples, library=None, passes=1):
    """Weighted cell accuracy with parsimony pressure."""
    total = 0.0
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library, passes=passes)
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


def solves(tree, examples, library=None, passes=1):
    """Check if tree produces exact output for all examples."""
    for inp, out in examples:
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp, out_a.shape, library, passes=passes)
        if got is None or not np.array_equal(got, out_a):
            return False
    return True


def simplify(tree, examples, library=None, passes=1):
    """Greedily replace subtrees with simpler equivalents."""
    if not solves(tree, examples, library, passes):
        return tree
    R, C = ("r",), ("c",)
    replacements = [
        R, C, ("max_r",), ("max_c",), ("inp_r",), ("inp_c",), ("mode_color",),
        ("obj_count",), ("max_obj_size",), ("obj8_count",), ("max_obj8_size",),
        IDENTITY, ("get", C, R),
        ("inp", ("r",), ("c",)),                                   # inp(r,c)
        ("sub", ("sub", ("max_r",), ("const", 1)), R),
        ("sub", ("sub", ("max_c",), ("const", 1)), C),
        ("inp", ("mod", ("r",), ("inp_r",)), ("mod", ("c",), ("inp_c",))),  # tiling
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
                if solves(candidate, examples, library, passes):
                    tree = candidate
                    changed = True
                    break
            if changed:
                break
    return tree


# ── Seeds ────────────────────────────────────────────────────────────

def make_seeds():
    """Generate initial program templates expressed from atomic primitives.

    Returns list of (tree, passes) tuples.
    Categories:
      - Geometric remaps (7): identity, transpose, flips, rotations
      - Shifts (8): circular shifts by 1-4 in each axis
      - Recolors (90): if cell==X, change to Y (all color pairs)
      - Multi-pass templates: neighbor-based rules for iterative computation
    """
    R, C, MR, MC, ONE = ("r",), ("c",), ("max_r",), ("max_c",), ("const", 1)
    flip_r = ("sub", ("sub", MR, ONE), R)
    flip_c = ("sub", ("sub", MC, ONE), C)
    seeds = []

    # Single-pass geometric transforms
    for tree in [
        IDENTITY,
        ("get", C, R),                                      # transpose
        ("get", flip_r, C),                                  # flip vertical
        ("get", R, flip_c),                                  # flip horizontal
        ("get", flip_r, flip_c),                             # rotate 180
        ("get", C, flip_r),                                  # rotate cw
        ("get", flip_c, R),                                  # rotate ccw
    ]:
        seeds.append((tree, 1))

    # Shifts by 1-4
    for k in range(1, 5):
        K = ("const", k)
        seeds.append((("get", ("mod", ("add", R, K), MR), C), 1))
        seeds.append((("get", R, ("mod", ("add", C, K), MC)), 1))

    # Recolors: if cell == x, change to y
    for x in range(gp.NUM_COLORS):
        for y in range(gp.NUM_COLORS):
            if x != y:
                seeds.append((
                    ("if", ("eq", ("get", R, C), ("const", x)),
                     ("const", y), ("get", R, C)),
                    1
                ))
    # Color swaps: if x→y, also y→x
    for x in range(gp.NUM_COLORS):
        for y in range(x + 1, gp.NUM_COLORS):
            seeds.append((
                ("if", ("eq", ("get", R, C), ("const", x)), ("const", y),
                 ("if", ("eq", ("get", R, C), ("const", y)), ("const", x),
                  ("get", R, C))),
                1
            ))
    # Non-zero recolors: if cell != 0, change to color
    for y in range(1, gp.NUM_COLORS):
        seeds.append((
            ("if", ("get", R, C), ("const", y), ("get", R, C)),
            1
        ))
    # Conditional geometry: if cell == X, read from flipped position
    for x in range(gp.NUM_COLORS):
        # if cell == x, flip_h, else identity
        seeds.append((
            ("if", ("eq", ("get", R, C), ("const", x)),
             ("get", R, flip_c), ("get", R, C)),
            1
        ))
        # if cell == x, flip_v, else identity
        seeds.append((
            ("if", ("eq", ("get", R, C), ("const", x)),
             ("get", flip_r, C), ("get", R, C)),
            1
        ))
    # Zero-fill: if cell == 0, read from neighbor (gravity-like, 2 passes)
    seeds.append((("if", ("get", R, C), ("get", R, C),
                   ("get", ("sub", R, ONE), C)), 2))
    seeds.append((("if", ("get", R, C), ("get", R, C),
                   ("get", ("add", R, ONE), C)), 2))
    # Row/column broadcast: fill output from a single row or column
    for k in range(5):
        K = ("const", k)
        seeds.append((("get", K, C), 1))        # broadcast row k
        seeds.append((("get", R, K), 1))        # broadcast column k
    # Read from edges
    seeds.append((("get", R, ("const", 0)), 1))              # first column
    seeds.append((("get", R, ("sub", MC, ONE)), 1))           # last column
    seeds.append((("get", ("const", 0), C), 1))               # first row
    seeds.append((("get", ("sub", MR, ONE), C), 1))            # last row
    # Conditional: if cell==0, read from first row/column
    seeds.append((("if", ("get", R, C), ("get", R, C), ("get", ("const", 0), C)), 1))
    seeds.append((("if", ("get", R, C), ("get", R, C), ("get", R, ("const", 0))), 1))
    # Tiling: repeat a k×k tile across the grid
    for k in range(2, 6):
        K = ("const", k)
        seeds.append((("get", ("mod", R, K), ("mod", C, K)), 1))
    # Modular position patterns (checkerboard-like)
    seeds.append((("mod", ("add", R, C), ("const", 2)), 1))    # checkerboard 0/1
    seeds.append((("if", ("mod", ("add", R, C), ("const", 2)),
                   ("get", R, C), ("const", 0)), 1))           # checkerboard mask

    IR, IC = ("inp_r",), ("inp_c",)

    # (Double recolors moved to task-specific hypothesis generation)

    # ── Tiling with input dimensions ──────────────────────────────
    # Tile input into larger output: inp(r % inp_r, c % inp_c)
    seeds.append((("inp", ("mod", R, IR), ("mod", C, IC)), 1))
    # Tile + flip alternating: inp(r%inp_r, inp_c-1-(c%inp_c)) for even tiles
    seeds.append((("inp", ("mod", R, IR),
                   ("if", ("mod", ("sub", R, ("mod", R, IR)), IR),
                    ("sub", ("sub", IC, ONE), ("mod", C, IC)),
                    ("mod", C, IC))), 1))
    # Double horizontally: inp(r, c % inp_c)
    seeds.append((("inp", R, ("mod", C, IC)), 1))
    # Double vertically: inp(r % inp_r, c)
    seeds.append((("inp", ("mod", R, IR), C), 1))

    # ── Cropping/extraction seeds ─────────────────────────────────
    # Read from offset position in input (extract subgrid)
    for offset_r in range(5):
        for offset_c in range(5):
            if offset_r == 0 and offset_c == 0:
                continue
            seeds.append((("inp", ("add", R, ("const", offset_r)),
                                  ("add", C, ("const", offset_c))), 1))

    # ── Derived primitive seeds ─────────────────────────────────────

    # Background removal: if cell == mode_color, set to 0
    seeds.append((("if", ("eq", ("get", R, C), ("mode_color",)),
                   ("const", 0), ("get", R, C)), 1))
    # Background removal: if cell == mode_color, keep 0; else keep cell
    for y in range(1, gp.NUM_COLORS):
        seeds.append((("if", ("eq", ("get", R, C), ("mode_color",)),
                       ("const", y), ("get", R, C)), 1))

    # Edge detection: keep cells with 3+ same-color neighbors
    seeds.append((("if", ("gt", ("n_count", R, C, ("get", R, C)), ("const", 2)),
                   ("get", R, C), ("const", 0)), 1))
    # Edge detection: keep cells with <4 same-color neighbors (boundary cells)
    seeds.append((("if", ("gt", ("const", 4), ("n_count", R, C, ("get", R, C))),
                   ("get", R, C), ("const", 0)), 1))
    # Interior detection: cells fully surrounded (4 same-color neighbors)
    seeds.append((("if", ("eq", ("n_count", R, C, ("get", R, C)), ("const", 4)),
                   ("get", R, C), ("const", 0)), 1))
    # Flood fill via n_count: if any neighbor has color X, become X (multi-pass)
    for color in range(1, 5):
        CV = ("const", color)
        seeds.append((("if", ("n_count", R, C, CV),
                       CV, ("get", R, C)), 2))
        seeds.append((("if", ("n_count", R, C, CV),
                       CV, ("get", R, C)), 3))
    # Neighbor-based: if cell is 0 but has non-zero neighbor, take neighbor color
    seeds.append((("if", ("get", R, C), ("get", R, C),
                   ("if", ("n_count", R, C, ("const", 0)),
                    ("get", R, C), ("const", 0))), 2))

    # Row-based patterns: if row has any of color X, fill row with Y
    for x in range(1, 5):
        seeds.append((("if", ("row_count", R, ("const", x)),
                       ("const", x), ("get", R, C)), 1))
        seeds.append((("if", ("row_count", R, ("const", x)),
                       ("const", x), ("const", 0)), 1))
    # Column-based patterns: if col has any of color X, fill col with Y
    for x in range(1, 5):
        seeds.append((("if", ("col_count", C, ("const", x)),
                       ("const", x), ("get", R, C)), 1))
        seeds.append((("if", ("col_count", C, ("const", x)),
                       ("const", x), ("const", 0)), 1))
    # Row/col intersection: color cell if both row and col have color X
    for x in range(1, 5):
        seeds.append((("if", ("gt", ("row_count", R, ("const", x)), ("const", 0)),
                       ("if", ("gt", ("col_count", C, ("const", x)), ("const", 0)),
                        ("const", x), ("get", R, C)),
                       ("get", R, C)), 1))

    # Multi-pass templates: neighbor-aware rules (2-3 passes)
    # "If any neighbor has color X, become X" — flood-fill-like
    for color in range(1, 5):
        CV = ("const", color)
        neighbor_check = ("if",
            ("eq", ("get", ("add", R, ONE), C), CV), CV,
            ("if", ("eq", ("get", ("sub", R, ONE), C), CV), CV,
             ("if", ("eq", ("get", R, ("add", C, ONE)), CV), CV,
              ("if", ("eq", ("get", R, ("sub", C, ONE)), CV), CV,
               ("get", R, C)))))
        seeds.append((neighbor_check, 2))
        seeds.append((neighbor_check, 3))

    # "Copy non-zero from input, propagate via neighbors" (2 passes)
    propagate = ("if", ("inp", R, C), ("inp", R, C),
                 ("if", ("get", ("add", R, ONE), C), ("get", ("add", R, ONE), C),
                  ("if", ("get", R, ("add", C, ONE)), ("get", R, ("add", C, ONE)),
                   ("get", R, C))))
    seeds.append((propagate, 2))
    seeds.append((propagate, 3))

    # Propagate from all 4 directions (more complete flood fill)
    propagate_4 = ("if", ("inp", R, C), ("inp", R, C),
                   ("if", ("get", ("add", R, ONE), C), ("get", ("add", R, ONE), C),
                    ("if", ("get", ("sub", R, ONE), C), ("get", ("sub", R, ONE), C),
                     ("if", ("get", R, ("add", C, ONE)), ("get", R, ("add", C, ONE)),
                      ("if", ("get", R, ("sub", C, ONE)), ("get", R, ("sub", C, ONE)),
                       ("get", R, C))))))
    for p in [2, 3, 5]:
        seeds.append((propagate_4, p))

    # Fill zeros using n_count: if cell is 0 and has neighbors of color X, become X
    for color in range(1, 5):
        CV = ("const", color)
        fill_near = ("if", ("get", R, C), ("get", R, C),
                     ("if", ("n_count", R, C, CV), CV, ("get", R, C)))
        for p in [2, 3, 5]:
            seeds.append((fill_near, p))

    # Symmetric fill: if cell is 0, try reading from mirror position
    sym_h = ("if", ("get", R, C), ("get", R, C),
             ("get", R, ("sub", ("sub", MC, ONE), C)))
    sym_v = ("if", ("get", R, C), ("get", R, C),
             ("get", ("sub", ("sub", MR, ONE), R), C))
    seeds.append((sym_h, 1))
    seeds.append((sym_v, 1))
    seeds.append((sym_h, 2))
    seeds.append((sym_v, 2))

    # Majority fill: if cell is 0, become mode_color
    seeds.append((("if", ("get", R, C), ("get", R, C), ("mode_color",)), 1))

    # ── Object (connected component) seeds ─────────────────────────────
    # Keep largest object
    seeds.append((("if", ("eq", ("obj_size", R, C), ("max_obj_size",)),
                   ("get", R, C), ("const", 0)), 1))
    # Remove largest object
    seeds.append((("if", ("eq", ("obj_size", R, C), ("max_obj_size",)),
                   ("const", 0), ("get", R, C)), 1))
    # Keep objects of size N (N=1..5)
    for n in range(1, 6):
        seeds.append((("if", ("eq", ("obj_size", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Remove isolated pixels (size==1)
    seeds.append((("if", ("gt", ("obj_size", R, C), ("const", 1)),
                   ("get", R, C), ("const", 0)), 1))
    # Keep only object N (N=1..3)
    for n in range(1, 4):
        seeds.append((("if", ("eq", ("obj_id", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Recolor all objects to single color
    for color in range(1, 5):
        seeds.append((("if", ("obj_id", R, C), ("const", color), ("const", 0)), 1))
    # Keep objects touching top edge (obj_top == 0)
    seeds.append((("if", ("eq", ("obj_top", R, C), ("const", 0)),
                   ("get", R, C), ("const", 0)), 1))
    # Keep objects touching left edge (obj_left == 0)
    seeds.append((("if", ("eq", ("obj_left", R, C), ("const", 0)),
                   ("get", R, C), ("const", 0)), 1))
    # Remove objects touching top/left edge
    seeds.append((("if", ("eq", ("obj_top", R, C), ("const", 0)),
                   ("const", 0), ("get", R, C)), 1))
    seeds.append((("if", ("eq", ("obj_left", R, C), ("const", 0)),
                   ("const", 0), ("get", R, C)), 1))
    # Fill bounding box of each object with its color
    seeds.append((("if", ("obj_id", R, C), ("obj_color", R, C), ("const", 0)), 1))
    # Keep objects of specific color (color 1-5)
    for color in range(1, 6):
        seeds.append((("if", ("eq", ("obj_color", R, C), ("const", color)),
                       ("get", R, C), ("const", 0)), 1))
        # Remove objects of specific color
        seeds.append((("if", ("eq", ("obj_color", R, C), ("const", color)),
                       ("const", 0), ("get", R, C)), 1))
    # Keep largest + recolor to specific color
    for color in range(1, 5):
        seeds.append((("if", ("eq", ("obj_size", R, C), ("max_obj_size",)),
                       ("const", color), ("const", 0)), 1))
    # Keep non-largest objects
    seeds.append((("if", ("eq", ("obj_size", R, C), ("max_obj_size",)),
                   ("const", 0), ("if", ("obj_id", R, C), ("get", R, C), ("const", 0))), 1))
    # Size thresholding: keep objects with size > N
    for n in range(2, 6):
        seeds.append((("if", ("gt", ("obj_size", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Remove objects touching bottom/right edge
    seeds.append((("if", ("eq", ("obj_bottom", R, C), ("sub", MR, ONE)),
                   ("const", 0), ("get", R, C)), 1))
    seeds.append((("if", ("eq", ("obj_right", R, C), ("sub", MC, ONE)),
                   ("const", 0), ("get", R, C)), 1))

    # ── 8-connectivity object seeds ──────────────────────────────────────
    # Mirror key obj seeds with obj8 variants for diagonal connectivity
    # Keep largest 8-connected object
    seeds.append((("if", ("eq", ("obj8_size", R, C), ("max_obj8_size",)),
                   ("get", R, C), ("const", 0)), 1))
    # Remove largest 8-connected object
    seeds.append((("if", ("eq", ("obj8_size", R, C), ("max_obj8_size",)),
                   ("const", 0), ("get", R, C)), 1))
    # Keep 8-connected objects of size N (N=1..5)
    for n in range(1, 6):
        seeds.append((("if", ("eq", ("obj8_size", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Remove isolated pixels (8-conn size==1)
    seeds.append((("if", ("gt", ("obj8_size", R, C), ("const", 1)),
                   ("get", R, C), ("const", 0)), 1))
    # Keep only 8-connected object N (N=1..3)
    for n in range(1, 4):
        seeds.append((("if", ("eq", ("obj8_id", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Recolor all 8-connected objects to single color
    for color in range(1, 5):
        seeds.append((("if", ("obj8_id", R, C), ("const", color), ("const", 0)), 1))
    # Keep 8-connected objects of specific color
    for color in range(1, 6):
        seeds.append((("if", ("eq", ("obj8_color", R, C), ("const", color)),
                       ("get", R, C), ("const", 0)), 1))
        seeds.append((("if", ("eq", ("obj8_color", R, C), ("const", color)),
                       ("const", 0), ("get", R, C)), 1))
    # Size thresholding (8-conn)
    for n in range(2, 6):
        seeds.append((("if", ("gt", ("obj8_size", R, C), ("const", n)),
                       ("get", R, C), ("const", 0)), 1))
    # Fill bounding box of each 8-connected object with its color
    seeds.append((("if", ("obj8_id", R, C), ("obj8_color", R, C), ("const", 0)), 1))

    # ── n_count8 (8-directional neighbor) seeds ──────────────────────────
    # Edge detection with 8-connectivity: cells with <8 same-color neighbors
    seeds.append((("if", ("gt", ("const", 8), ("n_count8", R, C, ("get", R, C))),
                   ("get", R, C), ("const", 0)), 1))
    # Interior detection: cells fully surrounded (8 same-color neighbors)
    seeds.append((("if", ("eq", ("n_count8", R, C, ("get", R, C)), ("const", 8)),
                   ("get", R, C), ("const", 0)), 1))
    # Diagonal-aware flood fill: if any 8-neighbor has color X, become X
    for color in range(1, 5):
        CV = ("const", color)
        seeds.append((("if", ("n_count8", R, C, CV), CV, ("get", R, C)), 2))

    return seeds


# ── Search ───────────────────────────────────────────────────────────

def search_budget(prev_fitness):
    """Search effort scales continuously with fitness. Higher fitness = much more search."""
    scale = 1.0 + prev_fitness * 2  # steeper scaling for near-misses
    return int(POP_SIZE_BASE * scale), int(GENERATIONS_BASE * scale)


def search(examples, seed=None, seed_passes=1, library=None,
           pop_size=None, gens=None, transfer=None):
    """Evolve a (tree, passes) program for a single task.

    Population contains programs with varying pass counts. Evolution jointly
    optimizes the tree structure and number of passes. Transfer programs from
    other solved tasks are included as population seeds.
    Returns (best_tree, best_passes, best_fitness).
    """
    if pop_size is None or gens is None:
        pop_size, gens = POP_SIZE_BASE, GENERATIONS_BASE

    # Rank all seed templates by fitness, keep top ones for focused search
    all_seeds = list(make_seeds()) + list(task_hypotheses(examples))
    seed_fits = [(fitness(t, examples, library, p), (t, p)) for t, p in all_seeds]
    seed_fits.sort(reverse=True)
    # Keep top 40 seeds + all that solve the task
    top_seeds = [tp for _, tp in seed_fits[:40]]
    pop = list(top_seeds)

    # Transfer: include programs that solved other tasks + mutations
    if transfer:
        for tree, passes in transfer:
            pop.append((tree, passes))
            pop.append((gp.mutate(tree, library=library), passes))

    # Build from previous best (seed) via mutations
    n_mutants = pop_size // ELITE_FRACTION
    if seed is not None and seed != IDENTITY:
        pop.append((seed, seed_passes))
        for _ in range(n_mutants):
            pop.append((gp.mutate(seed, library=library), seed_passes))

    # Mutate the best seeds too — explore their neighborhood
    for i in range(min(10, len(top_seeds))):
        t, p = top_seeds[i]
        for _ in range(3):
            pop.append((gp.mutate(t, library=library), p))

    # Identity mutations
    for _ in range(n_mutants):
        pop.append((gp.mutate(IDENTITY, library=library), 1))

    # Fill rest with random trees (varied passes)
    pop_size = max(pop_size, len(pop) + 10)
    while len(pop) < pop_size:
        passes = np.random.choice([1, 1, 1, 1, 2, 2, 3])
        pop.append((gp.random_tree(max_depth=4, library=library), passes))

    # Track best
    best_t, best_p, best_f = IDENTITY, 1, fitness(IDENTITY, examples, library, 1)
    if seed is not None:
        sf = fitness(seed, examples, library, seed_passes)
        if sf > best_f:
            best_t, best_p, best_f = seed, seed_passes, sf

    patience = int(3 + best_f * 6)
    stale = 0
    for gen in range(gens):
        fits = [fitness(t, examples, library, p) for t, p in pop]
        improved = False
        for (t, p), f in zip(pop, fits):
            if f > best_f:
                best_t, best_p, best_f = t, p, f
                improved = True
        if improved:
            stale = 0
            patience = int(3 + best_f * 6)
        else:
            stale += 1
        if solves(best_t, examples, library, best_p):
            return simplify(best_t, examples, library, best_p), best_p, 1.0
        if stale > patience:
            break

        # Next generation
        ranked = sorted(zip(fits, pop), reverse=True)
        elite = [(t, p) for _, (t, p) in ranked[:pop_size // ELITE_FRACTION]]
        new_pop = list(elite) + [(IDENTITY, 1), (best_t, best_p)]

        n_best_mutants = pop_size // (ELITE_FRACTION * 2)
        for _ in range(n_best_mutants):
            new_pop.append((gp.mutate(best_t, library=library), best_p))

        while len(new_pop) < pop_size:
            # Tournament selection by index (population items are tuples)
            k = min(gp.TOURNAMENT_K, len(pop))
            idx1 = max(np.random.choice(len(pop), size=k, replace=False),
                       key=lambda i: fits[i])
            idx2 = max(np.random.choice(len(pop), size=k, replace=False),
                       key=lambda i: fits[i])
            t1, p1 = pop[idx1]
            t2, p2 = pop[idx2]

            child_tree = gp.crossover(t1, t2)
            child_tree = gp.mutate(child_tree, library=library)
            # Inherit passes from fitter parent; occasionally mutate ±1
            child_passes = p1 if fits[idx1] >= fits[idx2] else p2
            if np.random.random() < 0.1:
                child_passes += np.random.choice([-1, 1])
                child_passes = max(1, min(gp.MAX_PASSES, child_passes))
            new_pop.append((child_tree, child_passes))

        pop = new_pop

    # Iterative refinement: try single-node edits, chain improvements
    if best_f > 0.5:
        for _ in range(4):  # up to 4 rounds of refinement
            t2, f2 = refine(best_t, examples, library, best_p)
            if f2 <= best_f:
                break
            best_t, best_f = t2, f2
            if best_f >= 1.0:
                break
    # Error-driven refinement for high-fitness near-misses
    if 0.5 < best_f < 1.0:
        t2, f2 = error_driven_refine(best_t, examples, library, best_p)
        if f2 > best_f:
            best_t, best_f = t2, f2
    return best_t, best_p, best_f


def refine(tree, examples, library=None, passes=1):
    """Try all single-node edits of tree."""
    best_t, best_f = tree, fitness(tree, examples, library, passes)
    terminals = [("r",), ("c",), ("max_r",), ("max_c",), ("inp_r",), ("inp_c",), ("mode_color",),
                 ("obj_count",), ("max_obj_size",), ("obj8_count",), ("max_obj8_size",)]
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
                if solves(c, examples, library, passes):
                    return simplify(c, examples, library, passes), 1.0
                f = fitness(c, examples, library, passes)
                if f > best_f:
                    best_t, best_f = c, f
        elif gp.ARITY[op] == 2:
            for new_op in binary_ops:
                if new_op == op:
                    continue
                c = gp.replace(tree, path, (new_op, node[1], node[2]))
                if solves(c, examples, library, passes):
                    return simplify(c, examples, library, passes), 1.0
                f = fitness(c, examples, library, passes)
                if f > best_f:
                    best_t, best_f = c, f
    return best_t, best_f


def refine_wrap(tree, examples, library=None, passes=1):
    """Try wrapping subtrees in conditionals using derived primitives.

    This catches cases where the base program is right for MOST cells but
    needs a spatial condition to handle a subset differently.
    """
    best_t, best_f = tree, fitness(tree, examples, library, passes)
    R, C = ("r",), ("c",)

    # Analyze what colors need to change
    inp0, out0 = np.asarray(examples[0][0]), np.asarray(examples[0][1])
    if inp0.shape != out0.shape:
        return best_t, best_f

    diff = inp0 != out0
    if not diff.any():
        return best_t, best_f

    # For each changed color, try wrapping tree output with n_count condition
    changed_from = set(inp0[diff].tolist())
    for fc in changed_from:
        for tv in set(out0[diff].tolist()):
            for thresh in range(5):
                # If tree output == fc and n_count condition, change to tv
                for cmp_op in ["gt", "eq"]:
                    wrapper = ("if", ("eq", tree, ("const", fc)),
                               ("if", (cmp_op, ("n_count", R, C, ("const", fc)),
                                       ("const", thresh)),
                                ("const", tv), tree),
                               tree)
                    if gp.depth(wrapper) <= gp.MAX_TREE_DEPTH:
                        c = wrapper
                        if solves(c, examples, library, passes):
                            return simplify(c, examples, library, passes), 1.0
                        f = fitness(c, examples, library, passes)
                        if f > best_f:
                            best_t, best_f = c, f

    # Try wrapping with obj_size conditions (4-conn and 8-conn)
    for fc in changed_from:
        for tv in set(out0[diff].tolist()):
            for sz_thresh in range(1, 6):
                for cmp_op in ["gt", "eq"]:
                    for obj_op in ["obj_size", "obj8_size"]:
                        wrapper = ("if", ("eq", tree, ("const", fc)),
                                   ("if", (cmp_op, (obj_op, R, C),
                                           ("const", sz_thresh)),
                                    ("const", tv), tree),
                                   tree)
                        if gp.depth(wrapper) <= gp.MAX_TREE_DEPTH:
                            if solves(wrapper, examples, library, passes):
                                return simplify(wrapper, examples, library, passes), 1.0
                            f = fitness(wrapper, examples, library, passes)
                            if f > best_f:
                                best_t, best_f = wrapper, f

    # Try wrapping with n_count8 conditions
    for fc in changed_from:
        for tv in set(out0[diff].tolist()):
            for thresh in range(5):
                for cmp_op in ["gt", "eq"]:
                    wrapper = ("if", ("eq", tree, ("const", fc)),
                               ("if", (cmp_op, ("n_count8", R, C, ("const", fc)),
                                       ("const", thresh)),
                                ("const", tv), tree),
                               tree)
                    if gp.depth(wrapper) <= gp.MAX_TREE_DEPTH:
                        if solves(wrapper, examples, library, passes):
                            return simplify(wrapper, examples, library, passes), 1.0
                        f = fitness(wrapper, examples, library, passes)
                        if f > best_f:
                            best_t, best_f = wrapper, f

    # Try wrapping internal subtrees of obj-based trees with spatial conditions
    # This handles cases like "keep largest object BUT only rows 3-6"
    for path, node in gp.subtrees(tree):
        if node == ("get", R, C) or (node[0] == "const" and node[1] != 0):
            for cond in [
                ("gt", R, ("obj_top", R, C)),     # r > obj_top (exclude top boundary)
                ("gt", C, ("obj_left", R, C)),     # c > obj_left
                ("gt", ("obj_bottom", R, C), R),   # r < obj_bottom
                ("gt", ("obj_right", R, C), C),    # c < obj_right
                ("gt", ("obj_size", R, C), ("const", 1)),
                ("gt", ("obj_size", R, C), ("const", 2)),
                ("gt", ("obj8_size", R, C), ("const", 1)),
                ("gt", ("obj8_size", R, C), ("const", 2)),
            ]:
                wrapped = ("if", cond, node, ("const", 0))
                c = gp.replace(tree, path, wrapped)
                if gp.depth(c) <= gp.MAX_TREE_DEPTH:
                    if solves(c, examples, library, passes):
                        return simplify(c, examples, library, passes), 1.0
                    f = fitness(c, examples, library, passes)
                    if f > best_f:
                        best_t, best_f = c, f

    return best_t, best_f


def error_driven_refine(tree, examples, library=None, passes=1):
    """Targeted fix for near-miss programs: find discriminating conditions at error cells.

    1. Compute error mask across all training examples
    2. Find conditions that are True at ALL error cells and False at ALL correct cells (or vice versa)
    3. Try wrapping tree (or subtrees) with if(condition, fix_value, tree)
    """
    if not solves(tree, examples[:1], library, passes) is False:
        # Quick check: if it already works on first example, nothing to do
        pass

    R, C = ("r",), ("c",)
    ONE = ("const", 1)
    MR, MC = ("max_r",), ("max_c",)
    flip_r = ("sub", ("sub", MR, ONE), R)
    flip_c = ("sub", ("sub", MC, ONE), C)

    # Collect error info across all examples
    all_conditions_work = True  # track if conditions are consistent across examples
    candidate_fixes = []  # list of (condition_tree, fix_tree) that work on ALL examples

    # First, gather error masks and properties for each example
    example_data = []
    for inp, out in examples:
        inp_a = np.asarray(inp)
        out_a = np.asarray(out)
        got = gp.evaluate(tree, inp_a, out_a.shape, library, passes=passes)
        if got is None or got.shape != out_a.shape:
            return tree, fitness(tree, examples, library, passes)
        error_mask = got != out_a
        if not error_mask.any():
            example_data.append(None)
            continue
        example_data.append({
            "inp": inp_a, "out": out_a, "got": got,
            "error_mask": error_mask,
            "error_rows": set(np.where(error_mask)[0].tolist()),
            "error_cols": set(np.where(error_mask)[1].tolist()),
            "expected_at_errors": out_a[error_mask],
            "got_at_errors": got[error_mask],
        })

    if all(d is None for d in example_data):
        return tree, 1.0  # already solves

    # Build candidate conditions based on first non-None example
    ref = next(d for d in example_data if d is not None)
    h, w = ref["out"].shape
    conditions = []  # list of (name, condition_tree)

    # Row conditions
    for k in sorted(ref["error_rows"])[:5]:
        conditions.append((f"r=={k}", ("eq", R, ("const", k))))
    for k in sorted(ref["error_rows"])[:3]:
        conditions.append((f"r>{k}", ("gt", R, ("const", k))))
    # Also try boundaries: rows just before/after error region
    err_rows = sorted(ref["error_rows"])
    if err_rows:
        if err_rows[0] > 0:
            conditions.append((f"r>={err_rows[0]}", ("gt", R, ("const", err_rows[0] - 1))))
        conditions.append((f"r<={err_rows[-1]}", ("gt", ("const", err_rows[-1] + 1), R)))

    # Column conditions
    for k in sorted(ref["error_cols"])[:5]:
        conditions.append((f"c=={k}", ("eq", C, ("const", k))))
    for k in sorted(ref["error_cols"])[:3]:
        conditions.append((f"c>{k}", ("gt", C, ("const", k))))
    err_cols = sorted(ref["error_cols"])
    if err_cols:
        if err_cols[0] > 0:
            conditions.append((f"c>={err_cols[0]}", ("gt", C, ("const", err_cols[0] - 1))))
        conditions.append((f"c<={err_cols[-1]}", ("gt", ("const", err_cols[-1] + 1), C)))

    # Modular conditions
    conditions.append(("(r+c)%2==0", ("eq", ("mod", ("add", R, C), ("const", 2)), ("const", 0))))
    conditions.append(("(r+c)%2==1", ("eq", ("mod", ("add", R, C), ("const", 2)), ("const", 1))))
    conditions.append(("r%2==0", ("eq", ("mod", R, ("const", 2)), ("const", 0))))
    conditions.append(("c%2==0", ("eq", ("mod", C, ("const", 2)), ("const", 0))))

    # Input value conditions
    inp_colors = set(ref["inp"].flat)
    for k in inp_colors:
        conditions.append((f"inp=={k}", ("eq", ("inp", R, C), ("const", int(k)))))

    # Neighbor conditions
    for v in list(inp_colors)[:5]:
        for nk in range(5):
            conditions.append((f"nc({v})>{nk}", ("gt", ("n_count", R, C, ("const", int(v))), ("const", nk))))
            if nk <= 2:
                conditions.append((f"nc({v})=={nk}", ("eq", ("n_count", R, C, ("const", int(v))), ("const", nk))))

    # Object conditions (only evaluate if obj ops might help)
    try:
        obj_size_vals = gp.evaluate(("obj_size", R, C), ref["inp"], ref["out"].shape, library)
        obj_color_vals = gp.evaluate(("obj_color", R, C), ref["inp"], ref["out"].shape, library)
        obj_top_vals = gp.evaluate(("obj_top", R, C), ref["inp"], ref["out"].shape, library)
        obj_left_vals = gp.evaluate(("obj_left", R, C), ref["inp"], ref["out"].shape, library)
        if obj_size_vals is not None:
            for sz in sorted(set(obj_size_vals[obj_size_vals > 0].flat))[:5]:
                conditions.append((f"osz=={sz}", ("eq", ("obj_size", R, C), ("const", int(sz)))))
                conditions.append((f"osz>{sz}", ("gt", ("obj_size", R, C), ("const", int(sz)))))
        if obj_color_vals is not None:
            for oc in sorted(set(obj_color_vals[obj_color_vals > 0].flat))[:5]:
                conditions.append((f"ocol=={oc}", ("eq", ("obj_color", R, C), ("const", int(oc)))))
        if obj_top_vals is not None:
            conditions.append(("otop==0", ("eq", ("obj_top", R, C), ("const", 0))))
            conditions.append(("r==otop", ("eq", R, ("obj_top", R, C))))
        if obj_left_vals is not None:
            conditions.append(("oleft==0", ("eq", ("obj_left", R, C), ("const", 0))))
            conditions.append(("c==oleft", ("eq", C, ("obj_left", R, C))))
    except Exception:
        pass

    # Object conditions (8-connectivity)
    try:
        obj8_size_vals = gp.evaluate(("obj8_size", R, C), ref["inp"], ref["out"].shape, library)
        obj8_color_vals = gp.evaluate(("obj8_color", R, C), ref["inp"], ref["out"].shape, library)
        if obj8_size_vals is not None:
            for sz in sorted(set(obj8_size_vals[obj8_size_vals > 0].flat))[:5]:
                conditions.append((f"o8sz=={sz}", ("eq", ("obj8_size", R, C), ("const", int(sz)))))
                conditions.append((f"o8sz>{sz}", ("gt", ("obj8_size", R, C), ("const", int(sz)))))
        if obj8_color_vals is not None:
            for oc in sorted(set(obj8_color_vals[obj8_color_vals > 0].flat))[:5]:
                conditions.append((f"o8col=={oc}", ("eq", ("obj8_color", R, C), ("const", int(oc)))))
    except Exception:
        pass

    # n_count8 conditions
    for v in list(inp_colors)[:5]:
        for nk in range(1, 5):
            conditions.append((f"nc8({v})>{nk}", ("gt", ("n_count8", R, C, ("const", int(v))), ("const", nk))))

    # Row/col count conditions
    for v in list(inp_colors)[:4]:
        conditions.append((f"rc({v})>0", ("gt", ("row_count", R, ("const", int(v))), ("const", 0))))
        conditions.append((f"cc({v})>0", ("gt", ("col_count", C, ("const", int(v))), ("const", 0))))

    # Now evaluate each condition against error masks
    # For each condition, check if it discriminates errors from correct cells
    best_t, best_f = tree, fitness(tree, examples, library, passes)

    for _cname, cond_tree in conditions:
        # Check discrimination on each example
        cond_true_at_errors = True
        cond_false_at_correct = True
        cond_inv = True  # inverse: cond False at errors, True at correct
        cond_inv_correct = True

        for ed in example_data:
            if ed is None:
                continue
            cond_val = gp.evaluate(cond_tree, ed["inp"], ed["out"].shape, library)
            if cond_val is None:
                cond_true_at_errors = False
                cond_inv = False
                break
            cond_bool = cond_val != 0
            em = ed["error_mask"]
            if not cond_bool[em].all():
                cond_true_at_errors = False
            if cond_bool[~em].any():
                cond_false_at_correct = False
            # Inverse
            if cond_bool[em].any():
                cond_inv = False
            if not cond_bool[~em].all():
                cond_inv_correct = False

        # Try both orientations
        orientations = []
        if cond_true_at_errors and cond_false_at_correct:
            orientations.append(True)   # if(cond, fix, tree)
        if cond_inv and cond_inv_correct:
            orientations.append(False)  # if(cond, tree, fix)

        if not orientations:
            continue

        # Try fix values
        fix_values = [("const", i) for i in range(gp.NUM_COLORS)]
        fix_values.append(("inp", R, C))
        fix_values.append(("get", flip_r, C))
        fix_values.append(("get", R, flip_c))

        # Check if expected output at error cells is consistent
        consistent_expected = True
        expected_val = None
        for ed in example_data:
            if ed is None:
                continue
            unique_expected = set(ed["expected_at_errors"].flat)
            if len(unique_expected) == 1:
                v = unique_expected.pop()
                if expected_val is None:
                    expected_val = v
                elif expected_val != v:
                    consistent_expected = False
            else:
                consistent_expected = False

        for orient in orientations:
            for fix in fix_values:
                if orient:
                    candidate = ("if", cond_tree, fix, tree)
                else:
                    candidate = ("if", cond_tree, tree, fix)
                if gp.depth(candidate) > gp.MAX_TREE_DEPTH:
                    continue
                if solves(candidate, examples, library, passes):
                    return simplify(candidate, examples, library, passes), 1.0
                f = fitness(candidate, examples, library, passes)
                if f > best_f:
                    best_t, best_f = candidate, f

    # Also try subtree-level wrapping for smaller fixes
    for path, sub in gp.subtrees(tree):
        if not path or gp.size(sub) < 2:
            continue
        if len(path) + gp.depth(sub) + 3 > gp.MAX_TREE_DEPTH:
            continue
        # Check if this subtree produces consistent wrong values at error positions
        # Only try top-3 discriminating conditions (to keep cost bounded)
        for _cname, cond_tree in conditions[:15]:
            for fix_val in [("const", i) for i in range(gp.NUM_COLORS)][:5] + [("inp", R, C)]:
                wrapped = ("if", cond_tree, fix_val, sub)
                candidate = gp.replace(tree, path, wrapped)
                if gp.depth(candidate) > gp.MAX_TREE_DEPTH:
                    continue
                if solves(candidate, examples, library, passes):
                    return simplify(candidate, examples, library, passes), 1.0
                f = fitness(candidate, examples, library, passes)
                if f > best_f:
                    best_t, best_f = candidate, f
        # Limit subtree iterations for performance
        if gp.size(tree) > 15:
            break

    return best_t, best_f


# ── Abstraction ──────────────────────────────────────────────────────

def abstract(solvers, library):
    """Find subtrees shared by 2+ solvers and add to library."""
    if len(solvers) < MIN_SHARED_SOLVERS:
        return
    sub_sources = {}
    for i, (tree, _passes) in enumerate(solvers):
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

def task_hypotheses(examples):
    """Generate task-specific program hypotheses from input/output analysis.

    Examines the training examples to find consistent patterns and generates
    targeted seeds. This is principled — it's smarter search, not new primitives.
    """
    R, C = ("r",), ("c",)
    ONE = ("const", 1)
    hypotheses = []
    inp0, out0 = np.asarray(examples[0][0]), np.asarray(examples[0][1])

    # 1. Color mapping: if every cell of color X always maps to color Y
    if inp0.shape == out0.shape:
        color_map = {}
        consistent = True
        for inp, out in examples:
            ia, oa = np.asarray(inp), np.asarray(out)
            if ia.shape != oa.shape:
                consistent = False
                break
            for r in range(ia.shape[0]):
                for c in range(ia.shape[1]):
                    v_in, v_out = int(ia[r, c]), int(oa[r, c])
                    if v_in in color_map:
                        if color_map[v_in] != v_out:
                            consistent = False
                            break
                    else:
                        color_map[v_in] = v_out
                if not consistent:
                    break
            if not consistent:
                break
        if consistent and color_map:
            changed = {k: v for k, v in color_map.items() if k != v}
            if 0 < len(changed) <= 4:
                tree = ("get", R, C)
                for src, dst in changed.items():
                    tree = ("if", ("eq", ("get", R, C), ("const", src)),
                            ("const", dst), tree)
                hypotheses.append((tree, 1))

    # 2. Size-ratio tiling: if output is integer multiple of input
    if inp0.shape[0] > 0 and inp0.shape[1] > 0:
        ratio_r = out0.shape[0] / inp0.shape[0]
        ratio_c = out0.shape[1] / inp0.shape[1]
        IR, IC = ("inp_r",), ("inp_c",)
        if ratio_r == int(ratio_r) and ratio_c == int(ratio_c) and (ratio_r > 1 or ratio_c > 1):
            # Simple tiling
            hypotheses.append((("inp", ("mod", R, IR), ("mod", C, IC)), 1))
            # Tiling with horizontal flip on odd columns
            flip_c = ("sub", ("sub", IC, ONE), ("mod", C, IC))
            hypotheses.append((("inp", ("mod", R, IR),
                                ("if", ("mod", ("sub", C, ("mod", C, IC)), IC),
                                 flip_c, ("mod", C, IC))), 1))

    # 3. For same-size tasks: analyze which cells change and what they need
    if inp0.shape == out0.shape:
        diff = inp0 != out0
        if diff.any():
            changed_from = set(inp0[diff].tolist())
            changed_to = set(out0[diff].tolist())
            unchanged_vals = set(inp0[~diff].tolist())

            # "Change all of color X to Y, but only when neighbor condition"
            # For each (from_color, to_color) pair, try n_count conditions
            for fc in changed_from:
                mask_fc = inp0 == fc
                actually_changed = mask_fc & diff
                stayed_same = mask_fc & ~diff
                if actually_changed.any() and stayed_same.any():
                    # Some cells of this color change, others don't → spatial condition
                    to_vals = set(out0[actually_changed].tolist())
                    for tv in to_vals:
                        # Try: change fc→tv if n_count of fc neighbors < threshold
                        for thresh in range(1, 5):
                            hypotheses.append((
                                ("if", ("eq", ("get", R, C), ("const", fc)),
                                 ("if", ("gt", ("const", thresh),
                                         ("n_count", R, C, ("const", fc))),
                                  ("const", tv), ("get", R, C)),
                                 ("get", R, C)), 1))
                            hypotheses.append((
                                ("if", ("eq", ("get", R, C), ("const", fc)),
                                 ("if", ("gt", ("n_count", R, C, ("const", fc)),
                                         ("const", thresh)),
                                  ("const", tv), ("get", R, C)),
                                 ("get", R, C)), 1))

            # "Fill zeros based on row/col content"
            if 0 in changed_from and not (set(changed_from) - {0}):
                # Only zeros change — fill pattern
                fill_colors = list(changed_to - {0})
                for fc in fill_colors[:3]:
                    FC = ("const", fc)
                    # Fill zeros with fc if row has fc
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("if", ("row_count", R, FC), FC,
                          ("get", R, C))), 1))
                    # Fill zeros with fc if col has fc
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("if", ("col_count", C, FC), FC,
                          ("get", R, C))), 1))
                    # Fill zeros with fc if both row and col have fc
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("if", ("row_count", R, FC),
                          ("if", ("col_count", C, FC),
                           FC, ("get", R, C)),
                          ("get", R, C))), 1))
                    # Symmetric fill: if cell is 0, read from mirror
                    MR, MC = ("max_r",), ("max_c",)
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("get", R, ("sub", ("sub", MC, ONE), C))), 1))
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("get", ("sub", ("sub", MR, ONE), R), C)), 1))
                    # Flood fill: if 0 and neighbor has fc, become fc (multi-pass)
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("if", ("n_count", R, C, FC), FC, ("get", R, C))), 2))
                    hypotheses.append((
                        ("if", ("get", R, C), ("get", R, C),
                         ("if", ("n_count", R, C, FC), FC, ("get", R, C))), 3))

    # 4. For diff-size: try cropping to subregion containing non-background
    if inp0.shape != out0.shape:
        or0, oc0 = out0.shape
        # Try reading input from various offsets
        for dr in range(min(5, inp0.shape[0] - or0 + 1) if inp0.shape[0] > or0 else 1):
            for dc in range(min(5, inp0.shape[1] - oc0 + 1) if inp0.shape[1] > oc0 else 1):
                if dr == 0 and dc == 0:
                    continue
                hypotheses.append((("inp", ("add", R, ("const", dr)),
                                          ("add", C, ("const", dc))), 1))

    # 5. Dimension-aware hypotheses
    h, w = out0.shape
    # Mirror at various axes
    for axis_c in set([w // 2, w - 1]):
        AC = ("const", axis_c)
        hypotheses.append((("get", R, ("sub", AC, C)), 1))
        hypotheses.append((("if", ("get", R, C), ("get", R, C),
                           ("get", R, ("sub", AC, C))), 1))
    for axis_r in set([h // 2, h - 1]):
        AR = ("const", axis_r)
        hypotheses.append((("get", ("sub", AR, R), C), 1))
        hypotheses.append((("if", ("get", R, C), ("get", R, C),
                           ("get", ("sub", AR, R), C)), 1))
    # Modular with actual grid dims
    for div_r in set([h, h // 2, h // 3]) - {0}:
        for div_c in set([w, w // 2, w // 3]) - {0}:
            hypotheses.append((("inp", ("mod", R, ("const", div_r)),
                                     ("mod", C, ("const", div_c))), 1))
            hypotheses.append((("get", ("mod", R, ("const", div_r)),
                                     ("mod", C, ("const", div_c))), 1))
    # Column/row conditional: keep only one column/row (for extraction tasks)
    if inp0.shape == out0.shape:
        for k in range(min(w, 8)):
            hypotheses.append((("if", ("eq", C, ("const", k)),
                               ("get", R, C), ("const", 0)), 1))
        for k in range(min(h, 8)):
            hypotheses.append((("if", ("eq", R, ("const", k)),
                               ("get", R, C), ("const", 0)), 1))
    # Column shift: get(r, sub(K, c)) for dimension-based K values
    for k in set([w - 1, w // 2, h - 1, h // 2]) - {0}:
        K = ("const", k)
        hypotheses.append((("get", R, ("sub", K, C)), 1))
        hypotheses.append((("get", ("sub", K, R), C), 1))
        # Conditional: if modular condition, shift
        hypotheses.append((
            ("if", ("mod", C, ("const", 2)),
             ("get", R, ("sub", K, C)), ("get", R, C)), 1))

    # 6. Geometric + recolor compositions
    if inp0.shape == out0.shape:
        MR, MC = ("max_r",), ("max_c",)
        flip_r = ("sub", ("sub", MR, ONE), R)
        flip_c = ("sub", ("sub", MC, ONE), C)
        # For each color that appears in output but not input, or changes
        out_colors = set(out0.flat)
        inp_colors = set(inp0.flat)
        for x in inp_colors:
            for y in out_colors:
                if x == y:
                    continue
                # Recolor + flip: if flipped cell == x, then y
                hypotheses.append((
                    ("if", ("eq", ("get", flip_r, C), ("const", x)),
                     ("const", y), ("get", R, C)), 1))
                hypotheses.append((
                    ("if", ("eq", ("get", R, flip_c), ("const", x)),
                     ("const", y), ("get", R, C)), 1))
                # If cell == x AND n_count == 0 (isolated), recolor to y
                hypotheses.append((
                    ("if", ("eq", ("get", R, C), ("const", x)),
                     ("if", ("n_count", R, C, ("const", x)),
                      ("get", R, C), ("const", y)),
                     ("get", R, C)), 1))

    # 7. Object-aware hypotheses from CCL analysis on first example (4-conn)
    try:
        import gp as _gp
        _obj_size_tree = ("obj_size", ("r",), ("c",))
        _sizes = _gp.evaluate(_obj_size_tree, inp0)
        _obj_id_tree = ("obj_id", ("r",), ("c",))
        _ids = _gp.evaluate(_obj_id_tree, inp0)
        _obj_color_tree = ("obj_color", ("r",), ("c",))
        _colors = _gp.evaluate(_obj_color_tree, inp0)
        if _sizes is not None and _ids is not None:
            _unique_sizes = sorted(set(_sizes[_sizes > 0].tolist()))
            _unique_ids = sorted(set(_ids[_ids > 0].tolist()))
            # Keep/remove objects of each observed size
            for sz in _unique_sizes[:6]:
                SZ = ("const", int(sz))
                hypotheses.append((("if", ("eq", ("obj_size", R, C), SZ),
                                   ("get", R, C), ("const", 0)), 1))
                hypotheses.append((("if", ("eq", ("obj_size", R, C), SZ),
                                   ("const", 0), ("get", R, C)), 1))
            # Try selecting each object by ID (up to 5)
            for oid in _unique_ids[:5]:
                hypotheses.append((("if", ("eq", ("obj_id", R, C), ("const", int(oid))),
                                   ("get", R, C), ("const", 0)), 1))
            # For each observed object color, try keep/remove by color
            if _colors is not None:
                _unique_obj_colors = sorted(set(_colors[_colors > 0].tolist()))
                for oc in _unique_obj_colors[:5]:
                    OC = ("const", int(oc))
                    hypotheses.append((("if", ("eq", ("obj_color", R, C), OC),
                                       ("get", R, C), ("const", 0)), 1))
                    hypotheses.append((("if", ("eq", ("obj_color", R, C), OC),
                                       ("const", 0), ("get", R, C)), 1))
                    # Recolor objects of this color to other observed colors
                    for oc2 in _unique_obj_colors[:5]:
                        if oc2 != oc:
                            hypotheses.append((("if", ("eq", ("obj_color", R, C), OC),
                                               ("const", int(oc2)), ("get", R, C)), 1))
            # Size-based recoloring with thresholds
            if len(_unique_sizes) > 1 and inp0.shape == out0.shape:
                out_colors = set(out0.flat) - {0}
                # Compute gap thresholds (midpoints between distinct sizes)
                _thresholds = set(_unique_sizes)
                for i in range(len(_unique_sizes) - 1):
                    _thresholds.add((_unique_sizes[i] + _unique_sizes[i+1]) // 2)
                for tv in out_colors:
                    TV = ("const", int(tv))
                    for sz in _thresholds:
                        SZ = ("const", int(sz))
                        # Recolor objects of exactly this size
                        hypotheses.append((("if", ("eq", ("obj_size", R, C), SZ),
                                           TV, ("get", R, C)), 1))
                        # Recolor objects larger than threshold
                        hypotheses.append((("if", ("gt", ("obj_size", R, C), SZ),
                                           TV, ("get", R, C)), 1))
                        # Recolor objects smaller than/equal to threshold
                        hypotheses.append((("if", ("gt", SZ, ("obj_size", R, C)),
                                           TV, ("get", R, C)), 1))
                    # Also try threshold at max_obj_size
                    hypotheses.append((("if", ("eq", ("obj_size", R, C), ("max_obj_size",)),
                                       TV, ("get", R, C)), 1))
    except Exception:
        pass

    # 8. Object-aware hypotheses using 8-connectivity
    try:
        import gp as _gp
        _obj8_size_tree = ("obj8_size", ("r",), ("c",))
        _sizes8 = _gp.evaluate(_obj8_size_tree, inp0)
        _obj8_id_tree = ("obj8_id", ("r",), ("c",))
        _ids8 = _gp.evaluate(_obj8_id_tree, inp0)
        _obj8_color_tree = ("obj8_color", ("r",), ("c",))
        _colors8 = _gp.evaluate(_obj8_color_tree, inp0)
        if _sizes8 is not None and _ids8 is not None:
            _unique_sizes8 = sorted(set(_sizes8[_sizes8 > 0].tolist()))
            _unique_ids8 = sorted(set(_ids8[_ids8 > 0].tolist()))
            for sz in _unique_sizes8[:6]:
                SZ = ("const", int(sz))
                hypotheses.append((("if", ("eq", ("obj8_size", R, C), SZ),
                                   ("get", R, C), ("const", 0)), 1))
                hypotheses.append((("if", ("eq", ("obj8_size", R, C), SZ),
                                   ("const", 0), ("get", R, C)), 1))
            for oid in _unique_ids8[:5]:
                hypotheses.append((("if", ("eq", ("obj8_id", R, C), ("const", int(oid))),
                                   ("get", R, C), ("const", 0)), 1))
            if _colors8 is not None:
                _unique_obj8_colors = sorted(set(_colors8[_colors8 > 0].tolist()))
                for oc in _unique_obj8_colors[:5]:
                    OC = ("const", int(oc))
                    hypotheses.append((("if", ("eq", ("obj8_color", R, C), OC),
                                       ("get", R, C), ("const", 0)), 1))
                    hypotheses.append((("if", ("eq", ("obj8_color", R, C), OC),
                                       ("const", 0), ("get", R, C)), 1))
            # Size-based recoloring with 8-connectivity
            if len(_unique_sizes8) > 1 and inp0.shape == out0.shape:
                out_colors = set(out0.flat) - {0}
                for tv in out_colors:
                    TV = ("const", int(tv))
                    for sz in _unique_sizes8[:4]:
                        SZ = ("const", int(sz))
                        hypotheses.append((("if", ("eq", ("obj8_size", R, C), SZ),
                                           TV, ("get", R, C)), 1))
                        hypotheses.append((("if", ("gt", ("obj8_size", R, C), SZ),
                                           TV, ("get", R, C)), 1))
                    hypotheses.append((("if", ("eq", ("obj8_size", R, C), ("max_obj8_size",)),
                                       TV, ("get", R, C)), 1))
    except Exception:
        pass

    return hypotheses


def seed_sweep(tasks, library):
    """Fast first pass: test all seed templates on all tasks. No evolution.

    Returns (solved_dict, best_fit_dict) with results from seeds alone.
    This is cheap — ~100 seeds × N tasks × 3 examples — and finds all
    tasks solvable by a single known transform.
    """
    seeds = make_seeds()
    solved = {}
    best_fit = {}
    for tid, task in tasks.items():
        examples = task["train"]
        best_t, best_p, best_f = IDENTITY, 1, 0.0
        # Test generic seeds + task-specific hypotheses
        task_seeds = seeds + task_hypotheses(examples)
        for tree, passes in task_seeds:
            if solves(tree, examples, library, passes):
                tree = simplify(tree, examples, library, passes)
                solved[tid] = (tree, passes)
                p_str = f" x{passes}" if passes > 1 else ""
                print(f"  ✓ {tid}{p_str}: {gp.to_str(tree)}")
                break
            f = fitness(tree, examples, library, passes)
            if f > best_f:
                best_t, best_p, best_f = tree, passes, f
        if tid not in solved:
            best_fit[tid] = (best_t, best_p, best_f)

    # Quick refinement pass on top near-misses within seed sweep
    refine_count = 0
    near = [(tid, t, p, f) for tid, (t, p, f) in best_fit.items() if f > 0.85]
    near.sort(key=lambda x: -x[3])
    for tid, t, p, f in near[:40]:
        examples = tasks[tid]["train"]
        # Single round of node-level refinement
        t2, f2 = refine(t, examples, library, p)
        if f2 > f:
            t, f = t2, f2
        if not solves(t, examples, library, p) and f > 0.85:
            # Try error-driven refinement
            t2, f2 = error_driven_refine(t, examples, library, p)
            if f2 > f:
                t, f = t2, f2
        if solves(t, examples, library, p):
            t = simplify(t, examples, library, p)
            solved[tid] = (t, p)
            del best_fit[tid]
            refine_count += 1
            p_str = f" x{p}" if p > 1 else ""
            print(f"  ✓ {tid}{p_str} (refined): {gp.to_str(t)}")
        else:
            best_fit[tid] = (t, p, f)
    if refine_count:
        print(f"  Seed refinement: +{refine_count} solved")

    return solved, best_fit


def evolve(tasks, rounds, library, tasks_per_round=50):
    """Per-task evolution with cross-task abstraction.

    Phase 1: fast seed sweep on all tasks (finds tasks solvable by templates).
    Phase 2: focused evolutionary search on top near-misses with larger budget.
    """
    # Phase 1: seed sweep (fast — no evolution, all tasks)
    solved, best_fit = seed_sweep(tasks, library)
    if len(solved) >= MIN_SHARED_SOLVERS:
        abstract(list(solved.values()), library)
    n_near = sum(1 for v in best_fit.values() if v[2] > 0.8)
    print(f"Seed sweep: {len(solved)}/{len(tasks)} solved, "
          f"{n_near} near, lib={len(library)}")

    # Phase 1.5: deeper refinement on remaining near-misses
    refine_count = 0
    near_misses = [(tid, t, p, f) for tid, (t, p, f) in best_fit.items() if f > 0.7]
    near_misses.sort(key=lambda x: -x[3])
    for tid, t, p, f in near_misses[:100]:
        examples = tasks[tid]["train"]
        # Multi-round node-level refinement
        for _ in range(3):
            t2, f2 = refine(t, examples, library, p)
            if f2 <= f:
                break
            t, f = t2, f2
            if solves(t, examples, library, p):
                t = simplify(t, examples, library, p)
                solved[tid] = (t, p)
                refine_count += 1
                p_str = f" x{p}" if p > 1 else ""
                print(f"  ✓ {tid}{p_str} (deep-refined): {gp.to_str(t)}")
                break
        # Try wrapping with n_count conditions
        if tid not in solved and f > 0.8:
            t2, f2 = refine_wrap(t, examples, library, p)
            if f2 > f:
                t, f = t2, f2
                if solves(t, examples, library, p):
                    t = simplify(t, examples, library, p)
                    solved[tid] = (t, p)
                    refine_count += 1
                    p_str = f" x{p}" if p > 1 else ""
                    print(f"  ✓ {tid}{p_str} (wrapped): {gp.to_str(t)}")
        # Try error-driven refinement
        if tid not in solved and f > 0.8:
            t2, f2 = error_driven_refine(t, examples, library, p)
            if f2 > f:
                t, f = t2, f2
                if solves(t, examples, library, p):
                    t = simplify(t, examples, library, p)
                    solved[tid] = (t, p)
                    refine_count += 1
                    p_str = f" x{p}" if p > 1 else ""
                    print(f"  ✓ {tid}{p_str} (error-refined): {gp.to_str(t)}")
        if tid not in solved:
            best_fit[tid] = (t, p, f)
    if refine_count:
        print(f"Refinement: +{refine_count} solved")
        if len(solved) >= MIN_SHARED_SOLVERS:
            abstract(list(solved.values()), library)

    # Phase 2: evolutionary search — bigger budget on fewer tasks
    for rnd in range(1, rounds + 1):
        new_count = 0
        tids = [t for t in tasks if t not in solved]
        tids.sort(key=lambda t: -(best_fit[t][2] if t in best_fit else 0))
        if len(tids) > tasks_per_round:
            tids = tids[:tasks_per_round]

        transfer_progs = list({id(v): v for v in solved.values()}.values())

        for tid in tids:
            prev_t = best_fit[tid][0] if tid in best_fit else None
            prev_p = best_fit[tid][1] if tid in best_fit else 1
            prev_f = best_fit[tid][2] if tid in best_fit else 0
            ps, gs = search_budget(prev_f)
            tree, passes, f = search(tasks[tid]["train"], seed=prev_t,
                                     seed_passes=prev_p, library=library,
                                     pop_size=ps, gens=gs,
                                     transfer=transfer_progs)
            if f > best_fit.get(tid, (None, 0, 0))[2]:
                best_fit[tid] = (tree, passes, f)
            if solves(tree, tasks[tid]["train"], library, passes):
                solved[tid] = (tree, passes)
                new_count += 1
                p_str = f" x{passes}" if passes > 1 else ""
                print(f"  ✓ {tid}{p_str}: {gp.to_str(tree)}")

        if len(solved) >= MIN_SHARED_SOLVERS:
            abstract(list(solved.values()), library)

        n_near = sum(1 for t in tids if best_fit.get(t, (None, 0, 0))[2] > 0.8)
        print(f"Round {rnd}: {len(solved)}/{len(tasks)} solved "
              f"(+{new_count}), {n_near} near, lib={len(library)}")
    return solved


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(tasks, library, train_solvers=None):
    """Score on held-out test examples."""
    transfer = list({id(v): v for v in (train_solvers or {}).values()}.values())
    solved_train, solved_test, total = 0, 0, 0

    for tid, task in tasks.items():
        train_ex, test_ex = task["train"], task["test"]
        if not test_ex:
            continue
        total += 1

        # Try direct transfer from training solvers
        found = False
        for tree, passes in transfer:
            if solves(tree, train_ex, library, passes):
                solved_train += 1
                if solves(tree, test_ex, library, passes):
                    solved_test += 1
                    print(f"  ✓ {tid} (transfer): {gp.to_str(tree)}")
                found = True
                break
        if found:
            continue

        # Per-task search
        tree, passes, _ = search(train_ex, library=library)
        if solves(tree, train_ex, library, passes):
            solved_train += 1
            if solves(tree, test_ex, library, passes):
                solved_test += 1
                print(f"  ✓ {tid}: {gp.to_str(tree)}")

    print(f"Eval: {solved_test}/{total} test ({solved_train} train)")
    return solved_test


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/ARC-AGI/data/training"
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "data/ARC-AGI/data/evaluation"
    rounds     = int(sys.argv[3]) if len(sys.argv) > 3 else 10
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
