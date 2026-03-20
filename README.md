# agi-core-minimal

**Evolutionary program synthesis from atomic primitives.**

A genetic programming system that discovers grid transformations (flips, rotations, recoloring, conditional logic) by composing 11 truly primitive operations. No hand-crafted transforms — the system evolves them.

Based on the research and principles proposed by [Vibhor Jain](https://github.com/vibhor-77).

## How it works

The system evolves **expression trees** that compute `f(grid, r, c) → color` for each cell of the output grid. Starting from atomic operations (read a cell, add two numbers, compare values), evolution discovers compositions like:

```
get(c, r)                                          → transpose
get(sub(sub(max_r, 1), r), c)                      → flip vertical
if(eq(get(r,c), 5), 8, if(eq(get(r,c), 8), 5, get(r,c)))  → swap colors 5↔8
if(eq(get(r,c), 4), get(r, sub(sub(max_c,1), c)), get(r,c)) → conditional mirror
```

These are not built-in — evolution finds them from scratch.

### The primitives

Every program is built from these 11 operations:

| Category | Operations | Purpose |
|----------|-----------|---------|
| **Position** | `r`, `c`, `max_r`, `max_c` | Where am I? How big is the grid? |
| **Perception** | `get(row, col)` | Read the input grid at a computed position |
| **Constants** | `0`–`9` | The 10 ARC color values |
| **Arithmetic** | `add`, `sub`, `mod` | Compute positions (mirror, shift, wrap) |
| **Logic** | `eq`, `gt`, `if` | Compare values and branch |

This set is **universal**: any computable grid transformation can be expressed as a composition of these operations (given sufficient depth).

### The evolution loop

For each task, the system runs a mini-evolution:

1. **Seed** the population with known-useful templates (geometric transforms, recolors) and random trees
2. **Evaluate** each program's fitness on the task's training examples
3. **Select** the fittest via tournament selection
4. **Breed** new programs via crossover (swap subtrees between parents) and mutation
5. **Repeat** for multiple generations until solved or stale
6. **Simplify** solutions by greedily removing unnecessary subtrees

### Compounding via abstraction

After each round of per-task evolution, subtrees that appear in 2+ solver programs are promoted to the **library** as new single-node primitives. A 10-node subtree that took many generations to discover becomes a `("lib", "L0")` terminal — usable in any future random tree at zero depth cost.

This is the compounding mechanism: round 1 discovers `get(c, r)` (transpose). It gets abstracted as `L0`. In round 5, a program uses `L0` as a building block to solve a different task that requires transpose + something else.

## Architecture

Two files, clear separation:

**`gp.py`** — The GP engine (~200 lines). Tree representation, vectorized numpy evaluation, crossover/mutation operators. Knows nothing about ARC.

**`solve.py`** — The application (~280 lines). Data loading, fitness scoring, seed generation, per-task search, abstraction, evolution loop. Knows nothing about tree internals.

## Quick start

```bash
# Get the ARC dataset
git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI

pip install numpy

# Run (default: 20 rounds on all 400 training tasks + 400 eval tasks)
python solve.py

# Or specify paths and rounds
python solve.py data/ARC-AGI/data/training data/ARC-AGI/data/evaluation 10
```

### What you'll see

```
Loaded 400 training tasks
  ✓ 74dd1130: get(c, r)                              ← discovered transpose
  ✓ 67a3c6ac: get(r, sub(sub(max_c, 1), c))          ← discovered flip_h
  ✓ d511f180: if(eq(get(r,c),8), 5, if(eq(get(r,c),5), 8, get(r,c)))  ← color swap
  ++ lib 'L0' = get(c, r) (size=3, shared by 2)      ← abstracted for reuse
Round 1: 19/400 solved (+19), 155 near, lib=5
```

## Design principles

### Continuous, not threshold-based

The system avoids hard cutoffs wherever possible:

- **Search budget** scales linearly with previous fitness. A task at fitness 0.6 gets 60% more compute than one at 0.0. No cliff between "near-miss" and "not."
- **Patience** (when to stop) scales with fitness. High fitness = closer to solving = worth trying longer. At fitness 0, patience is 3 generations. At 0.9, it's 8.
- **Neighborhood refinement** triggers probabilistically: probability = fitness. A task at 0.3 fitness has 30% chance of refinement; at 0.9, 90%.
- **Library terminal probability** scales with library size: `lib_size / (lib_size + num_terminals)`. An empty library contributes nothing; a large one gets proportionally more sampling.

### Every constant has a reason

| Constant | Value | Why |
|----------|-------|-----|
| `P_TERMINAL` | 0.35 | ≈ 1/(1 + mean_arity). Mean arity across our ops is ~2.1. This keeps expected tree size finite. |
| `MAX_TREE_DEPTH` | 7 | Allows trees up to ~128 nodes. Deep enough for complex compositions, shallow enough for tractable search. |
| `TOURNAMENT_K` | 4 | Gives ~75% chance of selecting from the top quartile. Standard GP range is 2–7. |
| `CHANGED_CELL_WEIGHT` | 3.0 | In typical ARC tasks, ~10-20% of cells change. Weight of 3 means a program must fix ~75% of changed cells to beat identity — creating a gradient above the identity plateau. |
| `PARSIMONY_PENALTY` | 0.002 | A 50-node tree loses ~0.1 fitness. Enough to prefer simpler programs among equals, not enough to prevent finding complex solutions. |
| `MUTATE_WEIGHTS` | 5:3:2 | Subtree mutation is most exploratory, point mutation is conservative, hoist simplifies. Weights follow the big-medium-small mutation spectrum from GP literature. |

### Minimal and universal

The primitive set is deliberately small (11 operations) but **universal** — any computable grid transformation can be expressed. The goal is not to add domain-specific primitives but to build a mechanism that discovers useful compositions from atomic operations.

## Current results

- **20-22/400** training tasks solved (seed sweep + 3-5 evolutionary rounds)
- 12 library entries abstracted, reused in later solutions
- 112 near-misses (fitness > 0.8) — the system is making partial progress on many tasks
- Multi-pass programs discovered (2-pass gravity, 3-pass iterative refinement)

### What works

| Category | Examples | Count |
|----------|---------|-------|
| Geometric transforms | transpose, flip_h, flip_v, rotate | 6 |
| Shifts | shift down by k, column tiling | 3 |
| Color recoloring | if(cell==7, 5, cell), swap 5↔8 | 5 |
| Conditional geometry | if(cell==0, flip_v, identity) | 3 |
| Multi-pass | gravity (propagate down), iterative fill | 2 |
| Compositions | conditional mirror, library-based combos | 3 |

### What doesn't work (yet)

The remaining ~380 tasks require **object-level reasoning** — identifying connected components, detecting patterns, counting objects, understanding spatial relationships. The cell-level primitives are universal in theory but the search space for object-level behaviors is too large for the current evolutionary budget.

The principled path forward: multi-pass evaluation (already implemented) enables cellular-automata-like computation where local rules produce global behavior through iteration. The challenge is evolving the right local rules.

## License

MIT
