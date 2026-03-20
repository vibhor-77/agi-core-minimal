# agi-core-minimal

4-pillar wake-sleep compounding loop for ARC-AGI-1 in **99 lines of Python**.

## The 4 Pillars

1. **Primitives** — 12 atomic grid→grid transforms (rotate, flip, crop, tile, mirror, scale)
2. **Composition** — Programs are chains of primitives: `f(g(x))` = `["f", "g"]`
3. **Search (Wake)** — Enumerate all compositions up to depth-2, find programs that solve tasks
4. **Abstraction (Sleep)** — Extract recurring sub-programs from solutions, promote them as new primitives

The **compounding loop**: each round's learned abstractions become the next round's building blocks, enabling deeper solutions without exponential search cost.

## Quickstart

```bash
# Clone ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI

# Run (all 400 training tasks)
python solve.py

# Run on subset
python solve.py data/ARC-AGI/data/training 50
```

## Example Output

```
Loaded 400 tasks, 12 primitives

Round 1: 21/400 solved (+21 new), 6 learned
  L12 = mir_h → mir_v (3 tasks)
  L13 = crop → tile_h (1 tasks)
  ...
Round 2: 22/400 solved (+1 new), 7 learned        ← compounding!
  L18 = crop → L14 (1 tasks)                       ← depth-3 via promoted L14
Round 3: 22/400 solved (+0 new), 7 learned
  converged

Final: 22/400 solved, 19 primitives (7 learned)
```

Round 2 solves a task Round 1 couldn't by composing `crop` with `L14` (which encodes `flip_v → mir_v`). This is effectively a depth-3 program (`crop(flip_v(mir_v(x)))`) discovered at depth-2 search cost.

## How It Works

The key insight: promoting a depth-2 solution as a single primitive lets depth-2 search reach depth-3, depth-4, etc. Each round extends the effective search depth by 1 without exponential cost growth.

With 12 primitives, depth-2 search tries 144 compositions per task. After promoting 6 abstractions (18 total primitives), depth-2 search tries 324 — linearly more, not exponentially. But the effective depth reached is now 3+.
