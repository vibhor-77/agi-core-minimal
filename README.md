# agi-core-minimal: The 5-Pillar Learning Loop

**One algorithm. Five pillars. Compounding intelligence.**

```
explore → feedback → abstract → repeat
  │          │          │          │
  │          │          │          └─ library grows, search shrinks, harder problems yield
  │          │          └─ extract recurring sub-programs as new primitives (weighted by quality)
  │          └─ continuous error signal: how close is a program?
  └─ search the space of programs by composing primitives
```

Based on the research and principles proposed by [Vibhor Jain](https://github.com/vibhor-77).

## The 5 Pillars

| Pillar | What it does | Code |
|--------|-------------|------|
| **Composition** | Programs are sequences of primitives; execute by chaining | `execute(program, grid) → grid` |
| **Feedback** | Continuous error signal — how close is a program to solving a task? | `error(program, examples) → float` |
| **Exploration** | Search the space of programs by composing primitives | `candidates(depth) → generator` |
| **Approximability** | Near-miss programs are valuable — closer is better, even if imperfect | best program per task, quality = 1 - error |
| **Abstraction** | Extract recurring sub-programs as new primitives, weighted by quality | `abstract(scored_programs) → [names]` |

Each pillar is **one function** with a clear name and docstring. The learning loop ties them together.

**Approximability is the key to compounding.** Even when a program doesn't solve a task, if it got *closer* than anything else, its sub-programs are worth promoting. A near-miss in round 1 becomes a stepping stone in round 2.

## Quick Start

```bash
# Clone ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI

pip install numpy

# Run the learning loop (3 rounds by default)
python solve.py

# Or specify data path and rounds
python solve.py data/ARC-AGI/data/training 50
```

### What you'll see

```
Loaded 400 tasks, 5 primitives
Round 1: 7/400 solved, 5 near-misses, 6 primitives (+1 new)
Round 2: 7/400 solved, 5 near-misses, 6 primitives (+0 new)
Round 3: 7/400 solved, 5 near-misses, 6 primitives (+0 new)
```

The scaffolding starts with 5 primitives (rotate, flip, transpose). Accuracy comes from adding more primitives — the loop and architecture are what matter.

## How it works

### `solve.py` (~100 lines)

```
docstring          — the 5 pillars, one sentence each
Domain             — primitives dict + load()
Composition        — execute(program, grid)
Feedback           — error(program, examples)
Exploration        — candidates(max_depth)
Abstraction        — abstract(scored_programs)  ← quality-weighted, not just perfect solves
The Loop           — learn(tasks, rounds)
Main               — load tasks, call learn()
```

**Adding a new primitive** is one line in the `primitives` dict:

```python
"crop_top": lambda g: g[1:],
```

**Adding a new domain** means replacing `primitives` and `load()` — everything else stays the same.

### The learning loop

1. **WAKE** (explore + feedback): For each task, search for the best program — perfect solve or closest approximation.
2. **SLEEP** (abstract): Extract recurring sub-programs from *all* best programs (solved and near-misses), weighted by quality.
3. **REPEAT**: The grown library expands effective search depth — a depth-1 search over a promoted depth-2 composition reaches depth-3.

This is the compounding mechanism: each round builds on the last. Near-misses contribute proportionally, so even failed attempts feed the next round.

## Full System

The minimal `solve.py` demonstrates the architecture. The full system lives at [agi-core](https://github.com/vibhor-77/agi-core) and adds:

- 75 atomic primitives (transforms + perception + parameterized)
- 10 wake phases (exhaustive enumeration + 9 ARC-specific strategies)
- Bounded library with eviction and ROI tracking
- Multi-domain support (ARC-AGI-1/2, Zork, list ops, symbolic math)
- Interleaved train → eval pipeline with culture transfer

## License

MIT
