# agi-core-minimal: The 4-Pillar Learning Loop

**One algorithm. Four pillars. Compounding intelligence.**

```
explore → feedback → keep best (even near-misses) → abstract → repeat
    │         │              │                           │
    │         │              │                           └─ promote sub-programs as new primitives
    │         │              └─ approximability: imperfect attempts are valuable
    │         └─ closed-loop error signal drives every decision
    └─ search the space of programs by composing primitives
```

Based on the research and principles proposed by [Vibhor Jain](https://github.com/vibhor-77).

## The 4 Pillars

| # | Pillar | What it does | In the code |
|---|--------|-------------|-------------|
| 1 | **Feedback Loops** | Cell-level error signal that drives the entire cycle — which programs to keep, which sub-programs to promote, when to stop | `error(program, examples) → float` |
| 2 | **Approximability** | Near-miss programs are valuable — closer is better, even if imperfect. The loop keeps the *best* program per task, not just perfect solves | quality = 1 - error, fed into abstraction |
| 3 | **Abstraction & Composition** | Two sides of one coin: *compose* programs from primitives, *abstract* recurring sub-programs back into new primitives | `execute(program, grid)` + `abstract(scored_programs)` |
| 4 | **Exploration** | Weighted sampling of program compositions — proven primitives get sampled more, new ones maintain exploration | `candidates(max_depth, budget) → generator` |

**Why these 4 and not more?** Composition and abstraction are inseparable — composition builds programs *down* from primitives, abstraction lifts sub-programs *up* into new primitives. They're the same mechanism in two directions. Approximability is what makes the cycle compound: without it, only perfect solves feed abstraction and most tasks contribute nothing.

## How Compounding Works

The loop runs multiple rounds. Each round:

1. **Explore** (Pillar 4) — sample compositions weighted by primitive usefulness
2. **Score** (Pillar 1) — cell-level feedback tells us how close each program got
3. **Keep best** (Pillar 2) — even a 60%-correct program is valuable
4. **Abstract** (Pillar 3) — sub-programs recurring across good attempts become new primitives
5. **Repeat** — the library grows, so depth-2 search over promoted depth-2 compositions reaches depth-4

The key: a near-miss in round 1 gets its sub-programs promoted. In round 2, those promoted primitives let exploration reach programs it couldn't before. That's compounding — even failures contribute.

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
Round 1: 7/400 solved, 251 near-misses, 10 primitives (+5 new)
Round 2: 7/400 solved, 251 near-misses, 10 primitives (+5 new)
Round 3: 7/400 solved, 251 near-misses, 10 primitives (+0 new)
```

The scaffolding starts with 5 primitives (rotate, flip, transpose). Accuracy comes from adding more primitives — the loop and architecture are what matter.

## `solve.py` (~100 lines)

```
docstring                — the 4 pillars, one sentence each
Domain                   — primitives dict + load()
Pillar 3a: Composition   — execute(program, grid)
Pillar 1:  Feedback      — error(program, examples)     ← cell-level
Pillar 4:  Exploration   — candidates(max_depth, budget) ← weighted sampling
Pillar 3b: Abstraction   — abstract(scored_programs)  ← quality-weighted
The Loop                 — learn(tasks, rounds)        ← Pillar 2 lives here
Main                     — load tasks, call learn()
```

**Adding a new primitive** is one line in the `primitives` dict:

```python
"crop_top": lambda g: g[1:],
```

**Adding a new domain** means replacing `primitives` and `load()` — everything else stays the same.

## Full System

The minimal `solve.py` demonstrates the architecture. The full system lives at [agi-core](https://github.com/vibhor-77/agi-core) and adds:

- 75 atomic primitives (transforms + perception + parameterized)
- 10 wake phases (exhaustive enumeration + 9 ARC-specific strategies)
- Bounded library with eviction and ROI tracking
- Multi-domain support (ARC-AGI-1/2, Zork, list ops, symbolic math)
- Interleaved train → eval pipeline with culture transfer

## License

MIT
