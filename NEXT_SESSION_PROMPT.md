# Next Session Prompt — WuBuMath

## Context

You are continuing work on **WuBuMath** (https://github.com/waefrebeorn/WuBuMath), a pure C11 mathematical & media encoding library. It was forked from bytropix to be the dedicated home for all WuBu mathematical work.

## Current State (as of 2026-06-26)

- **64/64 tests pass** (29 WuBuMath + 35 slermed JAX)
- **346 knowledge files** vaulted (mind palace, papers, python source, jax reference)
- **16 C source files**, 5 headers, ~4,477 LOC
- **Build**: `make clean && make test` → 0 errors

## Repository Structure

```
WuBuMath/
├── include/wubumath.h          # Master header
├── src/math/                   # Color manifold, positional encoding, utils
├── src/model/                  # Hamilton encoder, VHF decoder, VHF audio
├── src/train/                  # Q-controller, loss manifold
├── src/jax/                    # Slermed JAX core (arena, SIMD, GEMM, MLP, lax, IR)
├── src/tests/                  # wubu_tests.c (29), jax_slermed_test.c (35)
├── examples/media_creator.c    # Generates frames + VHF tones → WAV
├── vault/                      # All knowledge lives here
│   ├── mind-palace/            # 143 files — INDEX.md has full map
│   ├── python/                 # 131 Python source files (reference)
│   ├── papers/                 # 27 research papers
│   └── jax-source/             # 12 JAX reference docs
├── ROADMAP.md                  # 15 phases, 322 test targets
└── Makefile
```

## What To Do Next

### Priority 1: Phase 1 Encoder (ROADMAP.md)

Port `phase1_encoder_update.py` → `src/encoders/phase1.c`. This is the biggest remaining gap. Key functions to slerm:

1. **Complex mask generation** (ellipse, rect, boolean union/diff/inter)
2. **Synthetic RGBA texture batch** (alpha mask × checkerboard + high-freq texture)
3. **Color pattern generation** (32 frames, radial HSL gradients)
4. **Moving shape pattern generation** (animated ellipse with phase offset)
5. **VHF tone pipeline** (8 tones × 4410 samples → WAV)

Reference: `vault/python/encoders/phase1-symmetric-encoder/phase1_encoder_update.py`

### Priority 2: Phase 2 Topological AE

Port `QAE_Advanced.py` → `src/encoders/phase2.c`

### Priority 3: Hamilton CPU Encoder

Port the 35 files in `hamilton-encoder-cpu/` → `src/encoders/hamilton.c`
Focus on: Wubu_Geodesic_Sphere, Wubu_Quaternion_Monolith, Wubu_Spectral_Field

### Priority 4: HashMind

Port `WuBuMindJAX.py` (V7.1) → `src/encoders/hashmind.c`

### Priority 5: Attention Mechanisms

Port hyperbolic attention (Clockwork, TgT) → `src/attention/hyperbolic.c`

### Priority 6: Diffusion

Port `WuBu_Funnel_Diffusionv0.1.py` → `src/diffusion/funnel.c`

## Verification

Always run after changes:
```bash
cd /home/wubu/WuBuMath
make clean && make test
```

Target: **64/64 tests must remain green** before committing.

## Git Workflow

```bash
git add -A
git commit -m "feat: <what you did>"
git push
```

## Key Files to Read First

1. `ROADMAP.md` — Full 15-phase plan
2. `vault/mind-palace/INDEX.md` — Knowledge map with tier system
3. `vault/mind-palace/plans/devils_advocate_v13.md` — Latest plan
4. `src/tests/wubu_tests.c` — Existing test patterns
5. `include/wubumath.h` — Current API surface

## The Divergence Story

bytropix started as math/training, then added C inference server (commit 816aea8, May 28). This caused confusion — bytropix was "2 things." WuBuMath is the reconciliation: **all math comes home**. bytropix keeps the inference engine.

## Session End Ritual

When done:
1. `make clean && make test` — verify 64/64 pass
2. `git add -A && git commit -m "..." && git push`
3. Write next session prompt (this file, updated)
4. Update ROADMAP.md progress

---

*Generated: 2026-06-26 | Last commit: prestige session wrap-up*
