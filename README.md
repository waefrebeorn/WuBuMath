# WuBuMath — C11 math library + WUBQ video codec

> Pure C11, zero third-party deps, 78 tests green. Slermed from Python/JAX.

## What this is

Two things in one repo:

1. **WuBuMath library** — manifolds (Poincaré/Lorentz/nested-hyperbolic/SO3/quaternion), CVC11 slerm of JAX core (arena, SIMD, GEMM, MLP, lax, opt, IR), Hamilton encoder, VHF canvas, GAAD encoder, learned codec, flow matching, Riemannian SGD, manifold AD, hyperbolic attention, manifold CLIP components. 14 phases on the roadmap (Phase 0–5 complete, Phase 6–15 TODO).

2. **WUBQ video codec** — the 25-category SOTA war plan (`docs/SOTA_WARPLAN_25.md`). C1 (CABAC round-trip) CLOSED. Building toward beating x264/x265/VP9 on real content.

## Quick start

```bash
make test          # full test suite (78 tests, ~54s)
make all           # build everything
```

Per-module: `make test_<name>` (e.g. `make test_gaad`, `make test_vhf_engine`).

## Benchmark corpus (C19 — formalized)

See `../codec_ab1080/MANIFEST.md5.md` for the MD5-verified corpus manifest.

```bash
./tools/run_ab.sh [1080p|8k|all]   # one-command A/B rerun
```

## Test coverage

| Suite | Tests | Status |
|-------|-------|--------|
| WuBuMath core | 29/29 | ✅ |
| Slermed JAX | 35/35 | ✅ |
| GAAD encoder | 18/18 | ✅ |
| Learned codec | 14/14 | ✅ |
| VHF canvas | 40/40 | ✅ |
| Nested encoder | 6/6 | ✅ |
| **Total** | **78/78** | ✅ |

## 25-category state (see SOTA_WARPLAN_25.md for full details)

Run `python3 wubu_dominance_loop.py` for the live state matrix.

|| Wave 1 (C1,C4,C12,C19) | C1 ✓, C4 ✓ (redesigned tokctx, 15.1% at 45% density, exact round-trip), C12 ✓ (2/2 green), C19 ✓ | — ||
|| Wave 2 (C9,C10,C5,C6) | — | C9 ✗, C10 ✗, C5 ✗, C6 ✗ ||
|| Wave 3 (C13,C14,C7,C15) | — | C13 ✗, C14 ✗, C7 ✗, C15 ◐ (RDO framework exists, no unified encoder loop yet) ||
|| Wave 4 (C2,C3,C16,C17) | — | all ✗ ||
|| Wave 5 (C20,C21,C22) | — | all ✗ ||
|| Wave 6 (C23,C24,C25,C11,C18) | — | C25 ✗, C11 ✗, C18 ✗, C23 ◇ frontier, C24 ◇ frontier ||

**Keystone: C15** — without a unified RDO encoder loop, Waves 2–5 can't be measured against the corpus.

## License

Umbrella License v3.0 — see LICENSE.
