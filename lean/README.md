# WuBuMath — Lean Formal-Proof Library

**Canonical home of the WuBu Nesting formal proofs.** Migrated from
`bytropix/MATH/lean/` on 2026-07-10 as part of consolidating *all*
WuBu math into this repo (see root `README.md`).

This is a standalone [Lake](https://github.com/leanprover/lean4) project
(Mathlib 4.29.1 dependency). It holds the **proven** geometric core of the
WuBu Nesting framework — the verified foundation the ~ WuBu ~ AGI's nested
hyperbolic manifolds stand on. **See [`../docs/LINKAGE.md`](../docs/LINKAGE.md)
for the theorem → capability → codebase map.**

## What is proven (imported in `WubuProofs.lean`, 0 `sorry`)

| # | File | Result | Status |
|---|---|---|---|
| 1 | `PoincareBall.lean` | `exp_0^c ∘ log_0^c = id` (round-trip) | ✅ |
| 2 | `MobiusAdd.lean` | Möbius addition preserves ball (1D) | ✅ |
| 3 | `MobiusAddND.lean` | **Möbius addition preserves ball (nD)** — `D²−c‖N‖²=(1−ca)(1−cb)[(1+cp)²+c²(ab−p²)]` | ✅ |
| 4 | `MLACompression.lean` | Frobenius error bound (KV compression) | ✅ |
| 5 | `HyperbolicGyration.lean` | 1D gyration + gyroassociativity | ✅ |
| 6 | `GyrationND.lean` | Mobius norm formula (nD) | ✅ |
| 7 | `GyrationND.lean` | Gyration preserves the ball (nD) | ✅ |
| 8 | `GyrationND.lean` | **‖gyr[u,v]w‖² = ‖w‖²** (Ungar 1.33) | ✅ |
| 9 | `GyrationND.lean` | **⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a,b⟩** (Ungar 1.32 — gyration is a rotation) | ✅ |
| 10 | `KleinDeck.lean` | Klein deck group; B² orientation-preserving; ABA=B | ✅ |
| 11 | `QuaternionRotation.lean` | ‖p·q‖² = ‖p‖²‖q‖² (SO(4) BSP foundation) | ✅ |
| 12 | `FiberBundle.lean` | SO(3) Lie algebra `[Lx,Ly]=Lz`; closure | ✅ |
| 13 | `NestedHyperbolicSpaces.lean` | φ-curvature progression; nesting invariants | ✅ |
| 14 | `HolographicOptimizer.lean` | `g = q·2π + r`; soul/echo Lazarus recovery | ✅ |
| 15 | `ParallelTransport.lean` | **PT is a Riemannian isometry** [GAP C003] | ✅ |
| 16 | `ExpMapBall.lean` | **c·‖exp_0^c(v)‖² < 1** [GAP C002] | ✅ |
| 17 | `GeodesicInterp.lean` | **γ_{x→y}(0) = x** [GAP C009] | ✅ |

**14 proof files, 16 oleans, 0 `sorry`.** Theorems 3, 9, and 15 are the
load-bearing ones (ball invariant, gyration-is-rotation, on-manifold
guarantee) — see LINKAGE.md §"three load-bearing theorems".

## What is NOT proven (kept as notebook only)

- **`WubuProofs/PowerTower.lean`** — *excluded* from the imported library.
  Written against a different mathlib version; does not compile on 4.29.1
  (rpow argument order + type mismatches). Left in-tree for a future port.
  See `../docs/theory/AUDIT_POWERTOWER.md`.

## Fast iteration tools

```bash
./fastcheck.sh WubuProofs/GyrationND.lean   # single file, ~4s (olean-cached)
./leanfind.sh tanh_lt_one                    # mathlib lemma lookup, 0 build cycles
lake build                                   # whole library, 0 sorry
./.lake/build/bin/wubu_proofs                # prints the proven-theorem table
```

`fastcheck.sh` uses `lake build <module>` (reuses cached oleans, ~97× faster
than `lake env lean`); `leanfind.sh` greps mathlib source for exact lemma
signatures — together they collapse the proof-iteration loop.

## Numerical demonstration

```bash
/usr/bin/python3.12 ../examples/wubu_manifold_demo.py
```

Shows every headline theorem (exp/log round-trip, ball preservation, gyration
rotation, PT isometry) numerically with the formulas transcribed directly
from these Lean definitions.

## Build (from scratch)

```bash
cd lean
lake build         # fetches mathlib4 (one-time ~hours), builds WubuProofs
```

> Requires Lean 4 + a working `lake` toolchain. The 11 GB `.lake` build cache
> is **never committed** (see root `.gitignore`).

## Relationship to the C library

The C11 math in `src/` (quaternion, hyperbolic, nested encoder, Riemannian
SGD, parallel transport) is the *implementation*; this Lean project is the
*formal specification* of the hyperbolic/nesting core. Where they overlap
(Möbius addition, Poincaré maps, gyration), the Lean side is the source of
truth — the C code is correct *because* these theorems hold.
