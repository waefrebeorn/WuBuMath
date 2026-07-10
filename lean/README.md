# WuBuMath — Lean Formal-Proof Library

**Canonical home of the WuBu Nesting formal proofs.** Migrated from
`bytropix/MATH/lean/` on 2026-07-10 as part of consolidating *all*
WuBu math into this repo (see root `README.md`).

This is a standalone [Lake](https://github.com/leanprover/lean4) project
(Mathlib 4 dependency). It holds the **proven** geometric core of the
WuBu Nesting framework.

## What is proven (imported in `WubuProofs.lean`, no `sorry` in core)

| File | Result | Status |
|---|---|---|
| `WubuProofs/MobiusAdd.lean` | Möbius addition preserves the Poincaré ball (1D, closed form) | ✅ proven |
| `WubuProofs/HyperbolicGyration.lean` | 1D gyration is the identity (gyrogroup laws) | ✅ proven |
| `WubuProofs/HolographicOptimizer.lean` | `g = q·2π + r` decomposition; soul/echo `Lazarus` recovery | ✅ proven |
| `WubuProofs/NestedHyperbolicSpaces.lean` | Nesting-chain curvature/scale invariants; φ-progression | ✅ proven |
| `WubuProofs/MLACompression.lean` | Low-rank KV compression factor identity | ✅ proven |
| `WubuProofs/FiberBundle.lean` | SO(3) Lie-algebra commutation (`[Lx,Ly]=Lz`); bundle statement | ⚠️ partial (`wubu_is_principal_bundle := trivial`) |
| `WubuProofs/PoincareBall.lean` | `exp_0^c ∘ log_0^c = id`; ball metric conformal; `curvature_scaling` proven | ✅ proven (0 `sorry`) |
| `LeanCopies.lean` | Copies of the above (poincaré identity, Möbius, MLA, gyration) | ⚠️ 2 `sorry` in Möbius-closure alts |

## What is NOT proven (kept as notebook only)

- **`WubuProofs/PowerTower.lean`** — *excluded* from the imported
  library. Its `power_tower_not_integer` claim is **not a proof** (false
  lemma + 14 `sorry` + category error). See
  `../docs/theory/AUDIT_POWERTOWER.md`.

## Build

```bash
cd lean
lake build         # fetches mathlib4, builds WubuProofs
```

> Requires Lean 4 + a working `lake` toolchain and network access to
> fetch Mathlib. The 11 GB `.lake` build cache is **never committed**
> (see root `.gitignore`).

## Relationship to the C library

The C11 math in `src/` (quaternion, hyperbolic, nested encoder, Riemannian
SGD, parallel transport) is the *implementation*; this Lean project is the
*formal specification* of the hyperbolic/nesting core. Where they overlap
(Möbius addition, Poincaré maps), the Lean side is the source of truth.
