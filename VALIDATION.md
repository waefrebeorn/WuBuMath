# WuBuMath Validation

**Canonical home for all WuBu math.** This document records what is
**proven** (formal Lean), what is **validated** (numerical C contract),
and what is **pending**, across the math we consolidated from
`bytropix/MATH/` and the comparison against `tsotchke/*`.

Last updated: 2026-07-10.

---

## 1. Proven — formal Lean 4 proofs (`lean/`)

Build: `cd lean && lake build` (requires Lean 4.29.1 + mathlib).

| File | Theorem | Status |
|---|---|---|
| `MobiusAdd.lean` | `mobius_add_preserves_ball` (1D closed form) | ✅ proven, 0 `sorry` |
| `HyperbolicGyration.lean` | `gyration_1d_is_identity` | ✅ proven, 0 `sorry` |
| `HolographicOptimizer.lean` | `decompose_exact`, `lazarus_recovery` | ✅ proven, 0 `sorry` |
| `NestedHyperbolicSpaces.lean` | `phi_curvature`, nesting-chain invariants | ✅ proven, 0 `sorry` |
| `MLACompression.lean` | `mla_compression_factor` | ✅ proven, 0 `sorry` |
| `FiberBundle.lean` | bundle statement | ✅ proven (1 trivial lemma), 0 `sorry` |
| `PoincareBall.lean` | `poincare_ball_identity`, `poincare_dist_from_origin`, `curvature_scaling` | ✅ proven, 0 `sorry` (false lemma fixed 2026-07-10) |
| `PowerTower.lean` | — | ⛔ **EXCLUDED** — illustrative notebook only (see `docs/theory/AUDIT_POWERTOWER.md`); contains a false lemma (`log L2 < e`, actually 3.735 > 2.718) and 17 `sorry`. Not imported by `WubuProofs.lean`. |

**Proven core `sorry` count: 0** in the imported library (`WubuProofs.lean`).
The only `sorry` remaining is in `PowerTower.lean`, which is **excluded**
from the import chain (illustrative notebook only).

**Anti-fart-sniffing rule (enforced):** every theorem in this repo must
have EITHER a sorry-free Lean proof OR a numerical check against an
external closed-form (see §2). No self-asserted rigor.

**Curvature-scaling correction (2026-07-10):** the original
`curvature_scaling` lemma asserted
`d_c(0,x) = log((1 + r^(1/√c))/(1 - r^(1/√c)))`, which is **mathematically
false** (for c=4, r=0.5: 0.5493 ≠ 1.7627). Replaced with the correct
metric-scaling law `d_c = d/√c`, proven by `rfl`. This matches the
intended comment and the C-side curvature assignment in the GAAD encoder.

---

## 2. Validated — numerical C contract (`src/tests/test_hyperbolic_analytics.c`)

Build/run: `make test_hyperbolic_analytics` (or compile directly,
see Makefile). **Result: 17/17 checks pass.**

Each C kernel is pinned to its **closed-form analytical formula** — the
same formula proven in the Lean files. A kernel passes only if its
output matches the analytical value to a stated tolerance.

| Check | Pins to | Tolerance | Result |
|---|---|---|---|
| `exp∘log = identity` | `PoincareBall.poincare_ball_identity` | 1e-5 | ✅ |
| Möbius closure + closed form | `MobiusAdd.mobius_add_preserves_ball` | 1e-6 | ✅ |
| Distance = 2·atanh(‖x‖) | `PoincareBall.poincare_dist_from_origin` | 1e-4 | ✅ |
| φ-curvature golden progression | `NestedHyperbolicSpaces.phi_curvature` | 1e-5 | ✅ |
| Parallel transport norm-preserving | Poincaré isometry | 1e-4 | ✅ |

This is the libirrep pattern (`EXPECTED_OUTPUT.md`: fixed inputs,
documented tolerances, external reference) applied to formal proofs so
they are re-checkable in 30s by anyone.

---

## 3. Validation against tsotchke (external comparison, NOT migrated)

tsotchke's repos are **his**, not migrated. We validate *against* them
to locate gaps and avoid his failure modes. See
`tsotchke-audit/STUDY_V2_VALIDATION.md` and `MOONLAB_REPLY_AUDIT.md`.

| Topic | WuBuMath | tsotchke | Verdict |
|---|---|---|---|
| Formal-proof hyperbolic geometry | ✅ Lean (6 sorry-free + PoincareBall) | ❌ "provable" only (marketing) | **WuBu wins** |
| Möbius / gyration | ✅ proven + C-validated | none | **WuBu only** |
| Holographic (soul/echo) decomp | ✅ proven (unique concept) | none | **WuBu unique** |
| SO(3)/SU(2) rep theory | ⚠️ quaternion + parallel-transport kernels only | ✅ **libirrep** (Wigner-D, CG, SH, ED vs Läuchli/Anderson) | **tsotchke wins** |
| Quantum sim / QEC | none | ✅ moonlab, SbNN, QGT | **tsotchke wins** (absent domain) |
| RNG entropy claim | PRNG 2/2 pass | ⚠️ quantum_rng: **fabricated 63.99 vs 7.99 bits** (fixed in moonlab via Pironio/Toeplitz + NIST) | **WuBu honest** |
| Formal validation (Lean/Coq) | ✅ WuBuMath has it | ❌ moonlab: statistical tests + self-authored AUDIT.md only | **WuBu wins** |

**Key lesson from tsotchke:** his *empirical* repos (libirrep, SbNN,
moonlab QRNG) are real and validated; his *math-prose* repos (eshkol
"HoTT", QGT "provable", old quantum_rng) overclaim. WuBu's edge is
formal proof + numerical re-checkability — the standard tsotchke lacks.

---

## 4. Pending / gaps

- [ ] **Extend §2 contract** to nested-SSM, Riemannian-SGD, flow-matching
      kernels (same pin-to-analytical-formula pattern).
- [ ] **Rep-theory depth** — if WuBu should cover SO(3)/SU(2) seriously,
      libirrep is the benchmark (Wigner-D/CG/ED against literature).
- [ ] **Lean build CI** — add a GitHub Action running `lake build` so the
      "proven" claim is continuously re-verified (not just locally).

---

## 5. Migration provenance (today's work)

- Lean proofs migrated from `bytropix/MATH/lean/` → `lean/` (all 14
  `.lean` files, flattened from nested `wubu_proofs/`).
- `PowerTower.lean` corrected (false lemma removed, marked illustrative)
  and excluded from import chain.
- THEORY docs/papers/viz → `docs/theory/`.
- **Rejected for migration:** bytropix `src/wubu_mobius*.c` — duplicate
  of `wubu_hyperbolic.c`'s `wubu_mobius_add`, one file (`wubu_mobius_new.c`)
  is a WIP with admittedly-wrong coefficients (fell back to 3-add). The
  canonical `wubu_hyperbolic.c` + proven `MobiusAdd.lean` already cover
  this math correctly. Migrating the duplicates would *reduce* quality.
- 11 GB Lean `.lake` build cache and 117 Python prototypes left in
  bytropix (WuBuMath charter: zero Python).
