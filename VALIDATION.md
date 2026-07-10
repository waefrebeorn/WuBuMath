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
| `FiberBundle.lean` | SO(3) commutation (Lx,Ly,Lz), `so3_contains_identity`, `so3_closed_under_compose`, `wubu_is_principal_bundle` (real proofs — placeholder removed) | ✅ proven, 0 `sorry` (transpose fixed 2026-07-10; placeholder replaced with real group-closure theorem) |
| `PoincareBall.lean` | `poincare_ball_identity`, `poincare_dist_from_origin`, `curvature_scaling` | ✅ proven, 0 `sorry` (false lemma fixed 2026-07-10) |
| `PowerTower.lean` | Verified bounds on π↑↑4: `log π > 1`, `π^π > 27`, `log L2 > e`, `log⁴N ∈ (0,1)` | ✅ proven, **0 `sorry`** (honest revision 2026-07-10; false lemmas + invalid "N∉ℤ" claim removed) |

**Proven core `sorry` count: 0** across ALL 8 imported files (verified by a
**clean from-scratch `lake build`, 2026-07-10 — see note below**).

**IMPORTANT — cache-masking bug fixed (2026-07-10):** An earlier
"build success, 0 sorry" was a *false all-clear*. Two files
(`FiberBundle.lean`, `HolographicOptimizer.lean`) had real errors that
Lean's incremental cache had been skipping (stale `.olean`). A full
recompile (triggered by adding `PowerTower.integrality_via_nested_log`)
exposed them:
- `FiberBundle.lean` used the undefined `ᵀ` transpose notation → never
  compiled. Fixed: `Rᵀ` → `R.transpose` (3 sites).
- `HolographicOptimizer.remainder_in_range` claimed `-π < r` (strict
  lower bound) which is **false** at `g = π` (`r = -π`). Fixed: weak
  lower bound `-π ≤ r`.

Both now compile. The current `0 sorry` is backed by a **clean rebuild
with no cached modules**, so it is trustworthy.

**Anti-fart-sniffing rule (enforced):** every theorem in this repo
requires a sorry-free Lean proof OR a numerical check. Allegory
(WuBu Nesting as intuition) is documentation, not mathematics.

**PowerTower revision (2026-07-10):** The original `PowerTower.lean`
contained 17 `sorry` and several MATHEMATICALLY FALSE lemmas
(`log L2 < e` — actually 3.735 > 2.718; `log log L2 < 1` — actually
1.317; `floor(log⁴N) = 1` — actually 0). Its headline "N ∉ ℤ" argument
was also logically invalid (a non-zero `{log⁴ n}` holds for every
integer n). Revised to a HONEST file: 11 true inequalities are now
proven with **0 `sorry`** (log π > 1, log log π > 0, π^π > 27,
π^π·log π > 27, L2 > e, log L2 > 1, log L2 > e, log log L2 > 0,
log⁴N > 0, {log⁴N} ∈ (0,1)), and the integrality claim is explicitly
retired as non-theorem. PowerTower is now IMPORTED by `WubuProofs.lean`.

**Anti-fart-sniffing rule (enforced):** every theorem in this repo must
have EITHER a sorry-free Lean proof OR a numerical check against an
external closed-form (see §2). No self-asserted rigor. The PowerTower
revision is the proof of this rule working: a file that *looked* like a
proof but contained false lemmas was caught, stripped of its falsehood,
and rebuilt as real theorems — not papered over with more `sorry`.

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

## libirrep port (2026-07-10) — tsotchke → WuBuMath

`src/math/wubu_so3.c` + `include/wubu_so3.h` port the SO(3) Lie-group
exp/log/geodesic algorithms from `tsotchke/libirrep` (MIT) into WuBuMath's
`float`/`Rot3` convention. Validated numerically in
`src/tests/test_wubu_so3.c` (20000 random trials):
- `rot_exp(rot_log(R)) == R`: worst error **8.3e-7**
- `geodesic_distance(I, rot_exp(w)) == |w|`: worst error **4.8e-7**
- `rot_log(rot_exp(w)) == w` (|w|<π): worst error **3.9e-7**

This closes the gap where `FiberBundle.lean` previously had a fake
`bundle_projection` returning `0`. The Lean side now proves
`so3_closed_under_compose` (SO(3) is a group) and references the C
validation.

## libirrep rep-theory port (2026-07-10)

`src/math/wubu_rep_theory.c` + `include/wubu_rep_theory.h` port the Wigner 3j
symbol (Racah closed form) and Clebsch-Gordan coefficient from
`tsotchke/libirrep/src/clebsch_gordan.c` (MIT). Validated in
`src/tests/test_wubu_rep_theory.c` against libirrep's own outputs (10-digit
agreement) and the CG orthonormality relation Σ_J CG² = 1:
- CG(1,1;1,1→2,2) = 1, CG(1,0;1,0→0,0) = −1/√3, CG(1,1;1,−1→2,0) = 1/√6
- Wigner 3j(1,0;1,0;2,0) = √(2/15)
- three orthonormality checks PASS

This fills the rep-theory depth gap flagged earlier (Wigner-D / CG).

## Lean CI (2026-07-10)

`.github/workflows/lean.yml` runs `lake build` + a zero-`sorry` audit on every
push/PR. This structurally prevents the cache-masking false-all-clear that hid
the FiberBundle/HolographicOptimizer errors earlier: CI always builds from a
clean cache miss path and fails if any `sorry` appears.

Full tsotchke review + pick-and-choose vault: `../tsotchke/EXAMPLE_VAULT.md`
and `../tsotchke/REVIEW_SUMMARY.md`.
