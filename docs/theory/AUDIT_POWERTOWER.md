# Audit: `PowerTower.lean` (formerly claimed to prove N = π^(π^(π^π)) ∉ ℤ)

**Status: ILLUSTRATIVE NOTEBOOK ONLY — NOT A PROOF.** Migrated from
`bytropix/MATH/lean/.../PowerTower.lean` on 2026-07-10. The file is
**intentionally excluded** from the imported `WubuProofs` library
(see `lean/WubuProofs.lean`) and kept only as intuition.

## What was wrong (Triple Devil's-Advocate finding)

1. **A mathematically FALSE lemma.** The original `h_log_L2_lt_exp_1`
   asserted `log(L2) < e` where
   `L2 = π^π·log π + log(log π)`.
   Numerically: `log(L2) ≈ 3.7347` and `e ≈ 2.7183`, so the
   correct inequality is `log(L2) > e`. The lemma was marked `sorry`
   (unproven), so it was never checked — it just silently broke the chain.

2. **Entire proof rests on `sorry`.** `power_tower_not_integer` depended on a
   chain of 14 `sorry` lemmas (numerical bounds on log⁴ N, the
   "boundary holonomy" construction, etc.). None are established.

3. **Category error even if all lemmas held.** The argument shape was:
   "fractional part of log⁴(N) ≠ 0  ⟹  N ∉ ℤ".
   But `log⁴(n)` has a non-zero fractional part for *every* integer `n`
   (since it is generally non-integral). So that implication is vacuous —
   it cannot distinguish integer from non-integer N. The arithmetic
   (`fractional part ≈ 0.31766`) is correct; the *logical link to
   integrality* is not.

## What was fixed on migration

- `h_log_L2_lt_exp_1` (false `< e`) → replaced by `h_log_L2_gt_exp_1`
  asserting the true `> e` bound (still `sorry`, but now not false).
- `power_tower_not_integer` (a fake theorem) → renamed
  `power_tower_not_integer_ILLUSTRATIVE` and documented as a non-proof.
- Excluded from the proven library import.

## To make it a real proof (future work)

Proving `N ∉ ℤ` is genuinely hard. Correct approaches would not use
iterated-log fractional parts. Candidate real routes:
- Show `N` lies strictly between two consecutive integers via a rigorous
  interval estimate of `log⁴(N)` (needs a verified `Real.pi` bound and
  a Lean proof that `1 < log⁴ N < 2`, i.e. floor = 1 — which
  alone gives `N ∉ ℤ` directly, *without* any holonomy machinery).
- The nested-hyperbolic "ray tracing / holonomy" framing is decorative
  relative to that elementary interval argument.
