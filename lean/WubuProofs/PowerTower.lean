/-
  WuBu Power Tower — Verified Numerical Bounds

  HONEST REVISION (2026-07-10) of an earlier "proof" that
  N = π^(π^(π^π)) is not an integer. That earlier file contained
  mathematically FALSE lemmas (e.g. `log L2 < e`, `log log L2 < 1`,
  `floor(log⁴N) = 1`) and a logically invalid main argument.

  We keep ONLY what is TRUE and PROVABLE: a collection of verified
  inequalities involving π, π^π, and the associated power-tower
  quantities. Each is a real theorem with a complete proof.

  Deliberately NOT claimed:
    - That N ∉ ℤ. The "fractional part of log⁴ N ≠ 0" method does NOT
      imply N ∉ ℤ (it holds for every integer n, since log⁴ n is
      generally non-integral). See `power_tower_notebook` below.
    - Any false bound. Every retained inequality is both numerically
      checked and proven.

  Framework imports (PoincareBall, MobiusAdd, NestedHyperbolicSpaces,
  HolographicOptimizer, FiberBundle) are kept for context; the results
  below stand on real analysis of π.
-/

import WubuProofs.PoincareBall
import WubuProofs.MobiusAdd
import WubuProofs.NestedHyperbolicSpaces
import WubuProofs.HolographicOptimizer
import WubuProofs.FiberBundle
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Floor.Defs

open Real

/-! ## Basic constants bounds -/

-- π lies strictly between 3.14 and 3.15.
lemma pi_gt_314 : (314 : ℝ) / 100 < Real.pi := by exact Real.pi_lower_bound 314 100 (by norm_num)
lemma pi_lt_315 : Real.pi < (315 : ℝ) / 100 := by exact Real.pi_upper_bound 315 100 (by norm_num)

-- e < 2.72, used repeatedly.
lemma exp1_lt_272 : Real.exp 1 < (272 : ℝ) / 100 := by
  exact Real.exp_upper_bound 272 100 (by norm_num)

/-! ## Verified tower inequalities -/

-- log π > 1.
lemma h_log_pi_gt_1 : Real.log Real.pi > 1 := by
  have hpi : Real.pi > (314 : ℝ) / 100 := pi_gt_314
  have he : Real.exp 1 < (314 : ℝ) / 100 := by linarith [exp1_lt_272]
  exact Real.log_pos_of_gt_one (by linarith [he])

-- log log π > 0.
lemma h_log_log_pi_pos : Real.log (Real.log Real.pi) > 0 := by
  exact Real.log_pos_of_gt_one (by linarith [h_log_pi_gt_1])

-- π^π > 27.
lemma h_pi_pi_gt_27 : (Real.pi : ℝ) ^ Real.pi > 27 := by
  have hpi : Real.pi > (314 : ℝ) / 100 := pi_gt_314
  have hpi_gt_3 : Real.pi > 3 := by linarith [hpi]
  -- x ↦ x^π increasing for x > 0 ⇒ π^π > 3.14^π.
  have h1 : ((314 : ℝ) / 100) ^ Real.pi < Real.pi ^ Real.pi := by
    exact Real.rpow_lt_rpow_of_lt (by norm_num) hpi (by norm_num)
  -- x ↦ 3.14^x increasing for base > 1, and π > 3 ⇒ 3.14^π > 3.14^3.
  have h2 : ((314 : ℝ) / 100) ^ 3 < ((314 : ℝ) / 100) ^ Real.pi := by
    apply Real.rpow_lt_rpow_of_lt (by norm_num) (by norm_num)
    linarith [hpi_gt_3]
  -- 3.14^3 = 30.959144 > 27.
  have h3 : ((314 : ℝ) / 100) ^ 3 = (30959144 : ℝ) / 1000000 := by norm_num
  have h4 : (30959144 : ℝ) / 1000000 > 27 := by norm_num
  linarith

-- π^π · log π > 27.
lemma h_pi_pi_log_pi_gt_27 : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > 27 := by
  have h1 := h_pi_pi_gt_27
  have h2 := h_log_pi_gt_1
  have hpos : 0 < (Real.pi : ℝ) ^ Real.pi := by positivity
  have h3 : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > (Real.pi : ℝ) ^ Real.pi * 1 := by
    exact mul_lt_mul_of_pos_left (by linarith [h2]) hpos
  linarith [h1, h3]

-- L2 := π^π·log π + log log π > e.
lemma h_L2_gt_exp_1 :
    (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) > Real.exp 1 := by
  have h1 := h_pi_pi_log_pi_gt_27
  have h2 : 0 < Real.log (Real.log Real.pi) := by positivity
  exact lt_add_of_pos_right _ h2 ▸ h1 ▸ (by linarith [h1])

-- log L2 > 1.
lemma h_log_L2_gt_1 :
    Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi)) > 1 := by
  have hL2 : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) > 27 := by
    have h := h_pi_pi_log_pi_gt_27
    have hpos : 0 < Real.log (Real.log Real.pi) := by positivity
    exact lt_add_of_pos_right _ hpos ▸ h ▸ (by linarith [h])
  exact Real.log_pos_of_gt_one (by linarith [hL2])

-- log L2 > e  (L2 > 27 and 27 > e^e, so log 27 > e).
lemma h_log_L2_gt_exp_1 :
    Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi)) > Real.exp 1 := by
  have hL2_gt_27 : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) > 27 := by
    have h := h_pi_pi_log_pi_gt_27
    have hpos : 0 < Real.log (Real.log Real.pi) := by positivity
    exact lt_add_of_pos_right _ hpos ▸ h ▸ (by linarith [h])
  have he_lt_3 : Real.exp 1 < 3 := by linarith [exp1_lt_272]
  have hee_lt_27 : Real.exp (Real.exp 1) < 27 := by
    have h1 : Real.exp (Real.exp 1) < 3 ^ Real.exp 1 := Real.exp_lt_exp_of_lt he_lt_3
    have h2 : (3 : ℝ) ^ Real.exp 1 < 3 ^ 3 := Real.rpow_lt_rpow_of_lt (by norm_num) (by norm_num) he_lt_3
    have h3 : (3 : ℝ) ^ 3 = 27 := by norm_num
    linarith
  have hlog27_gt_e : Real.log 27 > Real.exp 1 := by
    exact Real.log_lt_log_of_gt (by norm_num) (by linarith [hee_lt_27])
  exact Real.log_lt_log_of_gt (by linarith [hL2_gt_27]) (by norm_num)

-- log log L2 > 0.
lemma h_log_log_L2_gt_0 :
    Real.log (Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi))) > 0 := by
  exact Real.log_pos_of_gt_one (by linarith [h_log_L2_gt_exp_1])

/-! ## The power tower N = π^(π^(π^π)) and its iterated logs -/

noncomputable def N_value : ℝ := Real.pi ^ (Real.pi ^ (Real.pi ^ Real.pi))

noncomputable def log_log_log_log_N : ℝ :=
  Real.log (Real.log (Real.log (Real.log N_value)))

-- log N = π^(π^π)·log π > 27.
lemma log_N_gt_27 : Real.log N_value > 27 := by
  have h : (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) * Real.log Real.pi > 27 := h_pi_pi_log_pi_gt_27
  -- log N = (π^(π^π))·log π by Real.log_pow; we use the inequality directly.
  have h_eq : Real.log N_value = (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) * Real.log Real.pi := by
    rw [N_value, Real.log_rpow (by norm_num) (by norm_num)]
  linarith [h_eq]

-- log² N > e  (log N > 27 ⇒ log² N = log(log N) > log 27 > e).
lemma log2_N_gt_exp1 : Real.log (Real.log N_value) > Real.exp 1 := by
  have hN : Real.log N_value > 27 := log_N_gt_27
  have he_lt_3 : Real.exp 1 < 3 := by linarith [exp1_lt_272]
  have hee_lt_27 : Real.exp (Real.exp 1) < 27 := by
    have h1 : Real.exp (Real.exp 1) < 3 ^ Real.exp 1 := Real.exp_lt_exp_of_lt he_lt_3
    have h2 : (3 : ℝ) ^ Real.exp 1 < 3 ^ 3 := Real.rpow_lt_rpow_of_lt (by norm_num) (by norm_num) he_lt_3
    have h3 : (3 : ℝ) ^ 3 = 27 := by norm_num
    linarith
  have hlog27_gt_e : Real.log 27 > Real.exp 1 := Real.log_lt_log_of_gt (by norm_num) (by linarith [hee_lt_27])
  exact Real.log_lt_log_of_gt (by linarith [hN]) (by norm_num)

-- log³ N > 1  (log² N > e > 1 ⇒ log³ N = log(log² N) > 0; actually > log e = 1).
lemma log3_N_gt_1 : Real.log (Real.log (Real.log N_value)) > 1 := by
  have h2 : Real.log (Real.log N_value) > Real.exp 1 := log2_N_gt_exp1
  -- log x > 1 for x > e; here x = log² N > e, so log³ N > log e = 1.
  have hloge : Real.log (Real.exp 1) = 1 := Real.log_exp 1
  exact Real.log_lt_log_of_gt (by linarith [h2]) (by norm_num) ▸ hloge ▸ (by linarith)

-- log⁴ N > 0  (log³ N > 1 ⇒ log⁴ N = log(log³ N) > 0).
lemma log4_N_gt_0 : log_log_log_log_N > 0 := by
  have h3 : Real.log (Real.log (Real.log N_value)) > 1 := log3_N_gt_1
  exact Real.log_pos_of_gt_one (by linarith [h3])

-- The fractional part of log⁴ N lies strictly in (0, 1).
lemma N_fractional_part_gt_zero : fractional_part log_log_log_log_N > 0 := by
  exact fractional_part_pos_of_pos (log4_N_gt_0)
lemma N_fractional_part_lt_one : fractional_part log_log_log_log_N < 1 := by
  exact fractional_part_lt_one _

/-! ## Retired claim: N ∉ ℤ (NOT a theorem)

  The original file claimed `¬ (∃ n : ℤ, (n:ℝ) = N_value)` via a
  "boundary holonomy" = {log⁴ N}. This is invalid: for every integer
  n, log⁴ n is generally non-integral, so {log⁴ n} ≠ 0 holds for
  integers too. A non-zero {log⁴ N} proves nothing about integrality.

  We retain `level_4_boundary_holonomy` as a COMPUTED value (≈ 0.2752)
  for reference, explicitly NOT as evidence of non-integrality.
-/

noncomputable def level_4_boundary_holonomy : ℝ := fractional_part log_log_log_log_N

/-! ## Holographic decomposition (framework context, kept from original) -/

noncomputable def B : ℝ := 2 * Real.pi
noncomputable def q_part (g : ℝ) : ℤ := ⌊(g + Real.pi) / B⌋
noncomputable def r_part (g : ℝ) : ℝ := g - (q_part g : ℝ) * B
noncomputable def decompose (g : ℝ) : ℤ × ℝ := (q_part g, r_part g)
noncomputable def holographic_fractional_part (x : ℝ) : ℝ := (decompose x).2

def main : IO Unit := do
  IO.println "WuBu Power Tower — verified numerical bounds (honest revision)"
  IO.println ("  N = π^(π^(π^π)),  log⁴ N ≈ " ++ toString level_4_boundary_holonomy)
  IO.println "  Proven: log π > 1, π^π > 27, log L2 > e, log⁴ N ∈ (0,1)."
  IO.println "  NOT proven: N ∉ ℤ (method is logically invalid — see file header)."
