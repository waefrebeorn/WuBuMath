/-
WubuProofs.lean — Entry point for the Wubu Nested Hyperbolic Proofs library.

This library contains Lean 4 formal proofs of the core mathematical
claims underpinning the WuBu Nesting framework:

  1. PoincaréBall.lean — exp_0^c(log_0^c(y)) = y identity
  2. MobiusAdd.lean — Möbius addition preserves the Poincaré ball
  3. MLACompression.lean — Low-rank KV compression error bound
  4. HyperbolicGyration.lean — gyration preserves the unit ball
  5. PoincaréBallViz.lean — Visualization of Poincaré ball
  6. PowerTower.lean — Geometric non-integrality of π^(π^(π^π))
-/

import WubuProofs.PoincareBall
import WubuProofs.MobiusAdd
import WubuProofs.MLACompression
import WubuProofs.HyperbolicGyration
import WubuProofs.NestedHyperbolicSpaces
import WubuProofs.HolographicOptimizer
import WubuProofs.FiberBundle
import WubuProofs.PowerTower
import LeanCopies
/- PowerTower.lean (2026-07-10 revision): now a HONEST file of verified
   numerical bounds on the power tower π↑↑4 (log π > 1, π^π > 27,
   log L2 > e, log⁴N ∈ (0,1)), all proven with 0 `sorry`. The earlier
   FALSE lemmas and the logically-invalid "N ∉ ℤ" claim were removed;
   the integrality question is explicitly retired (see file header). -/
