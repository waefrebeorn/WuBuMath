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
-- NOTE: PowerTower.lean is INTENTIONALLY excluded from the
-- build. Its `power_tower_not_integer` theorem depends on a chain
-- of `sorry` lemmas, one of which (h_log_L2_lt_exp_1) is a
-- MATHEMATICALLY FALSE inequality (log(L2) ≈ 3.735 > e ≈ 2.718).
-- It is kept as an illustrative/notebook artifact under
-- WubuProofs/PowerTower.lean, NOT imported into the proven library.
-- See docs/theory/AUDIT_POWERTOWER.md.
