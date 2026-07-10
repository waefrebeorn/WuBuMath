import Lake
open Lake DSL

package wubu_proofs where
  -- Canonical home of the WuBu Nesting formal proofs.
  -- Migrated from bytropix/MATH/lean (2026-07-10).
  require mathlib from git
    "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.1"

lean_lib WubuProofs where
  -- Proven library (no `sorry` in the imported core).
  -- PowerTower.lean is excluded: see note in WubuProofs.lean.
