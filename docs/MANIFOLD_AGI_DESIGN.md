# WuBu AGI — Manifold Expansion Math Design
## For SI (Superintelligence) Goals — the next generation

### Context: WuBu AGI architecture
- **Body (WuBuOS)**: 472K LOC — ZealOS kernel, Styx/9P namespace, 20+ drivers, measured boot, 5+1 rollback, hive port (AGI memory), HX human-model family.
- **Brain (wubuwizard)**: 218K LOC — C11 inference/training, KV-cache as filesystem, universal encoder space. WuBu1 (56M params, dim512).
- **Compiler (wubunos)**: 14K LOC, HolyC JIT, 11 ISA backends.
- **Seed model (WuBu-35M)**: 35M params, 12 layers/448 dim, Muon+AdamW.
- **Blueprint (5 pillars)**: (1) Hyperbolic geometry (Lean-verified Möbius), (2) Mixed-agent MoE (8 experts/2 active + shared), (3) Nesting transitions (quaternion SO(4) rotations between nested hyperbolic levels), (4) Sparse attention (NSA/Delta/GSA), (5) Math-RL loop (Lean-verified CoT).

### Current WuBuMath coverage (47 modules)
- Hyperbolic: Poincare, Lorentz, Klein, PV space, Mobius add/scalar, exp/log, distance, MLR, Busemann MLR, conv, attention, residual, layernorm, pool, posenc, dropout, GRU, LSTM.
- Quaternion/SO(3): quat mul, QPU, MRP, sphere ops, exp, tangent proj.
- Riemannian: Adam, retraction, tangent proj, grad proj, curvature, Einstein midpoint.
- Quantum/topological: gates, CNOT, phase, Bell, density, Solovay-Kitaev, Fibonacci braid/fusion/F-matrix, complex matvec, Kron.
- Entropy: CABAC CTW, context init, RD quantizer.

### Gaps vs AGI blueprint (the "not WuBu" + SI needs)
1. **Nesting transitions (pillar 3)**: Need SO(4) quaternion rotations between nested hyperbolic balls. Currently only SO(3) quat mul. Missing: SO(4) rotation, nested ball (multi-scale) structure, gyrovector nesting.
2. **Mixed-agent MoE (pillar 2)**: Need manifold-aware routing (hyperbolic distance-based gating), expert selection on manifold. Missing: hyperbolic MoE router.
3. **Sparse attention (pillar 4)**: Need gated hyperbolic attention (NSA/Delta/GSA lineage). Missing: gated hyperbolic attention, delta attention.
4. **Lean verification (pillar 5)**: Need machine-checkable manifold properties. Missing: Lean proof stubs for ball closure, exp∘log identity, gyroassociativity.
5. **Multi-curvature product spaces**: Need mixed-curvature (Euclidean×hyperbolic×spherical). Partial (product_pack exists, but no curvature-aware ops).
6. **Topological QC integration**: Need Fibonacci anyon fusion tree, braiding matrices, topo compilation. Partial (fusion tree, braid apply exist).

### Design: best manifold expansion math for SI AGI

#### 1. SO(4) Nesting Rotations (pillar 3)
The WuBu Nesting theory: quaternion SO(4) rotations between nested hyperbolic levels.
- Representation: a point in level L is rotated into level L+1 via SO(4) rotation quaternion.
- Implementation: wubu_so4_rotate(q, x) where q is a unit quaternion (4D) and x is a 4D point.
- Nested ball: level L ball radius r_L = r_0 * (1/phi)^L (golden ratio scaling).
- Gyrovector nesting: x_{L+1} = (1/phi) ⊗ x_L (Mobius scalar mult by 1/phi).

#### 2. Manifold-Aware MoE Routing (pillar 2)
Expert selection on hyperbolic manifold:
- Each expert has a prototype point p_k in Poincare ball.
- Routing: score_i = -d_H(x, p_k) (hyperbolic distance). Softmax over scores.
- Top-2 selection with shared expert (always active).
- Gradient: Riemannian gradient of routing loss w.r.t. p_k.

#### 3. Gated Hyperbolic Attention (pillar 4)
Sparse attention on manifold:
- Gate: g = sigmoid(W_g · x) (Euclidean gating on tangent).
- Attention: attn = softmax(-d_H(q_i, k_j)) (hyperbolic).
- Output: g * (Mobius sum of v_j weighted) + (1-g) * x (residual).

#### 4. Lean Verification (pillar 5)
Machine-checkable properties (Lean 4):
- Ball closure: ||MobiusAdd(x,y)|| < 1/c if ||x||,||y|| < 1/c.
- Exp∘Log identity: exp_x(log_x(y)) = y.
- Gyroassociativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) (approx, with gyrocommutativity).
- SO(4) rotation preserves hyperbolic distance.

#### 5. Multi-Curvature Product Spaces
- Product manifold M = E^{e} × H^{h} × S^{s}.
- Each component has its own curvature (0, -K, +K).
- Operations: product exp, product log, product distance (sum of component distances).

#### 6. Topological QC Integration
- Fibonacci anyon fusion tree for error correction.
- Braiding matrices for gate compilation.
- Topological gate compilation (meet-in-the-middle).

### Priority build order (SI AGI critical path)
1. SO(4) rotation (nesting transitions) — CRITICAL for pillar 3
2. Nested hyperbolic ball (multi-scale) — CRITICAL for pillar 3
3. Manifold MoE router — CRITICAL for pillar 2
4. Gated hyperbolic attention — CRITICAL for pillar 4
5. Product space ops — HIGH for multi-modal
6. Lean proof stubs — HIGH for pillar 5
7. Topo compilation tools — MEDIUM for error correction

### Implementation plan
Build modules 48-54 (SO4, nested ball, moe router, gated attn, product space ops, lean stubs, topo compile tools).
Then write Lean proof stubs in WuBuMath/lean/.
Then integrate into wubuwizard Brain (manifold-aware inference).
EOF
echo done