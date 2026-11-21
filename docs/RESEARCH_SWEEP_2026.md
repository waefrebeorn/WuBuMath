# WuBuMath Research Sweep — 2026-11-02

Online sweep for expansion candidates. Topics aligned with existing
modules (hyperbolic, quaternion, manifold, anyon, Poincare, Lorentz,
Riemannian SGD, CABAC, flow matching).

## Hyperbolic / Poincare / Lorentz
- Poincare ball + Lorentz (hyperboloid) are isometric; Lorentz has
  unbounded manifold, no boundary singularities, cheaper distance (no
  arctanh), orthogonal-projection tangent structure. WuBuMath already
  has poincare_geom + lorentz + lorentz_poincare — add a unified
  model-switch layer (conformal map between ball & hyperboloid) and
  curvature-learning (learnable K, adaptive radius) per "Optimizing
  Curvature Learning" (2405.13979).
- Hyperbolic ResNet (Poincare convs + residual connections) and
  Hyperbolic Transformers (HexFormer, 2601.19849) are live targets.
  Candidate module: wubu_hyperbolic_conv.c, wubu_hyperbolic_attn.c.
- Higher-order simplicial embeddings (HHO, 2610.41684) — hyperbolic
  area of triangle as compactness measure. Candidate: wubu_simplicial_hyp.c.
- Biomedical KG link prediction (HGCN/AttH) shows HGCN > GCN at low
  dims — validates WuBuMath's low-dim hyperbolic story.

## Quaternions / SO(3)
- SO(3) has no global smooth minimal parametrization; quaternions
  double-cover S^3. Representations: Euler (gimbal lock), axis-angle
  (discontinuous at 0/2pi), quat (unit norm), 6D (Gram-Schmidt),
  9D (matrix), MRP (stereographic projection to R^3, open manifold).
- Key insight for WuBuMath: regress in tangent space (delta actions)
  avoids projection; per-step < pi/2 keeps Exp/Log well-behaved.
  Parallel transport of momentum between tangent spaces.
- Quaternion Product Units (QPU, CVPR 2020) — Hamilton-product chains
  give rotation-equivariant features. Candidate: wubu_qpu.c.
- RPMG (projective manifold gradient) backpropagates through the
  normalization map. Candidate integration into wubu_manifold_ad.c.

## Riemannian optimization
- RSGD: project Euclidean grad to tangent, retract, parallel-transport
  momentum. Exponential map expensive → use retraction.
- Adaptive variants: RGD-RMSProp, RGD-AdaDelta, Riemannian L-BFGS
  with SVRG. WuBuMath has riemannian_sgd — add adaptive + quasi-Newton.
- SAM on manifolds (NeurIPS 2023) — sharpness-aware minimization.
  Candidate: wubu_riemannian_sam.c.

## Anyons / topological QC
- SU(2)_k anyons: k>=3 (k!=4) universal by braiding. Fibonacci and
  Ising non-Abelian. Metaplectic (SO(3)_2) models — 6 variants.
- Solovay-Kitaev approximates arbitrary SU(2) from finite gate set
  with polylog length. WuBuMath has anyon module — add braidword
  compiler + SK approximator + fusion tree utilities.
- Controlled braiding via topological charge sectors (2410.14860).
  Candidate: wubu_topological_gate.c, wubu_solovay_kitaev.c.

## Parallel transport
- PTC (parallel transport convolution) on manifolds enables CNNs on
  curved domains. Centralize batches on SPD via parallel transport.
  WuBuMath has parallel_transport — extend to convolution kernels.

## CABAC / entropy coding
- CABAC: binarization → context model → arithmetic coder. Context-Tree
  Weighting (CTW) replaces FSM for +1.2-5.4% bitrate. DeepCABAC
  couples quantization with rate-distortion (lambda, eta robustness).
- Massively Parallel CABAC: syntax partitions + interleaved slices
  for throughput. WuBuMath has cabac + coeff + context — add CTW
  context model and rate-distortion quantizer (wubu_rd_quant.c).

## Flow / generative
- wubu_flow_matching + tangent_flow exist. Manifold flow matching
  on hyperbolic space is the natural extension (geodesic interpolants).

## Proposed new modules (priority order)
1. wubu_hyperbolic_conv.c        (Poincare/Lorentz convs)
2. wubu_riemannian_adam.c       (adaptive Riemannian opt)
3. wubu_solovay_kitaev.c        (topological gate approx)
4. wubu_ctw_context.c           (CTW context model for CABAC)
5. wubu_qpu.c                   (quaternion product units)
6. wubu_manifold_flow.c         (hyperbolic flow matching)
7. wubu_hyperbolic_attn.c       (hyperbolic attention)
8. wubu_rd_quant.c              (rate-distortion quantizer)

## Sources (arXiv / venues)
1705.10359, 2610.41684, 2609.18481, 2510.11103, 2110.11657,
2210.08126, 1902.03718, 1111.5280, 2507.17787, 2601.19849,
1001.4085, 2609.19484, 1802.06176, 2501.01745, 2410.14860,
1905.08318, plus HHI CABAC + CTW patent.
EOF
echo written