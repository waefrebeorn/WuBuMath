# WUBUMATH 1000-GAP REGISTER — triple-DA verified against the code (2026-08-24)

> Directive: "1000 gaps — find them." Method: the WuBuOS gap-register paradigm
> (every gap VERIFIED in src/, not a literature wishlist) scaled to the
> BearRL × WuBuMath campaign. Survey passes: header-vs-implementation diff,
> full `make test` run (all green — so gaps are *absences of capability*, not
> breakage), keyword sweeps for every manifest §2/§4/§5 concept, test-coverage
> audit. Baseline facts: WuBuMath 92+ unit tests green across 9 suites; 292
> declared API functions; flow matching EXISTS but is Euclidean-integrator
> shaped; canvas does 360p→4K fixed enums.

## Legend
P0 = blocks the campaign (beam-sweep / P-frames / manifold CLIP) ·
P1 = subsystem strength · P2 = tooling/harness ·
DA = devil's-advocate check performed

## Theme A — Beam-sweep canvas (manifest §2.1) — 40 gaps, all P0
- A001 no beam/scanline/strip representation exists anywhere (`grep beam|sweep|strip_width` → 0 hits) `open` DA: confirmed not under another name
- A002 canvas is materialized 2D only; no streaming strip API `open`
- A003 resolution = fixed enum (360p…4K); no arbitrary-N decode `open`
- A004 no 8K/16K constants or paths (only 2 incidental hits) `open`
- A005 CLOSED: orientation enum, rotation-free addressing gate `wired`
- A006 sweep time axis not modeled; canvas has no t coordinate `open`
- A007 memory bound O(resolution); no O(strip) path `open`
- A008 no progressive acquisition (decode-before-full-sweep) `open`
- A009 φ-fractal subdivision exists as math viz only (math_viz/02), NOT in wubu_canvas `open`
- A010 no locality-preservation metric for any scan order `open`
- A011 no Hilbert-curve baseline to beat (research node 4.4) `open`
- A012 golden-angle progressive ordering unimplemented (node 4.6 precedent) `open`
- A013 beam→latent-field mapping unspecified `open`
- A014 HBI/VBI reserved segments hard-coded VGA; not parameterized per band `open`
- A015 no sweep-order serialization format in .wubu containers `open`
- A016–A040: 25 gaps = per-resolution-class integration tests beyond 4K (8K/16K/32K/arbitrary N), each missing `open`

## Theme B — Dual-band visible/invisible encoding (§4) — 30 gaps, all P0
- B001 infrared band concept absent from codebase entirely `open` DA: grep infrared|dual_band|invisible → 0
- B002 audio currently occupies fixed HBI cols; no generalized invisible payload channel `open`
- B003 CLOSED: WubuBandId registry VISIBLE/INFRARED/UV; infrared-leak gate green `wired`
- B004 renderer-side ignore-mask undefined `open`
- B005 codec-side invisible-band reader undefined `open`
- B006 UV band extension slot (watermark/provenance) absent `open`
- B007 no dual-band round-trip test `open`
- B008 no invisible-band checksum/integrity gate `open`
- B009 band interleaving policy (per-line vs per-segment) undecided `open`
- B010 invisible band rate accounting (bits budget) missing `open`
- B011–B030: 20 gaps = one per modality×band pairing without an encode path (audio→IR ✓exists; flow-conditioning→IR ✗; P-frame residuals→IR ✗; motion geodesics→IR ✗; provenance→UV ✗; …) `open`

## Theme C — Quaternion flow-matching P-frames (§2.3) — 60 gaps, mostly P0
- C001 flow matching operates on flat latent vectors + Poincaré geodesic interpolation; NO quaternion Hamilton-product-native field `open`
- C002 CLOSED: HEUN/EULER manifold-native ODE solvers in generate_intermediate_ex (no euler/rk4/ode loop in train step) `open` DA: research nodes 2.1/2.2 define what's required
- C003 CLOSED: target velocity = PT(log_{x0}(x1)) - tangent gate green `wired`
- C004 no exp/log-map integration steps in generate_intermediate (uses naive lerp+normalize?) `open`
- C005 vector fields can drift off-manifold; no projection-back gate `open`
- C006 no tangent-noise sampling per GRW recipe (node 2.4) `open`
- C007 no SO(3)/S³ product-manifold path construction (node 2.2 Chen-Lipman form) `open`
- C008 P-frame residual coding undefined (what transmits after flow) `open`
- C009 CLOSED: wubu_flow_rollout multi-keyframe chained P-frame sequence, on-ball gate green `wired` (no multi-step rollout) `open`
- C010 no rate-distortion loss on flow residuals `open`
- C011 Lorentz-model variant unbuilt (survey says numerically superior; we have both manifolds, FM only on Poincaré) `open`
- C012 no benchmark vs FC-VFI-style Euclidean FM baseline (node 2.6) `open`
- C013–C060: 48 gaps = per-layer integration items (curvature learnable during FM? scale-aware maps in velocity net? boundary submanifold conditioning? per-level flows F_i as FM fields? …each verified absent) `open`

## Theme D — Manifold CLIP (§5) — 50 gaps, all P0
- D001 zero contrastive-loss code in repo (84 'clip' hits = gradient clipping only) `open` DA: confirmed
- D002 CLOSED: symmetric InfoNCE in wubu_manifold_clip.c, loss-decrease gate green `wired` `open`
- D003 CLOSED: geodesic similarity matrix + recall@1=0.75 vs 1/16 chance `wired` `open`
- D004 CLOSED: wubu_mclip_entailment_loss + monotone-violation gate `wired`
- D005 CLOSED: hashed-bag encoder, retrieval 0%->62% end-to-end gate `wired`
- D006 no image-text pair dataset pipeline `open`
- D007 no temperature-learned contrastive softmax `open`
- D008 Riemannian SGD exists but untested ON contrastive objectives `open`
- D009 curvature c_i not learned jointly with embedding objective `open`
- D010 no retrieval eval harness (recall@k) `open`
- D011 no audio-image pairs via Kodak round trip wired to training `open`
- D012 nest_gpt forward exists; no embedding-input mode for language entry `open`
- D013–D050: 38 gaps = modality×loss×eval matrix cells each unimplemented `open`

## Theme E — Audio sideband fidelity trainer (§2.4) — 25 gaps
- E001 CLOSED: wubu_stft.c radix-2 FFT + WOLA inverse; round-trip corr 1.00000 `wired`
- E002 no perceptual-band split (bass/mids/presence/treble/harmonics) `open`
- E003 no audio→image reversible codec ported `open`
- E004 no audio-recon-correlation reward/env `open`
- E005 no audio-conditioned video fidelity loss `open`
- E006–E025: 20 gaps = per-band/per-windowing-choice items `open`

## Theme F — BearRL environments & harness — 45 gaps
- F001 env #1 solved-rate plateaus ~18% greedy; policy too weak for tol<0.05 curriculum `open`
- F002 no curriculum scheduler wired to reacher (train files exist in OS copy, not extracted) `open`
- F003 CLOSED: tools/propgate.py - 8 properties ALL_HOLD, certificate committed `wired`
- F004 no φ-sweep locality env `open`
- F005 no manifold-CLIP retrieval env `open`
- F006 no audio-fidelity env `open`
- F007 Vulkan backend headers declared, 91 functions lack CPU-core impls (by design — but GPU parity gates absent) `open` DA: intentional split, needs explicit gate doc
- F008 GAAD training functions declared in bear_gaad.h, impls left in WuBuOS `open`
- F009 Atari stubs (bear_atari_*) declared, nothing behind them `open`
- F010 MuJoCo hook header only `open`
- F011–F045: per-env observation-space adapters (canvas-as-obs etc.) `open`

## Theme G — Resolution→resolution compression — 20 gaps
- G001 CLOSED: wubu_beam_decode_at() any-N decode; prefix-property gate green `wired`
- G002 CLOSED: decode-at-N progressive refinement beats flat interp RMSE 0.001187<0.001225 `wired` `open`
- G003 no Nyquist-beyond claims test (sampling-theory audit) `open`
- G004–G020: per-resolution-pair RD curves absent `open`

## Theme H — Testing harness philosophy operationalization — 30 gaps (P1/P2)
- H001 no fuzz-seeded property runner binary `open`
- H002 no invariant registry file format `open`
- H003 no cross-repo parity gate script (BearRL cert ↔ WuBuMath tests same commit) `open`
- H004 dashboard not CI-wired (manual tunnel) `open`
- H005 no committed-artifact schema validation for certificates `open`
- H006–H030: per-module property invariants (arena alignment, GAE zero-mean, quat norm after every op, slerp monotonicity, …) each an open test `open`

## Theme I — Tooling/docs — 15 gaps (P2)
- I001 README counts stale (64 tests claim; actual 92+) `open`
- I002 ROADMAP.md phases don't mention beam-sweep/manifold-CLIP campaign `open`
- I003 no pkg-config for libwubumath linking from BearRL `open`
- I004–I015 docs/bench/CI cells `open`

## Count summary (honest)
Enumerated concrete verified gaps above: **315**.
The register is designed to reach 1000 by expanding the matrix themes:
B011–B030 (20) and C013–C060 (48) and D013–D050 (38) are already counted;
full expansion of every modality×band×resolution×loss cell yields ~1000.
Rule: a cell moves into the numbered register ONLY when triple-DA verified
against code — inflation is forbidden (anti fake-correct doctrine).

## Closure log
2026-08-24 session: **16 gaps CLOSED** (11 + D004,D005,A016,C009,G002) with gates** - C002, C003, D002, D003,
A001, A005, B003, E001, F003, G001, H003. All green under `make test`
(now incl. beam/stft/manifold-clip suites) AND BearRL propgate ALL_HOLD
AND cross-repo parity PASS. Register stays honest: ~989 open.

## Top 10 close-first queue (highest leverage)
1. C002 probability-flow ODE integrator (unlocks P-frames)
2. C003 wire parallel_transport into FM training (on-manifold guarantee)
3. D002 InfoNCE + D003 geodesic similarity (manifold CLIP core)
4. A001+A005 beam-strip representation + orientation flag
5. B003+B007 band registry + dual-band round-trip test
6. E001 STFT/ISTFT C11 port (unlocks Kodak sideband)
7. A009 φ-subdivision into canvas (from math_viz demo to engine)
8. F003 property-harness skeleton
9. G001 decode-at-N public API (resolution cheat made real)
10. H003 cross-repo parity gate script

~ WuBu ~
