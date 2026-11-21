# WuBuMath — Manifold Domain Expansion Plan

## Vision
Shader cores (GPU) excel at custom C11 manifold operations. Hyperbolic/Poincare math
maps naturally to GPU parallelism. Goal: make manifold math the compression + physics
simulation layer, replacing heavy standard pipelines.

## What's Actually Built (verified compiling)
- Core math: hyperbolic, Poincare, gyrovector, exp/log maps, parallel transport, holonomy
- NN primitives: attention, FFN, conv1d/2d, RNN/LSTM, batchnorm, dropout, layernorm, residual
- Training: Adam, KFAC, Fisher/natural grad, grad clip, LR schedules, checkpointing, init
- Quantum/topo: Jones polynomial, Kauffman bracket, Coxeter, anyon, MITM search, topo filter
- Compression: CABAC context, CTW, psychoacoustic, bframe, codec_v2, cv4, gop_opt, qc_v3, qc_zlib
- I/O: tokenizer, config, profiler, pool allocator, RNG
- Eval: accuracy, perplexity, F1, benchmark timer

## What's Missing (real work, not stubs)
1. **GPU/shader integration** — no CUDA/OpenCL/Vulkan compute kernels yet. The C11
   manifold ops need to be ported to shader GLSL/HLSL for GPU execution.
2. **Lossless compression pipeline** — CABAC + CTW + trellis quantization exist as
   separate modules but aren't wired into a single encode/decode round-trip test.
3. **Manifold-native compression** — hyperbolic embedding for data compression
   (Poincare ball as latent space). No end-to-end test.
4. **Physics simulation** — Hamiltonian on hyperbolic space exists but no integrator
   (symplectic Euler/leapfrog on manifold). No N-body or rigid body.
5. **Test coverage** — 99 new modules have 0 tests. Existing tests cover ~30 modules.
6. **Makefile integration** — only 102 of 477 sources are in the build.

## Priority Order (code-review-first-then-plan)
1. **Wire real modules into Makefile + tests** (verify each compiles AND produces correct output)
2. **Build one lossless round-trip test** (encode → decode → bit-exact match)
3. **Add symplectic integrator** on Poincare ball (physics simulation foundation)
4. **GPU shader port** of top 5 hot paths (mobius_add, poincare_dist, exp/log, attention)
5. **Manifold compression benchmark** vs zlib/lzma on real data

## Immediate Next Steps
- Run `make all` equivalent with all sources to find remaining integration gaps
- Write round-trip test for CABAC+CTW compression
- Add symplectic Euler on Poincare ball (wubu_symplectic.h)
- Port mobius_add to GLSL compute shader

## Constraints
- Windows/MSYS, no make installed (use mingw32-make or direct gcc)
- zlib available (built from source)
- libjxl not available (skip jxl tests)
- RTX 4050 6GB (CUDA available but elevation needed for install)
EOF
echo done