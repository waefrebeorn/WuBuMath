# WuBuMath Roadmap

> **The home of all WuBu数学 (WuBu Math)** — slermed from Python/JAX to pure C11.

## Repository Structure

```
WuBuMath/
├── include/              # C11 headers
├── src/
│   ├── math/             # Core math: color manifold, positional encoding, utils
│   ├── model/            # Neural models: Hamilton encoder, VHF decoder
│   ├── train/            # Training: Q-controller, loss manifold
│   ├── jax/              # Slermed JAX core (arena, SIMD, GEMM, MLP, lax, IR)
│   └── tests/            # Test suite
├── examples/             # Media creator, demos
├── vault/                # Python source of truth (slermed FROM here)
│   └── python/
│       ├── encoders/     # Phase 1-3 encoders, Hamilton, hash-mind
│       ├── audio/        # VHF audio pipeline, wubusynth
│       ├── attention/    # Hyperbolic, sparse, entropic, topological
│       ├── optimizers/   # Q-controller, PID
│       ├── diffusion/    # Funnel diffusion, CLIP video
│       ├── math/lean/    # Lean 4 formal proofs
│       └── theory/       # Foundational philosophy, nesting paper
└── ROADMAP.md            # This file
```

## Progress

### Phase 0: Foundation ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| Arena allocator | jax source | `src/jax/jax_arena.c` | 1/1 | ✅ |
| SoA Tensors | jax source | `src/jax/jax_arena.c` | 3/3 | ✅ |
| SIMD (AVX2/NEON) | jax source | `src/jax/jax_simd.c` | 2/2 | ✅ |
| GEMM kernels | jax source | `src/jax/jax_nn.c` | 1/1 | ✅ |
| Activations | jax source | `src/jax/jax_nn.c` | 1/1 | ✅ |
| MLP forward/back | jax source | `src/jax/jax_nn.c` | 2/2 | ✅ |
| Optimizer (Adam) | jax source | `src/jax/jax_opt.c` | 1/1 | ✅ |
| Lax ops | jax source | `src/jax/jax_lax.c` | 15/15 | ✅ |
| Jaxpr IR | jax source | `src/jax/jax_ir.c` | 1/1 | ✅ |
| **JAX Total** | | | **35/35** | ✅ |

### Phase 1: Color Manifold ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| RGB→HSL | `vhf_audio.py:rgb_to_hsl_jax` | `src/math/wubu_color.c` | 4/4 | ✅ |
| HSL→RGB | `vhf_audio.py`(implicit) | `src/math/wubu_color.c` | 1/1 | ✅ |
| Circular L1 Loss | `vhf_audio.py:circular_l1_loss` | `src/math/wubu_color.c` | 3/3 | ✅ |
| Grayscale | `phase1_encoder:grayscale` | `src/math/wubu_color.c` | 1/1 | ✅ |
| Positional Encoding | `vhf_audio.py:PositionalEncoding` | `src/math/wubu_positional_encode.c` | 1/1 | ✅ |
| Bilinear Sampling | `vhf_audio.py:map_coordinates` | `src/math/wubu_utils.c` | — | ✅ |
| Box Blur 5×5 | `phase1_encoder:convolve2d` | `src/math/wubu_utils.c` | — | ✅ |
| **Color Total** | | | **9/9** | ✅ |

### Phase 2: VHF Audio Pipeline ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| Audio strip gen | `vhf_audio.py:video_audio_generator` | `src/model/wubu_vhf_audio.c` | 1/1 | ✅ |
| Canvas compositing | `vhf_audio.py:eval_canvases` | `src/model/wubu_vhf_audio.c` | 1/1 | ✅ |
| Audio strip→WAV | `vhf_audio.py`(implicit) | `examples/media_creator.c` | — | ✅ |
| **VHF Total** | | | **2/2** | ✅ |

### Phase 3: Hamilton Encoder ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| Quaternion latent | `vhf_audio.py:HamiltonEncoder` | `src/model/wubu_hamilton_encoder.c` | 1/1 | ✅ |
| VHF Decoder | `vhf_audio.py:VHFDecoder` | `src/model/wubu_vhf_decoder.c` | 1/1 | ✅ |
| Hamilton encode/decode | `vhf_audio.py:VHFEndToEndModel` | `src/model/wubu_vhf_audio.c` | 2/2 | ✅ |
| **Hamilton Total** | | | **4/4** | ✅ |

### Phase 4: Training Pipeline ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| Q-Controller | `vhf_audio.py:QControllerState` | `src/train/wubu_q_controller.c` | 3/3 | ✅ |
| HSL Loss Manifold | `vhf_audio.py:train_step` | `src/train/wubu_loss.c` | 2/2 | ✅ |
| Training step | `vhf_audio.py:train_step` | `src/model/wubu_vhf_audio.c` | 1/1 | ✅ |
| **Training Total** | | | **6/6** | ✅ |

### Phase 5: Media Generation ✅ COMPLETE

| Component | Python Source | C11 Slerm | Tests | Status |
|-----------|--------------|-----------|-------|--------|
| Color frame gen | `phase1_encoder_update.py` | `examples/media_creator.c` | — | ✅ |
| Shape frame gen | `phase1_encoder_update.py` | `examples/media_creator.c` | — | ✅ |
| VHF tone gen | `vhf_audio.py` | `examples/media_creator.c` | — | ✅ |
| WAV writer | (new) | `examples/media_creator.c` | — | ✅ |
| **Media Total** | | | **—** | ✅ |

---

## TODO: Next Phases (Not Yet Slemmed)

### Phase 6: Phase 1 Symmetric Encoder
- [ ] `phase1_encoder_update.py` → `src/encoders/phase1.c`
  - [ ] Complex mask generation (ellipse, rect, boolean ops)
  - [ ] Synthetic RGBA texture batch
  - [ ] Color pattern generation (32 frames)
  - [ ] Moving shape pattern generation
  - [ ] VHF tone pipeline (8 tones → WAV)
  - [ ] Poincare sphere co-polarized transmittance
  - [ ] FiLM layer
  - [ ] Coordinate decoder MLP

### Phase 7: Phase 2 Topological AE
- [ ] `QAE_Advanced.py` → `src/encoders/phase2.c`
  - [ ] Topological quantization
  - [ ] Persistent homology features
  - [ ] AE trainer

### Phase 8: Phase 3 Generative
- [ ] `phase3_generative.py` → `src/encoders/phase3.c`
  - [ ] Corpus builder
  - [ ] Tokenizer training
  - [ ] Generation pipeline

### Phase 9: Full Hamilton CPU Encoder
- [ ] `hamilton-encoder-cpu/*.py` → `src/encoders/hamilton.c`
  - [ ] Geodesic spiral layers
  - [ ] Quaternion monolith
  - [ ] Neural field
  - [ ] Chimera ResNet
  - [ ] Crystalline architecture

### Phase 10: HashMind
- [ ] `hash-mind/*.py` → `src/encoders/hashmind.c`
  - [ ] WuBuMind hash encoder (V1-V7)
  - [ ] Nesting trainer
  - [ ] Galactic core VAE+LQR

### Phase 11: Attention Mechanisms
- [ ] `hyperbolic-attention/*.py` → `src/attention/hyperbolic.c`
  - [ ] TgT test
  - [ ] Clockwork (fixed, stabilized, perceptron)
- [ ] `WuBuSparseAttention.py` → `src/attention/sparse.c`
- [ ] `xjdr_backup_sampler.py` → `src/attention/entropix.c`

### Phase 12: Diffusion
- [ ] `funnel-diffusion/*.py` → `src/diffusion/funnel.c`
  - [ ] CLIP video encoder
  - [ ] Funnel denoising
  - [ ] Inference pipeline

### Phase 13: Lean Proofs (Reference Only)
- [ ] `MATH/lean/wubu_proofs/` → Documented in `vault/`
  - [ ] PoincareBall.lean
  - [ ] HyperbolicGyration.lean
  - [ ] MobiusAdd.lean
  - [ ] MLACompression.lean
  - [ ] Basic.lean

### Phase 14: Theory (Reference Only)
- [ ] Document all theory papers
  - [ ] Foundational philosophy
  - [ ] Axiomatic emergent theory
  - [ ] WuBu nesting paper
  - [ ] Spatio-temporal findings
  - [ ] WuBuHypCD (LaTeX)

### Phase 15: Formal Verification Bridge
- [ ] Port Lean proofs to C11 assertions
- [ ] Property-based testing (arena contracts, quaternion normalization)
- [ ] Proof-carrying code

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| WuBuMath core | 29/29 | ✅ |
| Slermed JAX | 35/35 | ✅ |
| Phase 1 encoder | 0/40 | ⬜ |
| Phase 2 topological | 0/15 | ⬜ |
| Phase 3 generative | 0/20 | ⬜ |
| Hamilton CPU | 0/50 | ⬜ |
| HashMind | 0/30 | ⬜ |
| Attention | 0/25 | ⬜ |
| Diffusion | 0/15 | ⬜ |
| **Total** | **64/322** | **20%** |

---

## Divergence History

WuBuMath was forked from [bytropix](https://github.com/waefrebeorn/bytropix) to separate **mathematical work** from **inference engine** work:

1. **Pre-May 10**: bytropix = all math (117 Python files in root)
2. **May 10** (`488b821`): Massive reorg into THEORY/MATH/ENCODERS/AUDIO/DIFFUSION/ATTENTION/OPTIMIZERS
3. **May 28** (`816aea8`): bytropix gains C inference server → divergence begins
4. **Post-May 28**: bytropix = inference engine + math (confusing)
5. **Now**: WuBuMath = all math, bytropix = inference engine only

---

## License

MIT License — WuBu Slermes Project
