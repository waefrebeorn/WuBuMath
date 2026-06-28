# WuBuMath

Pure C11 mathematical & media encoding library — the home for all WuBu数学 (WuBu Math).

Slermed from the VHF Hamilton Modulator pipeline and JAX compute core. Zero Python. Zero PyTorch. Zero external dependencies.

## What It Is

WuBuMath is the **dedicated home for all mathematical work** that was previously scattered across bytropix. It computes:

- **Color manifolds** (RGB/HSL, circular L1 loss, grayscale)
- **Hamilton encoder/decoder** (quaternion latent space for video+audio)
- **VHF audio pipeline** (canvas compositing, HBI audio strips, FM encoding)
- **Q-controller** (adaptive learning rate with warmup + exploration)
- **Positional encoding** (sin/cos frequency bands)
- **Slermed JAX** (full JAX-compatible compute core in C11)
- **Media generation** (color patterns, shape animations, VHF tones)

## Origin

Primary source: [`vhf_audio.py`](https://github.com/waefrebeorn/bytropix/blob/master/AUDIO/wubusynth/vhf_audio.py) from the [bytropix](https://github.com/waefrebeorn/bytropix) repository.

The divergence: bytropix started as the **math/training** repo (Phase 1-5 encoders, geodesic spirals, Hamilton space), then split into **math + inference engine** (C inference server in commit `816aea8`). WuBuMath is the reconciliation — all math comes home.

## All 64 Tests Pass

```bash
make test
```

```
=== WuBuMath Tests ===
[Color Manifold]     6/6 PASS
[Circular L1 Loss]   3/3 PASS
[PRNG]               2/2 PASS
[Q-Controller]       3/3 PASS
[Positional Encode]  1/1 PASS
[Loss Computation]   2/2 PASS
[Hamilton Encoder]   1/1 PASS
[VHF Decoder]        1/1 PASS
[VHF Audio Pipeline] 6/6 PASS
=== Results: Passed: 29/29 ===

=== Slermed JAX Tests ===
[compute core]      35/35 PASS  
=== Results: Passed: 35/35 ===

TOTAL: 64/64 PASS
```

## Build

```bash
make          # Build all
make test     # Run 64 tests
make clean    # Clean
```

Only needs: GCC with C11. Runs on Linux/macOS/WSL.

## Project Structure

```
WuBuMath/
├── include/
│   ├── wubumath.h              # Master header (color, model, VHF, Q-controller)
│   ├── jax.h                   # Slermed JAX master header
│   ├── jax_arena.h             # Arena allocator + SoA tensors
│   ├── jax_simd.h             # AVX2/NEON/Scalar SIMD
│   └── jax_nn.h               # MLP, activation, optimizer
├── src/
│   ├── math/
│   │   ├── wubu_color.c        # RGB/HSL + circular L1 loss
│   │   ├── wubu_positional_encode.c
│   │   └── wubu_utils.c        # Bilinear sample, box blur, audio strip
│   ├── model/
│   │   ├── wubu_hamilton_encoder.c  # Quaternion latent space
│   │   ├── wubu_vhf_decoder.c       # Coordinate-sampled decode
│   │   └── wubu_vhf_audio.c         # Full VHF pipeline
│   ├── train/
│   │   ├── wubu_q_controller.c      # Adaptive LR controller
│   │   └── wubu_loss.c              # HSL loss manifold
│   ├── jax/                         # Slermed JAX core (35 tests)
│   │   ├── jax_arena.c
│   │   ├── jax_simd.c
│   │   ├── jax_nn.c
│   │   ├── jax_opt.c
│   │   ├── jax_lax.c
│   │   └── jax_ir.c
│   └── tests/
│       ├── wubu_tests.c            # 29 tests
│       └── jax_slermed_test.c     # 35 tests
├── examples/
│   └── media_creator.c         # Generates frames + VHF tones → WAV
└── Makefile
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              WuBuMath v0.1.0                │
├─────────────────────────────────────────────┤
│                                             │
│  RGB Images [-1,1]                          │
│       │                                     │
│       ▼                                     │
│  ┌─────────────────────┐                    │
│  │ Hamilton Encoder     │──► Quaternions    │
│  │ (quaternion latent)  │    Amplitude      │
│  │                      │    Context [B,3]  │
│  └─────────────────────┘                    │
│       │                                     │
│       ▼                                     │
│  ┌─────────────────────┐                    │
│  │ VHF Decoder          │──► coords → RGB  │
│  │ (coordinate-sample)  │    [-1,1]         │
│  └─────────────────────┘                    │
│       │                                     │
│       ▼                                     │
│  ┌─────────────────────┐                    │
│  │ Loss Manifold        │──► HSL → L1/circ │
│  │ 10*L + 2*H + 1*S   │    composite loss  │
│  └─────────────────────┘                    │
│       │                                     │
│       ▼                                     │
│  ┌─────────────────────┐                    │
│  │ Q-Controller         │──► Adaptive LR   │
│  │ warmup → ε-greedy   │    exploration    │
│  └─────────────────────┘                    │
│                                             │
│  ┌─────────────────────┐                    │
│  │ Slermed JAX          │──► 35 ops        │
│  │ arena, GEMM, MLP,   │    JAX-compatible │
│  │ lax, IR, optimizer  │                    │
│  └─────────────────────┘                    │
│                                             │
│  ┌─────────────────────┐                    │
│  │ Media Creator        │──► WAV audio     │
│  │ frames + VHF tones   │    RAW frames    │
│  └─────────────────────┘                    │
└─────────────────────────────────────────────┘
```

## Part of the WuBu Family

| Repo | Purpose |
|------|---------|
| [WuBuOS](https://github.com/waefrebeorn/WuBuOS) | Custom OS kernel |
| [bytropix](https://github.com/waefrebeorn/bytropix) | Inference engine (C server, GPU kernels) |
| **WuBuMath** | **All math, encoders, media generation** ← you are here |

## License

MIT License — WuBu Slermes Project
