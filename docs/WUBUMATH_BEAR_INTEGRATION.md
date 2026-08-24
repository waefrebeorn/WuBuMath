# WUBUMATH_BEAR_INTEGRATION.md — orientation: who does what (2026-08-24)

> The division of labor between the two repos, stated once so no agent
> confuses them again.

## The two repos

| | **WuBuMath** | **BearRL** |
|---|---|---|
| Role | The MATH library | the RL ENVIRONMENT + proof engine |
| Owns | manifolds (quaternion/SO3/Lorentz/Poincaré/nested-hyperbolic), Hamilton encoder, VHF canvas/codec pieces, flow-matching trainers, slermed JAX core | PPO/GAE/MinGRU trainer, envs, curriculum, Vulkan/CUDA backends, reward-as-proof certificates |
| Produces | algorithms, codecs, transforms — *the things being proven* | training runs + committed logs — *the proofs* |
| Tests | `make test` (64 tests) — unit gate for the math | property harness + env-solve certificates |

**Rule:** math lives in WuBuMath; environments that train/prove the math live
in BearRL. BearRL may LINK WuBuMath (`libwubumath` / its headers); it never
re-implements manifold ops. WuBuMath never contains an RL trainer.

## Dependency direction

```
BearRL  ──links──▶  WuBuMath   (envs observe/manipulate WuBuMath structures)
WuBuMath  ──(no dep on BearRL)──
```

## Where the current idea mixture lives

- Full write-up (beam-sweep canvas, φ-fractal sampling, quaternion
  flow-matching P-frames, dual-band visible/infrared encoding, manifold CLIP,
  testing harness philosophy): **`docs/BEAR_RL_MANIFEST.md` in BearRL**.
- The math components referenced there (wubu_canvas, wubu_tangent_flow,
  wubu_parallel_transport, wubu_poincare_geom, …) are implemented HERE in
  WuBuMath under `src/`.

## Work split for the current campaign

### WuBuMath (math side)
1. [ ] Beam-sweep canvas: strip representation + φ-fractal sweep order
      (`wubu_canvas` extension — sampling-depth API, not new canvas sizes)
2. [ ] Dual-band frame format: visible + infrared sweep segments
3. [ ] Quaternion probability-flow ODE: wire `wubu_parallel_transport.c`
      into `wubu_tangent_flow.c` integration steps
4. [ ] **Manifold CLIP core**: geodesic-similarity InfoNCE loss +
      Riemannian SGD on curvature/scale (new: `src/train/wubu_manifold_clip.c`)
5. [ ] Unit gates for each of the above (test IS the spec)

### BearRL (environment side)
1. [ ] Link WuBuMath as dependency (pkg-config or relative submodule path)
2. [ ] Property harness skeleton (fuzz seeds, invariant registry)
3. [ ] Env: quaternion geodesic reacher (obs = latent pair, action = flow
      vector field params, reward = −geodesic distance)
4. [ ] Env: φ-sweep locality (reward = k-NN recall under sweep order)
5. [ ] Env: manifold-CLIP retrieval (reward = recall@k) → certificate
6. [ ] Audio-sideband fidelity env → certificate

## Cross-repo parity rule

Anything BearRL proves about a WuBuMath component must ALSO pass WuBuMath's
own test suite against the same commit. No divergent truths between repos.
