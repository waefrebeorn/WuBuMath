/*
 * jax.h -- JAX-slermed: Pure C11 JAX implementation
 *
 * Slermed from WuBuOS/PufferC/BearRL foundations.
 * Zero Python, zero PyTorch, zero external deps.
 *
 * Features:
 *   - Arena allocation (epoch/rollout boundaries)
 *   - SoA tensors with strides (SIMD-ready)
 *   - AVX2/NEON/Scalar GEMM kernels
 *   - Fused MLP layers + activations (ReLU, Tanh, GELU, SiLU)
 *   - MinGRU recurrent cells
 *   - Analytical backprop (policy/value gradients)
 *   - Adam/Muon optimizer state
 */

#ifndef JAX_H
#define JAX_H

#include "jax_arena.h"
#include "jax_simd.h"
#include "jax_nn.h"

/* Version */
#define JAX_VERSION_MAJOR 0
#define JAX_VERSION_MINOR 1
#define JAX_VERSION_PATCH 0

/* Convenience: dtype enum aliases */
typedef JaxDType JaxDtype;
#define JAX_F32 JAX_DTYPE_F32
#define JAX_I32 JAX_DTYPE_I32
#define JAX_U8  JAX_DTYPE_U8
#define JAX_I64 JAX_DTYPE_I64

/* Convenience: tensor alias */
typedef JaxTensor JaxArray;

/* Convenience: arena alias */
typedef JaxArena JaxContext;

/* Key JAX concepts mapped to our C11 impl:
 *   jax.Array        -> JaxTensor (SoA with strides)
 *   jax.grad         -> jax_policy_backward / jax_value_backward
 *   jax.jit          -> AOT compiled kernels (our SIMD micro-kernels)
 *   jax.vmap         -> Batched tensor ops (first dim = batch)
 *   jax.pmap         -> Multi-device (TODO: MPI/NCCL)
 *   jax.lax          -> Low-level ops (gemm, act_batch, etc.)
 *   jax.nn           -> JaxPolicyNet / JaxValueNet
 *   jax.random       -> RNG state (uint64_t[2] splitmix64)
 *   jax.tree         -> Arena-managed param trees
 *   jax.checkpoint   -> jax_checkpoint_save/load
 */

#endif /* JAX_H */
