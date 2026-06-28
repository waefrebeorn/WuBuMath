/*
 * jax_arena.c -- Non-inline arena/tensor implementations
 */

#include "jax_arena.h"

JaxArena g_jax_global_arena;
JaxArena g_jax_rollout_arena;

/* Global arena initialization */
int jax_global_init(size_t global_cap, size_t rollout_cap) {
    if (jax_arena_create(&g_jax_global_arena, global_cap) != 0) return -1;
    if (jax_arena_create(&g_jax_rollout_arena, rollout_cap) != 0) {
        jax_arena_destroy(&g_jax_global_arena);
        return -1;
    }
    return 0;
}

void jax_global_shutdown(void) {
    jax_arena_destroy(&g_jax_global_arena);
    jax_arena_destroy(&g_jax_rollout_arena);
}

/* Tensor operations that benefit from being non-inline */
void jax_tensor_zero(JaxTensor* t) {
    if (t->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(t);
    float* p = (float*)t->data;
    for (int64_t i = 0; i < n; ++i) p[i] = 0.0f;
}

void jax_tensor_randn(JaxTensor* t, uint64_t rng_state[2]) {
    if (t->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(t);
    float* p = (float*)t->data;
    
    /* Box-Muller transform using splitmix64 */
    uint64_t s0 = rng_state[0];
    uint64_t s1 = rng_state[1];
    
    for (int64_t i = 0; i < n; i += 2) {
        /* splitmix64 */
        s0 += 0x9e3779b97f4a7c15;
        uint64_t z = s0;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = z ^ (z >> 31);
        float u1 = (float)(z >> 1) / (float)(UINT64_MAX >> 1);
        
        s1 += 0x9e3779b97f4a7c15;
        z = s1;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = z ^ (z >> 31);
        float u2 = (float)(z >> 1) / (float)(UINT64_MAX >> 1);
        
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265359f * u2;
        
        p[i] = r * cosf(theta);
        if (i + 1 < n) p[i + 1] = r * sinf(theta);
    }
    
    rng_state[0] = s0;
    rng_state[1] = s1;
}

/* SplitMix64 RNG */
uint64_t jax_rng_next(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

float jax_rng_uniform(uint64_t* state) {
    return (float)(jax_rng_next(state) >> 1) / (float)(UINT64_MAX >> 1);
}

float jax_rng_normal(uint64_t* state) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    float u1 = jax_rng_uniform(state);
    float u2 = jax_rng_uniform(state);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265359f * u2;
    
    spare = r * sinf(theta);
    has_spare = 1;
    return r * cosf(theta);
}
