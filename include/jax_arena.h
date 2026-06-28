/*
 * jax_arena.h  --  PufferC/JaxRL Arena Allocator + SoA Tensor Infrastructure
 *
 * JAX-slermed C11 stack  --  zero Python, zero PyTorch, zero external deps.
 * Performance: Arena allocation, SoA tensors, SIMD-ready.
 *
 * Design:
 *   - Arena per epoch/rollout (reset, don't free)
 *   - Global arena for weights/grads (persistent)
 *   - Tensors as Structure of Arrays (SoA) for SIMD vectorization
 *   - All shapes fixed at init  --  no runtime alloc after setup
 */

#ifndef JAX_ARENA_H
#define JAX_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ===================================================================
 * Configuration Constants
 * =================================================================== */

#define JAX_MAX_TENSOR_DIMS    4
#define JAX_MAX_TENSOR_NAME    32
#define JAX_ARENA_ALIGN        64  /* cache-line + AVX-512 friendly */
#define JAX_DEFAULT_ARENA_CAP  (64 * 1024 * 1024)  /* 64 MiB default */

/* ===================================================================
 * Arena Allocator  --  Linear bump pointer, reset per epoch
 * =================================================================== */

typedef struct {
    uint8_t* base;      /* start of allocation */
    uint8_t* ptr;       /* current bump pointer */
    size_t   cap;       /* total capacity in bytes */
    size_t   used;      /* bytes allocated so far */
    size_t   peak_used; /* high-water mark for debugging */
} JaxArena;

/* Initialize arena from pre-allocated memory (mmap, malloc, static) */
static inline void jax_arena_init(JaxArena* a, void* backing, size_t cap) {
    a->base = (uint8_t*)backing;
    a->ptr  = (uint8_t*)backing;
    a->cap  = cap;
    a->used = 0;
    a->peak_used = 0;
}

/* Create arena with malloc (for host) */
static inline int jax_arena_create(JaxArena* a, size_t cap) {
    void* mem = aligned_alloc(JAX_ARENA_ALIGN, cap);
    if (!mem) return -1;
    jax_arena_init(a, mem, cap);
    return 0;
}

/* Destroy arena (free backing if malloc'd) */
static inline void jax_arena_destroy(JaxArena* a) {
    if (a->base) {
        free(a->base);
        a->base = a->ptr = NULL;
        a->cap = a->used = 0;
    }
}

/* Reset bump pointer  --  fast epoch/rollout boundary */
static inline void jax_arena_reset(JaxArena* a) {
    if (a->used > a->peak_used) a->peak_used = a->used;
    a->ptr = a->base;
    a->used = 0;
}

/* Aligned allocation from arena  --  returns NULL on OOM */
static inline void* jax_arena_alloc(JaxArena* a, size_t size, size_t align) {
    uintptr_t p = (uintptr_t)a->ptr;
    uintptr_t aligned = (p + align - 1) & ~(uintptr_t)(align - 1);
    size_t padding = aligned - p;
    if (a->used + padding + size > a->cap) return NULL;
    a->ptr += padding;
    void* out = a->ptr;
    a->ptr += size;
    a->used += padding + size;
    return out;
}

/* Convenience: typed allocation */
#define JAX_ARENA_ALLOC(a, type, count) \
    (type*)jax_arena_alloc(a, (count) * sizeof(type), alignof(type))

/* Scratch arena for temporary allocations within a function scope */
typedef struct {
    JaxArena* arena;
    size_t     mark;
} JaxScratch;

static inline JaxScratch jax_scratch_begin(JaxArena* a) {
    return (JaxScratch){ .arena = a, .mark = a->used };
}

static inline void jax_scratch_end(JaxScratch s) {
    s.arena->ptr = s.arena->base + s.mark;
    s.arena->used = s.mark;
}

/* ===================================================================
 * SoA Tensor  --  Structure of Arrays for SIMD
 * =================================================================== */

typedef enum {
    JAX_DTYPE_F32 = 0,
    JAX_DTYPE_I32 = 1,
    JAX_DTYPE_U8  = 2,
    JAX_DTYPE_I64 = 3,
} JaxDType;

static inline size_t jax_dtype_size(JaxDType dt) {
    switch (dt) {
        case JAX_DTYPE_F32: return sizeof(float);
        case JAX_DTYPE_I32: return sizeof(int32_t);
        case JAX_DTYPE_U8:  return sizeof(uint8_t);
        case JAX_DTYPE_I64: return sizeof(int64_t);
    }
    return 0;
}

/* Tensor descriptor  --  data allocated from arena, strides computed at init */
typedef struct {
    void*        data;           /* base pointer (from arena) */
    int64_t      shape[JAX_MAX_TENSOR_DIMS];
    int64_t      stride[JAX_MAX_TENSOR_DIMS];
    int          ndim;
    JaxDType    dtype;
    char         name[JAX_MAX_TENSOR_NAME];
} JaxTensor;

/* Compute strides (row-major / C-contiguous) */
static inline void jax_tensor_compute_strides(JaxTensor* t) {
    t->stride[t->ndim - 1] = jax_dtype_size(t->dtype);
    for (int i = t->ndim - 2; i >= 0; --i) {
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    }
}

/* Create tensor  --  allocates data from arena */
static inline int jax_tensor_create(JaxArena* a, JaxTensor* t,
                                      const int64_t* shape, int ndim,
                                      JaxDType dtype, const char* name) {
    if (ndim > JAX_MAX_TENSOR_DIMS) return -1;
    
    t->ndim = ndim;
    t->dtype = dtype;
    size_t elems = 1;
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
        elems *= shape[i];
    }
    
    size_t byte_size = elems * jax_dtype_size(dtype);
    t->data = jax_arena_alloc(a, byte_size, JAX_ARENA_ALIGN);
    if (!t->data) return -1;
    
    jax_tensor_compute_strides(t);
    
    if (name) {
        strncpy(t->name, name, JAX_MAX_TENSOR_NAME - 1);
        t->name[JAX_MAX_TENSOR_NAME - 1] = '\0';
    } else {
        t->name[0] = '\0';
    }
    
    return 0;
}

/* Create tensor from existing memory (no alloc) */
static inline void jax_tensor_wrap(JaxTensor* t, void* data,
                                     const int64_t* shape, int ndim,
                                     JaxDType dtype, const char* name) {
    t->data = data;
    t->ndim = ndim;
    t->dtype = dtype;
    for (int i = 0; i < ndim; ++i) t->shape[i] = shape[i];
    jax_tensor_compute_strides(t);
    if (name) strncpy(t->name, name, JAX_MAX_TENSOR_NAME - 1);
    else t->name[0] = '\0';
}

/* Tensor element access (bounds-check in debug) */
#define JAX_TENSOR_AT(t, dtype, ...) \
    *(dtype*)jax_tensor_ptr(t, (int64_t[]){__VA_ARGS__})

static inline void* jax_tensor_ptr(const JaxTensor* t, const int64_t* idx) {
    uintptr_t offset = 0;
    for (int i = 0; i < t->ndim; ++i) {
        offset += (uintptr_t)idx[i] * (uintptr_t)t->stride[i];
    }
    return (uint8_t*)t->data + offset;
}

/* Total elements */
static inline int64_t jax_tensor_numel(const JaxTensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

/* Bytes per element */
static inline size_t jax_tensor_elem_bytes(const JaxTensor* t) {
    return jax_dtype_size(t->dtype);
}

/* ===================================================================
 * Common RL Tensor Shapes (convenience factories)
 * =================================================================== */

/* Observation batch: [num_envs, obs_dim] */
static inline int jax_tensor_obs_batch(JaxArena* a, JaxTensor* t,
                                         int num_envs, int obs_dim, const char* name) {
    int64_t shape[2] = { num_envs, obs_dim };
    return jax_tensor_create(a, t, shape, 2, JAX_DTYPE_F32, name);
}

/* Action batch: [num_envs, act_dim] */
static inline int jax_tensor_act_batch(JaxArena* a, JaxTensor* t,
                                         int num_envs, int act_dim, const char* name) {
    int64_t shape[2] = { num_envs, act_dim };
    return jax_tensor_create(a, t, shape, 2, JAX_DTYPE_F32, name);
}

/* Scalar per env: [num_envs] */
static inline int jax_tensor_scalar_batch(JaxArena* a, JaxTensor* t,
                                            int num_envs, JaxDType dtype, const char* name) {
    int64_t shape[1] = { num_envs };
    return jax_tensor_create(a, t, shape, 1, dtype, name);
}

/* 1D vector */
static inline int jax_tensor_vec(JaxArena* a, JaxTensor* t,
                                   int len, JaxDType dtype, const char* name) {
    int64_t shape[1] = { len };
    return jax_tensor_create(a, t, shape, 1, dtype, name);
}

/* 2D weight matrix: [out_features, in_features] */
static inline int jax_tensor_weight(JaxArena* a, JaxTensor* t,
                                      int out_f, int in_f, const char* name) {
    int64_t shape[2] = { out_f, in_f };
    return jax_tensor_create(a, t, shape, 2, JAX_DTYPE_F32, name);
}

/* ===================================================================
 * Tensor Operations (in-place, SIMD-friendly where possible)
 * =================================================================== */

/* Fill tensor with value */
static inline void jax_tensor_fill(JaxTensor* t, float val) {
    if (t->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(t);
    float* p = (float*)t->data;
    for (int64_t i = 0; i < n; ++i) p[i] = val;
}

/* Copy tensor (must have same shape/dtype) */
static inline void jax_tensor_copy(JaxTensor* dst, const JaxTensor* src) {
    if (dst->dtype != src->dtype || dst->ndim != src->ndim) return;
    for (int i = 0; i < dst->ndim; ++i) if (dst->shape[i] != src->shape[i]) return;
    size_t bytes = jax_tensor_numel(src) * jax_dtype_size(src->dtype);
    memcpy(dst->data, src->data, bytes);
}

/* Scale tensor: dst = src * scale */
static inline void jax_tensor_scale(JaxTensor* dst, const JaxTensor* src, float scale) {
    if (dst->dtype != JAX_DTYPE_F32 || src->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(src);
    float* d = (float*)dst->data;
    const float* s = (const float*)src->data;
    for (int64_t i = 0; i < n; ++i) d[i] = s[i] * scale;
}

/* Add: dst += src */
static inline void jax_tensor_add(JaxTensor* dst, const JaxTensor* src) {
    if (dst->dtype != JAX_DTYPE_F32 || src->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(src);
    float* d = (float*)dst->data;
    const float* s = (const float*)src->data;
    for (int64_t i = 0; i < n; ++i) d[i] += s[i];
}

/* Element-wise multiply: dst *= src */
static inline void jax_tensor_mul(JaxTensor* dst, const JaxTensor* src) {
    if (dst->dtype != JAX_DTYPE_F32 || src->dtype != JAX_DTYPE_F32) return;
    int64_t n = jax_tensor_numel(src);
    float* d = (float*)dst->data;
    const float* s = (const float*)src->data;
    for (int64_t i = 0; i < n; ++i) d[i] *= s[i];
}

/* ===================================================================
 * Weight / Gradient Pair (for optimizer)
 * =================================================================== */

typedef struct {
    JaxTensor weight;
    JaxTensor bias;     /* Bias vector */
    JaxTensor grad;
    JaxTensor mom;      /* Adam: 1st moment */
    JaxTensor var;      /* Adam: 2nd moment */
    JaxTensor vel;      /* Muon: velocity */
    int        step;     /* Adam: step counter */
} JaxParam;

/* Create parameter with all optimizer state */
static inline int jax_param_create(JaxArena* a, JaxParam* p,
                                     int out_f, int in_f, const char* name) {
    char wname[64], bname[64], gname[64], mname[64], vname[64], v2name[64];
    snprintf(wname, sizeof(wname), "%s.w", name);
    snprintf(bname, sizeof(bname), "%s.b", name);
    snprintf(gname, sizeof(gname), "%s.g", name);
    snprintf(mname, sizeof(mname), "%s.m", name);
    snprintf(vname, sizeof(vname), "%s.v", name);
    snprintf(v2name, sizeof(v2name), "%s.v2", name);
    
    if (jax_tensor_weight(a, &p->weight, out_f, in_f, wname) != 0) return -1;
    if (jax_tensor_weight(a, &p->bias,   1, out_f, bname) != 0) return -1;
    if (jax_tensor_weight(a, &p->grad,   out_f, in_f, gname) != 0) return -1;
    if (jax_tensor_weight(a, &p->mom,    out_f, in_f, mname) != 0) return -1;  /* Adam */
    if (jax_tensor_weight(a, &p->var,    out_f, in_f, vname) != 0) return -1;  /* Adam */
    if (jax_tensor_weight(a, &p->vel,    out_f, in_f, v2name) != 0) return -1; /* Muon */
    /* Zero-initialize optimizer state (arena memory is uninitialized) */
    jax_tensor_fill(&p->grad, 0.0f);
    jax_tensor_fill(&p->mom,  0.0f);
    jax_tensor_fill(&p->var,  0.0f);
    jax_tensor_fill(&p->vel,  0.0f);
    p->step = 0;
    /* Zero-init bias */
    jax_tensor_fill(&p->bias, 0.0f);
    return 0;
}

/* Zero gradient */
static inline void jax_param_zero_grad(JaxParam* p) {
    jax_tensor_fill(&p->grad, 0.0f);
}

/* ===================================================================
 * Global Arenas (for easy access)
 * =================================================================== */

extern JaxArena g_jax_global_arena;   /* weights, optimizer state */
extern JaxArena g_jax_rollout_arena;  /* trajectory data, reset per epoch */

#endif /* JAX_ARENA_H */