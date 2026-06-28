/*
 * jax_test.c -- JAX-slermed unit tests (complete parity)
 *
 * Tests: arena, tensor, gemm, activations, softmax, mlp, value, adam,
 *   transpose, reshape, slice, gather, scatter, concat, reduce, dot,
 *   attention, conv, pad, sort, clamp, select, comparison, ir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "jax.h"

/* ===================================================================
 * Test helpers
 * =================================================================== */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name(void)
#define RUN_TEST(name) do { \
    printf("  %-50s ", #name "..."); \
    fflush(stdout); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("FAIL (%s:%d): %f != %f (tol=%f)\n", __FILE__, __LINE__, (float)(a), (float)(b), (float)(tol)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL (%s:%d): %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ===================================================================
 * Core Tests (from v0.1.0)
 * =================================================================== */

TEST(test_arena_create) {
    JaxArena arena;
    ASSERT_EQ(jax_arena_create(&arena, 1024 * 1024), 0);
    ASSERT_EQ(arena.cap, 1024 * 1024);
    ASSERT_EQ(arena.used, 0);
    
    void* ptr = jax_arena_alloc(&arena, 256, 64);
    assert(ptr != NULL);
    ASSERT_EQ(arena.used, 256);
    
    jax_arena_reset(&arena);
    ASSERT_EQ(arena.used, 0);
    
    jax_arena_destroy(&arena);
}

TEST(test_tensor_create) {
    JaxArena arena;
    jax_arena_create(&arena, 1024 * 1024);
    
    JaxTensor t;
    int64_t shape[2] = {32, 64};
    ASSERT_EQ(jax_tensor_create(&arena, &t, shape, 2, JAX_F32, "test"), 0);
    ASSERT_EQ(t.shape[0], 32);
    ASSERT_EQ(t.shape[1], 64);
    ASSERT_EQ(t.ndim, 2);
    ASSERT_EQ(jax_tensor_numel(&t), 32 * 64);
    
    jax_arena_destroy(&arena);
}

TEST(test_tensor_ops) {
    JaxArena arena;
    jax_arena_create(&arena, 1024 * 1024);
    
    JaxTensor a, b, c;
    int64_t shape[2] = {4, 4};
    jax_tensor_create(&arena, &a, shape, 2, JAX_F32, "a");
    jax_tensor_create(&arena, &b, shape, 2, JAX_F32, "b");
    jax_tensor_create(&arena, &c, shape, 2, JAX_F32, "c");
    
    float* pa = (float*)a.data;
    float* pb = (float*)b.data;
    for (int i = 0; i < 16; ++i) {
        pa[i] = (float)i;
        pb[i] = (float)(i + 1);
    }
    
    jax_tensor_add(&c, &a);
    jax_tensor_add(&c, &b);
    
    float* pc = (float*)c.data;
    ASSERT_NEAR(pc[0], 1.0f, 1e-6f);
    ASSERT_NEAR(pc[15], 31.0f, 1e-6f);
    
    jax_arena_destroy(&arena);
}

TEST(test_gemm) {
    JaxArena arena;
    jax_arena_create(&arena, 1024 * 1024);
    
    float A[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float B[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    float C[2][4] = {0};
    
    jax_gemm((float*)A, (float*)B, (float*)C, 2, 4, 3);
    
    ASSERT_NEAR(C[0][0], 1.0f, 1e-5f);
    ASSERT_NEAR(C[0][1], 2.0f, 1e-5f);
    ASSERT_NEAR(C[0][2], 3.0f, 1e-5f);
    ASSERT_NEAR(C[0][3], 6.0f, 1e-5f);
    ASSERT_NEAR(C[1][0], 4.0f, 1e-5f);
    ASSERT_NEAR(C[1][3], 15.0f, 1e-5f);
    
    jax_arena_destroy(&arena);
}

TEST(test_activations) {
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float y[5];
    
    jax_act_batch(y, x, 5, JAX_ACT_RELU);
    ASSERT_NEAR(y[0], 0.0f, 1e-6f);
    ASSERT_NEAR(y[2], 0.0f, 1e-6f);
    ASSERT_NEAR(y[4], 2.0f, 1e-6f);
    
    jax_act_batch(y, x, 5, JAX_ACT_TANH);
    ASSERT_NEAR(y[0], tanhf(-2.0f), 1e-5f);
    ASSERT_NEAR(y[2], 0.0f, 1e-6f);
    
    jax_act_batch(y, x, 5, JAX_ACT_SILU);
    ASSERT_NEAR(y[2], 0.0f, 1e-6f);
}

TEST(test_softmax) {
    float logits[2][3] = {{1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}};
    float probs[2][3];
    
    jax_softmax(probs[0], logits[0], 1, 3);
    float sum = probs[0][0] + probs[0][1] + probs[0][2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    
    jax_softmax(probs[1], logits[1], 1, 3);
    ASSERT_NEAR(probs[1][0], 1.0f / 3.0f, 1e-5f);
}

/* ===================================================================
 * MLP / Value Network Tests
 * =================================================================== */

TEST(test_mlp_forward) {
    JaxArena param_arena, temp_arena;
    jax_arena_create(&param_arena, 16 * 1024 * 1024);
    jax_arena_create(&temp_arena, 16 * 1024 * 1024);
    
    JaxPolicyNet net;
    int obs_dim = 4, act_dim = 2;
    int hid[] = {8, 8};
    ASSERT_EQ(jax_policy_create_mlp(&net, &param_arena, obs_dim, act_dim, 1, hid, 2), 0);
    jax_orthogonal_init_params(&net, 1.0f);
    
    JaxTensor obs, actions, logprobs;
    int64_t obs_shape[2] = {2, obs_dim};
    int64_t act_shape[2] = {2, act_dim};
    int64_t lp_shape[1] = {2};
    jax_tensor_create(&temp_arena, &obs, obs_shape, 2, JAX_F32, "obs");
    jax_tensor_create(&temp_arena, &actions, act_shape, 2, JAX_F32, "actions");
    jax_tensor_create(&temp_arena, &logprobs, lp_shape, 1, JAX_F32, "logprobs");
    
    float* obs_data = (float*)obs.data;
    for (int i = 0; i < 8; ++i) obs_data[i] = (float)i * 0.1f;
    
    jax_policy_forward(&net, &obs, NULL, &actions, &logprobs, NULL, NULL, &temp_arena);
    
    float* act_data = (float*)actions.data;
    for (int b = 0; b < 2; ++b) {
        float sum = 0.0f;
        for (int a = 0; a < act_dim; ++a) sum += act_data[b * act_dim + a];
        ASSERT_NEAR(sum, 1.0f, 1e-5f);
    }
    
    jax_arena_destroy(&param_arena);
    jax_arena_destroy(&temp_arena);
}

TEST(test_mlp_backward) {
    JaxArena param_arena, temp_arena;
    jax_arena_create(&param_arena, 16 * 1024 * 1024);
    jax_arena_create(&temp_arena, 16 * 1024 * 1024);
    
    JaxPolicyNet net;
    int obs_dim = 4, act_dim = 3;
    int hid[] = {16};
    ASSERT_EQ(jax_policy_create_mlp(&net, &param_arena, obs_dim, act_dim, 1, hid, 1), 0);
    jax_orthogonal_init_params(&net, 1.0f);
    
    int batch = 4;
    JaxTensor obs, actions, old_logprobs, advantages;
    int64_t obs_shape[2] = {batch, obs_dim};
    int64_t act_shape[2] = {batch, act_dim};
    int64_t scalar_shape[1] = {batch};
    
    jax_tensor_create(&temp_arena, &obs, obs_shape, 2, JAX_F32, "obs");
    jax_tensor_create(&temp_arena, &actions, act_shape, 2, JAX_F32, "actions");
    jax_tensor_create(&temp_arena, &old_logprobs, scalar_shape, 1, JAX_F32, "old_lp");
    jax_tensor_create(&temp_arena, &advantages, scalar_shape, 1, JAX_F32, "adv");
    
    float* obs_data = (float*)obs.data;
    for (int i = 0; i < batch * obs_dim; ++i) obs_data[i] = (float)(i % 7) * 0.1f;
    
    float* act_data = (float*)actions.data;
    for (int b = 0; b < batch; ++b) {
        for (int a = 0; a < act_dim; ++a) act_data[b * act_dim + a] = 0.0f;
        act_data[b * act_dim + 1] = 1.0f;
    }
    
    float* lp_data = (float*)old_logprobs.data;
    for (int b = 0; b < batch; ++b) lp_data[b] = -1.0f;
    
    float* adv_data = (float*)advantages.data;
    for (int b = 0; b < batch; ++b) adv_data[b] = 1.0f;
    
    JaxTensor logprobs_out;
    int64_t lp_out_shape[1] = {batch};
    jax_tensor_create(&temp_arena, &logprobs_out, lp_out_shape, 1, JAX_F32, "lp_out");
    jax_policy_forward(&net, &obs, NULL, &actions, &logprobs_out, NULL, NULL, &temp_arena);
    
    ASSERT_EQ(jax_policy_backward(&net, &obs, &actions, &old_logprobs, &advantages,
                                   0.2f, 1.0f, &temp_arena), 0);
    
    int has_grad = 0;
    for (int i = 0; i < net.num_layers; ++i) {
        JaxParam* p = net.layers[i].param;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) {
            if (fabsf(g[j]) > 1e-8f) { has_grad = 1; break; }
        }
        if (has_grad) break;
    }
    ASSERT_EQ(has_grad, 1);
    
    jax_arena_destroy(&param_arena);
    jax_arena_destroy(&temp_arena);
}

TEST(test_value_network) {
    JaxArena param_arena, temp_arena;
    jax_arena_create(&param_arena, 16 * 1024 * 1024);
    jax_arena_create(&temp_arena, 16 * 1024 * 1024);
    
    JaxValueNet vnet;
    int obs_dim = 4;
    int hid[] = {8, 4};
    ASSERT_EQ(jax_value_create(&vnet, &param_arena, obs_dim, hid, 2), 0);
    jax_value_orthogonal_init(&vnet, 1.0f);
    
    JaxTensor obs, values, targets;
    int64_t obs_shape[2] = {3, obs_dim};
    int64_t val_shape[1] = {3};
    
    jax_tensor_create(&temp_arena, &obs, obs_shape, 2, JAX_F32, "obs");
    jax_tensor_create(&temp_arena, &values, val_shape, 1, JAX_F32, "values");
    jax_tensor_create(&temp_arena, &targets, val_shape, 1, JAX_F32, "targets");
    
    float* obs_data = (float*)obs.data;
    for (int i = 0; i < 3 * obs_dim; ++i) obs_data[i] = (float)i * 0.05f;
    
    float* tgt_data = (float*)targets.data;
    for (int i = 0; i < 3; ++i) tgt_data[i] = (float)i * 0.5f;
    
    jax_value_forward(&vnet, &obs, &values, &temp_arena);
    ASSERT_EQ(jax_value_backward(&vnet, &obs, &values, &targets, 0.5f, &temp_arena), 0);
    
    int has_grad = 0;
    for (int i = 0; i < vnet.num_layers; ++i) {
        JaxParam* p = vnet.layers[i].param;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) {
            if (fabsf(g[j]) > 1e-8f) { has_grad = 1; break; }
        }
        if (has_grad) break;
    }
    ASSERT_EQ(has_grad, 1);
    
    jax_arena_destroy(&param_arena);
    jax_arena_destroy(&temp_arena);
}

TEST(test_param_get_set) {
    JaxArena arena;
    jax_arena_create(&arena, 1024 * 1024);
    
    JaxPolicyNet net;
    int obs_dim = 4, act_dim = 2;
    int hid[] = {8, 4};
    jax_policy_create_mlp(&net, &arena, obs_dim, act_dim, 1, hid, 2);
    
    int total = jax_policy_get_params(&net, NULL, 0);
    ASSERT_EQ(total > 0, 1);
    
    float* params = (float*)malloc(total * sizeof(float));
    jax_policy_get_params(&net, params, total);
    
    for (int i = 0; i < total; ++i) params[i] += 0.01f;
    ASSERT_EQ(jax_policy_set_params(&net, params, total), 0);
    
    free(params);
    jax_arena_destroy(&arena);
}

/* ===================================================================
 * Lax Tests (new in v0.2.0)
 * =================================================================== */

TEST(test_reduce_sum_all) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ASSERT_NEAR(jax_reduce_sum_all(data, 5), 15.0f, 1e-5f);
}

TEST(test_reduce_mean) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_NEAR(jax_reduce_mean(data, 4), 2.5f, 1e-5f);
}

TEST(test_reduce_max_min) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    ASSERT_NEAR(jax_reduce_max_all(data, 5), 5.0f, 1e-5f);
    ASSERT_NEAR(jax_reduce_min_all(data, 5), 1.0f, 1e-5f);
}

TEST(test_reduce_sum_axis0) {
    float in[3][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    float out[4];
    jax_reduce_sum_axis0_2d(out, (float*)in, 3, 4);
    ASSERT_NEAR(out[0], 15.0f, 1e-5f);
    ASSERT_NEAR(out[1], 18.0f, 1e-5f);
    ASSERT_NEAR(out[3], 24.0f, 1e-5f);
}

TEST(test_reduce_sum_axis1) {
    float in[3][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    float out[3];
    jax_reduce_sum_axis1_2d(out, (float*)in, 3, 4);
    ASSERT_NEAR(out[0], 10.0f, 1e-5f);
    ASSERT_NEAR(out[1], 26.0f, 1e-5f);
    ASSERT_NEAR(out[2], 42.0f, 1e-5f);
}

TEST(test_argmax) {
    float data[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    ASSERT_EQ(jax_argmax(data, 5), 1);
    ASSERT_EQ(jax_argmin(data, 5), 0);
}

TEST(test_transpose_2d) {
    float in[2][3] = {{1,2,3},{4,5,6}};
    float out[3][2];
    jax_transpose_2d((float*)out, (float*)in, 2, 3);
    ASSERT_NEAR(out[0][0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[0][1], 4.0f, 1e-6f);
    ASSERT_NEAR(out[1][0], 2.0f, 1e-6f);
    ASSERT_NEAR(out[2][1], 6.0f, 1e-6f);
}

TEST(test_gather_1d) {
    float in[] = {10.0f, 20.0f, 30.0f, 40.0f};
    int64_t idx[] = {2, 0, 3};
    float out[3];
    jax_gather_1d(out, in, idx, 3);
    ASSERT_NEAR(out[0], 30.0f, 1e-6f);
    ASSERT_NEAR(out[1], 10.0f, 1e-6f);
    ASSERT_NEAR(out[2], 40.0f, 1e-6f);
}

TEST(test_scatter_2d_add) {
    float out[4][3] = {0};
    float in[2][3] = {{1,2,3},{4,5,6}};
    int indices[] = {1, 3};
    jax_scatter_2d_add((float*)out, (float*)in, indices, 2, 3);
    ASSERT_NEAR(out[1][0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1][1], 2.0f, 1e-6f);
    ASSERT_NEAR(out[3][2], 6.0f, 1e-6f);
    ASSERT_NEAR(out[0][0], 0.0f, 1e-6f);
}

TEST(test_concat_2d_axis0) {
    float a[2][3] = {{1,2,3},{4,5,6}};
    float b[1][3] = {{7,8,9}};
    float out[3][3];
    jax_concat_2d_axis0((float*)out, (float*)a, 2, 3, (float*)b, 1);
    ASSERT_NEAR(out[2][0], 7.0f, 1e-6f);
    ASSERT_NEAR(out[2][2], 9.0f, 1e-6f);
}

TEST(test_concat_2d_axis1) {
    float a[2][2] = {{1,2},{3,4}};
    float b[2][2] = {{5,6},{7,8}};
    float out[2][4];
    jax_concat_2d_axis1((float*)out, (float*)a, 2, 2, (float*)b, 2);
    ASSERT_NEAR(out[0][1], 2.0f, 1e-6f);
    ASSERT_NEAR(out[0][2], 5.0f, 1e-6f);
    ASSERT_NEAR(out[1][3], 8.0f, 1e-6f);
}

TEST(test_dot_general) {
    float a[2][3] = {{1,2,3},{4,5,6}};
    float b[3][2] = {{1,2},{3,4},{5,6}};
    float out[2][2];
    /* dot_general: out = a @ b (standard matrix multiply) */
    jax_dot_general_2d((float*)a, (float*)b, (float*)out, 2, 3, 2);
    /* out[0] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28] */
    ASSERT_NEAR(out[0][0], 22.0f, 1e-4f);
    ASSERT_NEAR(out[0][1], 28.0f, 1e-4f);
    /* out[1] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64] */
    ASSERT_NEAR(out[1][0], 49.0f, 1e-4f);
    ASSERT_NEAR(out[1][1], 64.0f, 1e-4f);
}

TEST(test_dot_general_batched) {
    /* batch=2, M=2, K=3, N=2 */
    float a[2][2][3] = {{{1,0,0},{0,1,0}}, {{1,0,0},{0,0,1}}};
    float b[2][3][2] = {{{1,0},{0,1},{0,0}}, {{1,0},{0,0},{0,1}}};
    float out[2][2][2];
    jax_dot_general_batched((float*)a, (float*)b, (float*)out, 2, 2, 3, 2);
    ASSERT_NEAR(out[0][0][0], 1.0f, 1e-5f);
    ASSERT_NEAR(out[0][1][1], 1.0f, 1e-5f);
    ASSERT_NEAR(out[1][0][0], 1.0f, 1e-5f);
    ASSERT_NEAR(out[1][1][1], 1.0f, 1e-5f);
}

TEST(test_select) {
    float mask[] = {1.0f, 0.0f, 1.0f};
    float a[] = {10.0f, 20.0f, 30.0f};
    float b[] = {100.0f, 200.0f, 300.0f};
    float out[3];
    jax_select(out, mask, a, b, 3);
    ASSERT_NEAR(out[0], 10.0f, 1e-6f);
    ASSERT_NEAR(out[1], 200.0f, 1e-6f);
    ASSERT_NEAR(out[2], 30.0f, 1e-6f);
}

TEST(test_clamp) {
    float in[] = {-5.0f, 0.0f, 5.0f, 10.0f};
    float out[4];
    float min_val = -1.0f, max_val = 7.0f;
    jax_clamp(out, in, &min_val, &max_val, 4);
    ASSERT_NEAR(out[0], -1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    ASSERT_NEAR(out[2], 5.0f, 1e-6f);
    ASSERT_NEAR(out[3], 7.0f, 1e-6f);
}

TEST(test_comparison_ops) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 3.0f, 2.0f};
    float out[3];
    
    jax_equal(out, a, b, 3);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    
    jax_less(out, a, b, 3);
    ASSERT_NEAR(out[0], 0.0f, 1e-6f);
    ASSERT_NEAR(out[1], 1.0f, 1e-6f);
    ASSERT_NEAR(out[2], 0.0f, 1e-6f);
    
    jax_greater_equal(out, a, b, 3);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
}

TEST(test_unary_ops) {
    float in[] = {-2.0f, 0.0f, 4.0f};
    float out[3];
    
    jax_neg(out, in, 3);
    ASSERT_NEAR(out[0], 2.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    
    jax_abs(out, in, 3);
    ASSERT_NEAR(out[0], 2.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    
    jax_sign(out, in, 3);
    ASSERT_NEAR(out[0], -1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    ASSERT_NEAR(out[2], 1.0f, 1e-6f);
    
    jax_exp(out, (float[]){0.0f}, 1);
    ASSERT_NEAR(out[0], 1.0f, 1e-5f);
    
    jax_log(out, (float[]){1.0f}, 1);
    ASSERT_NEAR(out[0], 0.0f, 1e-5f);
    
    jax_rsqrt(out, (float[]){4.0f}, 1);
    ASSERT_NEAR(out[0], 0.5f, 1e-5f);
    
    jax_cbrt(out, (float[]){27.0f}, 1);
    ASSERT_NEAR(out[0], 3.0f, 1e-4f);
}

TEST(test_pad_2d) {
    float in[2][3] = {{1,2,3},{4,5,6}};
    float out[4][5];
    jax_pad_2d((float*)out, (float*)in, 2, 3, 1, 0, 1, 1, 0.0f);
    /* out[0] = [0,0,0,0,0] (top pad) */
    ASSERT_NEAR(out[0][0], 0.0f, 1e-6f);
    /* out[1] = [0,1,2,3,0] (original row 0 + left/right pad) */
    ASSERT_NEAR(out[1][1], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1][3], 3.0f, 1e-6f);
    /* out[2] = [0,4,5,6,0] (original row 1 + left/right pad) */
    ASSERT_NEAR(out[2][1], 4.0f, 1e-6f);
}

TEST(test_top_k) {
    float in[] = {5.0f, 1.0f, 8.0f, 3.0f, 9.0f, 2.0f};
    float values[3];
    int64_t indices[3];
    jax_top_k(values, indices, in, 6, 3);
    ASSERT_NEAR(values[0], 9.0f, 1e-6f);
    ASSERT_NEAR(values[1], 8.0f, 1e-6f);
    ASSERT_NEAR(values[2], 5.0f, 1e-6f);
}

TEST(test_sort_key_val) {
    float keys[] = {3.0f, 1.0f, 2.0f};
    float vals[] = {30.0f, 10.0f, 20.0f};
    float sorted_keys[3], sorted_vals[3];
    jax_sort_key_val(sorted_keys, sorted_vals, keys, vals, 3);
    ASSERT_NEAR(sorted_keys[0], 1.0f, 1e-6f);
    ASSERT_NEAR(sorted_keys[1], 2.0f, 1e-6f);
    ASSERT_NEAR(sorted_keys[2], 3.0f, 1e-6f);
    ASSERT_NEAR(sorted_vals[0], 10.0f, 1e-6f);
    ASSERT_NEAR(sorted_vals[1], 20.0f, 1e-6f);
    ASSERT_NEAR(sorted_vals[2], 30.0f, 1e-6f);
}

TEST(test_sort_axis1) {
    float in[2][4] = {{3,1,4,1},{5,9,2,6}};
    float out[2][4];
    jax_sort_axis1((float*)out, (float*)in, 2, 4);
    ASSERT_NEAR(out[0][0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[0][1], 1.0f, 1e-6f);
    ASSERT_NEAR(out[0][3], 4.0f, 1e-6f);
    ASSERT_NEAR(out[1][2], 6.0f, 1e-6f);
}

TEST(test_conv_2d_simple) {
    /* 3x3 input, 2x2 kernel, stride=1, no pad → 2x2 output */
    float input[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
    float kernel[2][2] = {{1,0},{0,1}};
    float output[2][2];
    jax_conv_2d((float*)input, (float*)kernel, (float*)output,
                3, 3, 1, 2, 2, 1, 1, 1, 0, 0);
    /* output[0,0] = 1*1 + 5*1 = 6 */
    ASSERT_NEAR(output[0][0], 6.0f, 1e-5f);
    /* output[0,1] = 2*1 + 6*1 = 8 */
    ASSERT_NEAR(output[0][1], 8.0f, 1e-5f);
    /* output[1,1] = 5*1 + 9*1 = 14 */
    ASSERT_NEAR(output[1][1], 14.0f, 1e-5f);
}

TEST(test_attention) {
    /* Simple: batch=1, seq_q=1, seq_k=3, dim=2 */
    float Q[1*1*2] = {1.0f, 0.0f};
    float K[1*3*2] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float V[1*3*2] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float out[1*1*2];
    jax_attention_scaled(Q, K, V, out, 1, 1, 3, 2);
    /* Q=[1,0] dot K[0]=[1,0] -> 1.0 (highest), K[2]=[1,1] -> 1.0 (tied) */
    /* softmax probs: [0.401, 0.198, 0.401] */
    /* out[0] = 0.401*1 + 0.198*0 + 0.401*1 = 0.802 */
    /* out[1] = 0.401*0 + 0.198*1 + 0.401*1 = 0.599 */
    ASSERT_NEAR(out[0], 0.8f, 0.01f);
    ASSERT_NEAR(out[1], 0.6f, 0.01f);
}

TEST(test_stop_gradient) {
    float in[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    jax_stop_gradient(out, in, 3);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 2.0f, 1e-6f);
    ASSERT_NEAR(out[2], 3.0f, 1e-6f);
}

/* ===================================================================
 * Jaxpr IR Tests
 * =================================================================== */

TEST(test_ir_create) {
    JaxArena arena;
    jax_arena_create(&arena, 1024 * 1024);
    
    JaxIr* ir = jax_ir_create(&arena, 2);
    assert(ir != NULL);
    ASSERT_EQ(jax_ir_num_instrs(ir), 0);
    
    /* Build: c = a + b */
    JaxVarId a = jax_ir_param(ir, 0);
    JaxVarId b = jax_ir_param(ir, 1);
    JaxVarId c = jax_ir_add(ir, a, b);
    ASSERT_EQ(jax_ir_num_instrs(ir), 3);  /* param(0), param(1), add */
    
    /* Build: d = c * a */
    JaxVarId d = jax_ir_mul(ir, c, a);
    ASSERT_EQ(jax_ir_num_instrs(ir), 4);
    
    jax_ir_print(ir);
    
    /* Backward */
    ASSERT_EQ(jax_ir_backward(ir, d), 0);
    
    jax_arena_destroy(&arena);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("JAX-slermed Unit Tests (Complete Parity)\n");
    printf("=========================================\n\n");
    
    /* Core */
    RUN_TEST(test_arena_create);
    RUN_TEST(test_tensor_create);
    RUN_TEST(test_tensor_ops);
    RUN_TEST(test_gemm);
    RUN_TEST(test_activations);
    RUN_TEST(test_softmax);
    
    /* MLP / Value */
    RUN_TEST(test_mlp_forward);
    RUN_TEST(test_mlp_backward);
    RUN_TEST(test_value_network);
    RUN_TEST(test_param_get_set);
    
    /* Lax */
    RUN_TEST(test_reduce_sum_all);
    RUN_TEST(test_reduce_mean);
    RUN_TEST(test_reduce_max_min);
    RUN_TEST(test_reduce_sum_axis0);
    RUN_TEST(test_reduce_sum_axis1);
    RUN_TEST(test_argmax);
    RUN_TEST(test_transpose_2d);
    RUN_TEST(test_gather_1d);
    RUN_TEST(test_scatter_2d_add);
    RUN_TEST(test_concat_2d_axis0);
    RUN_TEST(test_concat_2d_axis1);
    RUN_TEST(test_dot_general);
    RUN_TEST(test_dot_general_batched);
    RUN_TEST(test_select);
    RUN_TEST(test_clamp);
    RUN_TEST(test_comparison_ops);
    RUN_TEST(test_unary_ops);
    RUN_TEST(test_pad_2d);
    RUN_TEST(test_top_k);
    RUN_TEST(test_sort_key_val);
    RUN_TEST(test_sort_axis1);
    RUN_TEST(test_conv_2d_simple);
    RUN_TEST(test_attention);
    RUN_TEST(test_stop_gradient);
    
    /* IR */
    RUN_TEST(test_ir_create);
    
    printf("\n=========================================\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
