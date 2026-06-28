/*
 * jax_nn.h  --  PufferC/JaxRL Neural Network: PolicyNet MLP + MinGRU
 *
 * Pure C11: MLP forward, recurrent (MinGRU), action sampling.
 * SIMD-accelerated via jax_simd.h.
 */

#ifndef JAX_NN_H
#define JAX_NN_H

#include "jax_arena.h"
#include "jax_simd.h"

/* ===================================================================
 * Policy Network: Actor-Critic with optional recurrence
 * =================================================================== */

typedef enum {
    JAX_NET_MLP   = 0,  /* Feedforward only */
    JAX_NET_MINGU = 1,  /* MinGRU recurrent */
} JaxNetType;

/* Layer configuration */
typedef struct {
    int in_features;
    int out_features;
    JaxAct act;
    JaxParam* param;  /* weight + bias + optimizer state */

    /* Forward-pass stored activations (for backward) */
    /* z_pre: [batch, out_features]  pre-activation  x @ W^T + b   */
    /* a_post: [batch, out_features] post-activation act(z_pre)   */
    /* For layer 0 the "input" (a_prev) is the observation tensor */
    JaxTensor z_pre;
    JaxTensor a_post;
    int act_storage;
} JaxLayer;

/* Policy Network */
typedef struct {
    JaxNetType type;
    int num_layers;
    JaxLayer* layers;       /* array of num_layers */
    JaxMinGRU* gru;         /* optional recurrent core */
    int obs_dim;
    int act_dim;
    int act_discrete;
    int hid_size;
    JaxArena* param_arena;  /* where params live */
    int fwd_stored;          /* flag: forward pass stored activations */
    /* Gaussian policy for continuous actions */
    float* logstd;           /* [act_dim] learned log-std (NULL if fixed) */
    float   logstd_fixed;    /* fixed logstd value when logstd==NULL */
} JaxPolicyNet;

/* Create MLP policy network */
int jax_policy_create_mlp(JaxPolicyNet* net, JaxArena* param_arena,
                            int obs_dim, int act_dim, int act_discrete,
                            const int* hid_sizes, int num_hid);

/* Create MinGRU policy network */
int jax_policy_create_mingru(JaxPolicyNet* net, JaxArena* param_arena,
                               int obs_dim, int act_dim, int act_discrete,
                               int hid_size);

/* Forward pass: obs -> (actions, logprobs, values, next_hidden) */
/* obs: [batch, obs_dim] or [batch, seq_len, obs_dim] for recurrent */
void jax_policy_forward(const JaxPolicyNet* net,
                          const JaxTensor* obs,        /* [batch, obs_dim] */
                          const JaxTensor* h_in,       /* [batch, hid] (optional) */
                          JaxTensor* actions,          /* [batch, act_dim] */
                          JaxTensor* logprobs,         /* [batch] */
                          JaxTensor* values,           /* [batch] */
                          JaxTensor* h_out,            /* [batch, hid] (optional) */
                          JaxArena* temp_arena);

/* Sample action from policy output (in-place) */
void jax_policy_sample(JaxPolicyNet* net, JaxTensor* actions, JaxTensor* logprobs,
                         uint64_t rng_state[2]);

/* Get deterministic action (argmax for discrete, mean for continuous) */
void jax_policy_deterministic(JaxPolicyNet* net, JaxTensor* actions);

/* Get all parameters as flat array (for checkpointing) */
int jax_policy_get_params(const JaxPolicyNet* net, float* out, int max_params);
int jax_policy_set_params(JaxPolicyNet* net, const float* in, int num_params);

/* ===================================================================
 * Value Network (separate critic optionally)
 * =================================================================== */

typedef struct {
    int num_layers;
    JaxLayer* layers;
    JaxArena* param_arena;
    int fwd_stored;
} JaxValueNet;

int jax_value_create(JaxValueNet* vnet, JaxArena* param_arena,
                       int obs_dim, const int* hid_sizes, int num_hid);

void jax_value_forward(const JaxValueNet* vnet,
                         const JaxTensor* obs,  /* [batch, obs_dim] */
                         JaxTensor* values,     /* [batch] */
                         JaxArena* temp_arena);

/* ===================================================================
 * Backward Pass (analytical gradients)
 * =================================================================== */

/*
 * Policy backward: compute gradients for policy network.
 * Must be called AFTER jax_policy_forward (which stores activations).
 *
 * For discrete actions (categorical policy):
 *   dlogit[i] = (ratio_clipped[i] * adv[i]) / mb_size
 *   where ratio_clipped = clip(exp(new_lp - old_lp), 1-eps, 1+eps)
 *
 * For continuous actions (Gaussian policy):
 *   dmu[i] = (ratio_clipped[i] * adv[i]) / mb_size * (action[i] - mu[i]) / std^2
 *
 * obs:       [mb, obs_dim]  minibatch observations
 * actions:   [mb, act_dim]  one-hot chosen actions (discrete) or raw actions (continuous)
 * old_logprobs: [mb]
 * advantages:   [mb]  (already normalized)
 * clip_coef:   PPO clip epsilon
 * policy_grad_scale: additional scale factor (e.g. 1.0)
 *
 * Returns 0 on success.
 */
int jax_policy_backward(JaxPolicyNet* net,
                          const JaxTensor* obs,
                          const JaxTensor* actions,
                          const JaxTensor* old_logprobs,
                          const JaxTensor* advantages,
                          float clip_coef,
                          float policy_grad_scale,
                          JaxArena* temp_arena);

/* Internal: discrete backward */
int jax_policy_backward_discrete(JaxPolicyNet* net,
                                    const JaxTensor* obs,
                                    const JaxTensor* actions,
                                    const JaxTensor* old_logprobs,
                                    const JaxTensor* advantages,
                                    float clip_coef,
                                    float policy_grad_scale,
                                    JaxArena* temp_arena);

/* Internal: continuous (Gaussian) backward */
int jax_policy_backward_continuous(JaxPolicyNet* net,
                                     const JaxTensor* obs,
                                     const JaxTensor* actions,
                                     const JaxTensor* old_logprobs,
                                     const JaxTensor* advantages,
                                     float clip_coef,
                                     float policy_grad_scale,
                                     JaxArena* temp_arena);

/*
 * Value backward: compute gradients for value network.
 * Must be called AFTER jax_value_forward (which stores activations).
 *
 * Loss = 0.5 * (V - target)^2
 * dV = (V - target) / mb_size
 *
 * values:   [mb]  current value predictions
 * targets:  [mb]  returns (GAE + old_values)
 * vf_coef:  value loss coefficient
 */
int jax_value_backward(JaxValueNet* vnet,
                         const JaxTensor* obs,
                         const JaxTensor* values,
                         const JaxTensor* targets,
                         float vf_coef,
                         JaxArena* temp_arena);

/* Zero all gradients in policy and value networks */
void jax_policy_zero_grad(JaxPolicyNet* net);
void jax_value_zero_grad(JaxValueNet* vnet);

/* ==================================================================
 * Utility: Xavier/Orthogonal Init, Checkpointing
 * =================================================================== */

void jax_orthogonal_init_params(JaxPolicyNet* net, float gain);
void jax_value_orthogonal_init(JaxValueNet* vnet, float gain);

int jax_checkpoint_save(const JaxPolicyNet* net, const char* path);
int jax_checkpoint_load(JaxPolicyNet* net, const char* path);

/* ===================================================================
 * Lax Operations (shape, reduce, gather, compare, etc.)
 * =================================================================== */

/* Shape manipulation */
void jax_transpose_2d(float* out, const float* in, int rows, int cols);
void jax_transpose(float* out, const JaxTensor* t, const int* perm);
void jax_reshape(float* out, const float* in, int64_t total);
void jax_flatten(float* out, const JaxTensor* t);

/* Slicing */
void jax_slice_2d(float* out, const float* in, int in_rows, int in_cols,
                  int start_row, int start_col, int nrows, int ncols);
void jax_slice_nd(float* out, const float* in,
                  const int64_t* in_shape, const int64_t* in_strides, int ndim,
                  const int64_t* starts, const int64_t* limits, const int64_t* strides);

/* Gather/Scatter */
void jax_gather_1d(float* out, const float* in, const int64_t* indices, int n);
void jax_gather_2d_axis0(float* out, const float* in, const int* indices, int n, int cols);
void jax_scatter_2d_add(float* out, const float* in, const int* indices, int n, int cols);

/* Concatenate */
void jax_concat_2d_axis0(float* out, const float* a, int a_rows, int a_cols,
                         const float* b, int b_rows);
void jax_concat_2d_axis1(float* out, const float* a, int rows, int a_cols,
                         const float* b, int b_cols);

/* Reductions */
void jax_reduce_sum_axis0_2d(float* out, const float* in, int rows, int cols);
void jax_reduce_sum_axis1_2d(float* out, const float* in, int rows, int cols);
void jax_reduce_max_axis0_2d(float* out, const float* in, int rows, int cols);
void jax_reduce_min_axis0_2d(float* out, const float* in, int rows, int cols);
float jax_reduce_sum_all(const float* in, int64_t n);
float jax_reduce_max_all(const float* in, int64_t n);
float jax_reduce_min_all(const float* in, int64_t n);
float jax_reduce_mean(const float* in, int64_t n);
float jax_reduce_variance(const float* in, int64_t n);
int64_t jax_argmax(const float* in, int64_t n);
int64_t jax_argmin(const float* in, int64_t n);

/* Binary ops */
void jax_add_broadcast(float* out, const float* a, const float* b, int64_t n);
void jax_sub_broadcast(float* out, const float* a, const float* b, int64_t n);
void jax_mul_broadcast(float* out, const float* a, const float* b, int64_t n);
void jax_div_broadcast(float* out, const float* a, const float* b, int64_t n);

/* Unary ops */
void jax_neg(float* out, const float* in, int64_t n);
void jax_abs(float* out, const float* in, int64_t n);
void jax_reciprocal(float* out, const float* in, int64_t n);
void jax_sign(float* out, const float* in, int64_t n);
void jax_floor(float* out, const float* in, int64_t n);
void jax_ceil(float* out, const float* in, int64_t n);
void jax_round(float* out, const float* in, int64_t n);
void jax_exp(float* out, const float* in, int64_t n);
void jax_log(float* out, const float* in, int64_t n);
void jax_pow(float* out, const float* in, float exponent, int64_t n);
void jax_rsqrt(float* out, const float* in, int64_t n);
void jax_cbrt(float* out, const float* in, int64_t n);

/* Comparison */
void jax_equal(float* out, const float* a, const float* b, int64_t n);
void jax_not_equal(float* out, const float* a, const float* b, int64_t n);
void jax_greater(float* out, const float* a, const float* b, int64_t n);
void jax_greater_equal(float* out, const float* a, const float* b, int64_t n);
void jax_less(float* out, const float* a, const float* b, int64_t n);
void jax_less_equal(float* out, const float* a, const float* b, int64_t n);

/* Select / Clamp */
void jax_select(float* out, const float* mask, const float* a, const float* b, int64_t n);
void jax_clamp(float* out, const float* in, const float* min_val, const float* max_val, int64_t n);

/* Dot General */
void jax_dot_general_2d(const float* a, const float* b, float* out, int M, int K, int N);
void jax_dot_general_batched(const float* a, const float* b, float* out,
                             int batch, int M, int K, int N);

/* Attention */
void jax_attention_scaled(const float* Q, const float* K, const float* V,
                          float* out, int batch, int seq_q, int seq_k, int dim);

/* Convolution */
void jax_conv_2d(const float* input, const float* kernel, float* output,
                 int H, int W, int C_in, int H_k, int W_k, int C_out,
                 int stride_h, int stride_w, int pad_h, int pad_w);

/* Pad */
void jax_pad_2d(float* out, const float* in, int in_rows, int in_cols,
                int pad_top, int pad_bottom, int pad_left, int pad_right, float pad_value);

/* Sort */
void jax_sort_axis1(float* out, const float* in, int rows, int cols);
void jax_sort_key_val(float* keys_out, float* vals_out,
                      const float* keys, const float* vals, int n);
void jax_top_k(float* values_out, int64_t* indices_out,
               const float* in, int n, int k);

/* Stop Gradient */
void jax_stop_gradient(float* out, const float* in, int64_t n);

/* ===================================================================
 * Jaxpr IR (opaque handle — struct defined in jax_ir.c)
 * =================================================================== */

typedef struct JaxIr JaxIr;
typedef int64_t JaxVarId;

JaxIr* jax_ir_create(JaxArena* arena, int n_inputs);
JaxVarId jax_ir_add(JaxIr* ir, JaxVarId a, JaxVarId b);
JaxVarId jax_ir_mul(JaxIr* ir, JaxVarId a, JaxVarId b);
JaxVarId jax_ir_gemm(JaxIr* ir, JaxVarId a, JaxVarId b);
JaxVarId jax_ir_relu(JaxIr* ir, JaxVarId a);
JaxVarId jax_ir_reduce_sum(JaxIr* ir, JaxVarId a);
JaxVarId jax_ir_literal(JaxIr* ir, float value);
JaxVarId jax_ir_param(JaxIr* ir, int index);
void jax_ir_print(JaxIr* ir);
int jax_ir_backward(JaxIr* ir, JaxVarId loss_var);
int jax_ir_num_instrs(JaxIr* ir);

/* ===================================================================
 * Optimizer (Adam, Muon, SGD)
 * =================================================================== */

void jax_adam_step(JaxParam* p, float lr, float beta1, float beta2, float eps);
void jax_muon_step(JaxParam* p, float lr, float momentum, float nesterov);
void jax_sgd_step(JaxParam* p, float lr);
void jax_clip_grad_norm(JaxPolicyNet* net, float max_norm);
void jax_clip_grad_norm_value(JaxValueNet* vnet, float max_norm);
void jax_weight_decay(JaxPolicyNet* net, float decay);
void jax_weight_decay_value(JaxValueNet* vnet, float decay);

#endif /* JAX_NN_H */