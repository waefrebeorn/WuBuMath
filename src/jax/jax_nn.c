/*
 * jax_nn.cpp -- JAX-slermed: MLP + Value network with forward/backward
 *
 * Slermed from bear_nn.c (WuBuOS) with renaming and enhancements.
 * Pure C11, analytical gradients via chain rule.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "jax_nn.h"
#include "jax_arena.h"
#include "jax_simd.h"

/* ===================================================================
 * Network Creation
 * =================================================================== */

int jax_policy_create_mlp(JaxPolicyNet* net, JaxArena* param_arena,
                           int obs_dim, int act_dim, int act_discrete,
                           const int* hid_sizes, int num_hid) {
    if (!net || !param_arena || obs_dim <= 0 || act_dim <= 0) return -1;
    if (num_hid < 1) return -1;

    net->type = JAX_NET_MLP;
    net->obs_dim = obs_dim;
    net->act_dim = act_dim;
    net->act_discrete = act_discrete;
    net->param_arena = param_arena;
    net->gru = NULL;
    net->num_layers = num_hid + 1;
    net->fwd_stored = 0;

    net->layers = JAX_ARENA_ALLOC(param_arena, JaxLayer, net->num_layers);
    if (!net->layers) return -1;

    int prev = obs_dim;
    int layer_idx = 0;

    for (int i = 0; i < num_hid; ++i) {
        JaxLayer* l = &net->layers[layer_idx++];
        l->in_features = prev;
        l->out_features = hid_sizes[i];
        l->act = JAX_ACT_RELU;
        l->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
        if (!l->param) return -1;
        if (jax_param_create(param_arena, l->param, hid_sizes[i], prev, "policy.hid") != 0) return -1;
        l->act_storage = 0;
        prev = hid_sizes[i];
    }

    /* Actor head */
    JaxLayer* actor = &net->layers[layer_idx++];
    actor->in_features = prev;
    actor->out_features = act_dim;
    actor->act = JAX_ACT_NONE;
    actor->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
    if (!actor->param) return -1;
    if (jax_param_create(param_arena, actor->param, act_dim, prev, "policy.actor") != 0) return -1;
    actor->act_storage = 0;

    net->hid_size = 0;
    return 0;
}

int jax_policy_create_mingru(JaxPolicyNet* net, JaxArena* param_arena,
                              int obs_dim, int act_dim, int act_discrete,
                              int hid_size) {
    if (!net || !param_arena || obs_dim <= 0 || act_dim <= 0 || hid_size <= 0) return -1;

    net->type = JAX_NET_MINGU;
    net->obs_dim = obs_dim;
    net->act_dim = act_dim;
    net->act_discrete = act_discrete;
    net->hid_size = hid_size;
    net->param_arena = param_arena;
    net->fwd_stored = 0;

    net->num_layers = 2;
    net->layers = JAX_ARENA_ALLOC(param_arena, JaxLayer, net->num_layers);
    if (!net->layers) return -1;

    int layer_idx = 0;

    JaxLayer* in_proj = &net->layers[layer_idx++];
    in_proj->in_features = obs_dim;
    in_proj->out_features = hid_size;
    in_proj->act = JAX_ACT_RELU;
    in_proj->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
    if (!in_proj->param) return -1;
    if (jax_param_create(param_arena, in_proj->param, hid_size, obs_dim, "policy.in_proj") != 0) return -1;
    in_proj->act_storage = 0;

    JaxLayer* actor = &net->layers[layer_idx++];
    actor->in_features = hid_size;
    actor->out_features = act_dim;
    actor->act = JAX_ACT_NONE;
    actor->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
    if (!actor->param) return -1;
    if (jax_param_create(param_arena, actor->param, act_dim, hid_size, "policy.actor") != 0) return -1;
    actor->act_storage = 0;

    net->gru = JAX_ARENA_ALLOC(param_arena, JaxMinGRU, 1);
    if (!net->gru) return -1;
    if (jax_mingru_create(param_arena, net->gru, hid_size, hid_size) != 0) return -1;

    net->param_arena = param_arena;
    return 0;
}

/* ===================================================================
 * Forward Pass -- stores activations for backward
 * =================================================================== */

void jax_policy_forward(const JaxPolicyNet* net,
                         const JaxTensor* obs,
                         const JaxTensor* h_in,
                         JaxTensor* actions,
                         JaxTensor* logprobs,
                         JaxTensor* values,
                         JaxTensor* h_out,
                         JaxArena* temp_arena) {
    int batch = (int)obs->shape[0];
    int act_dim = net->act_dim;

    JaxTensor x = *obs;
    JaxTensor layer_out;

    if (net->type == JAX_NET_MLP) {
        /* Hidden layers */
        for (int i = 0; i < net->num_layers - 1; ++i) {
            JaxLayer* l = (JaxLayer*)&net->layers[i];
            jax_tensor_create(temp_arena, &layer_out, (int64_t[]){batch, l->out_features}, 2, JAX_DTYPE_F32, "layer_out");
            jax_mlp_layer(&x, &l->param->weight, NULL, &layer_out, l->act, temp_arena);
            /* Store post-activation */
            l->a_post = layer_out;
            l->act_storage = 1;
            x = layer_out;
        }

        /* Actor head: last layer */
        JaxLayer* actor = (JaxLayer*)&net->layers[net->num_layers - 1];
        jax_tensor_create(temp_arena, &layer_out, (int64_t[]){batch, act_dim}, 2, JAX_DTYPE_F32, "actor_logits");
        jax_mlp_layer(&x, &actor->param->weight, &actor->param->bias, &layer_out, JAX_ACT_NONE, temp_arena);
        /* Store pre-activation (logits) for backward */
        actor->z_pre = layer_out;
        actor->act_storage = 1;

        if (net->act_discrete) {
            float* logits = (float*)layer_out.data;
            float* probs = (float*)actions->data;
            for (int b = 0; b < batch; ++b) {
                float max_logit = -INFINITY;
                for (int a = 0; a < act_dim; ++a)
                    if (logits[b * act_dim + a] > max_logit) max_logit = logits[b * act_dim + a];
                float sum_exp = 0.0f;
                for (int a = 0; a < act_dim; ++a)
                    sum_exp += expf(logits[b * act_dim + a] - max_logit);
                for (int a = 0; a < act_dim; ++a)
                    probs[b * act_dim + a] = expf(logits[b * act_dim + a] - max_logit) / sum_exp;
                int sampled = 0;
                float max_p = 0.0f;
                for (int a = 0; a < act_dim; ++a)
                    if (probs[b * act_dim + a] > max_p) { max_p = probs[b * act_dim + a]; sampled = a; }
                for (int a = 0; a < act_dim; ++a) probs[b * act_dim + a] = 0.0f;
                probs[b * act_dim + sampled] = 1.0f;
                ((float*)logprobs->data)[b] = logf(max_p + 1e-8f);
            }
        } else {
            /* Continuous actions: Gaussian policy with fixed std */
            float* mu = (float*)layer_out.data;
            float* act = (float*)actions->data;
            float* lp = (float*)logprobs->data;
            float ls = net->logstd ? 0.0f : net->logstd_fixed;
            float var = expf(2.0f * ls);
            for (int b = 0; b < batch; ++b) {
                for (int a = 0; a < act_dim; ++a) {
                    float u1 = (float)(rand() + 1) / (float)((unsigned)RAND_MAX + 2);
                    float u2 = (float)(rand() + 1) / (float)((unsigned)RAND_MAX + 2);
                    float eps = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                    float std = expf(ls);
                    act[b * act_dim + a] = mu[b * act_dim + a] + std * eps;
                    float diff = act[b * act_dim + a] - mu[b * act_dim + a];
                    lp[b] += -0.5f * (diff * diff / var + logf(2.0f * 3.14159265f * var));
                }
            }
        }

        if (values) memset(values->data, 0, batch * sizeof(float));

    } else if (net->type == JAX_NET_MINGU) {
        int hid = net->hid_size;

        JaxTensor x_proj;
        jax_tensor_create(temp_arena, &x_proj, (int64_t[]){batch, hid}, 2, JAX_DTYPE_F32, "x_proj");
        jax_mlp_layer(obs, &net->layers[0].param->weight, NULL, &x_proj, JAX_ACT_RELU, temp_arena);
        net->layers[0].a_post = x_proj;
        net->layers[0].act_storage = 1;

        JaxTensor h_next;
        jax_tensor_create(temp_arena, &h_next, (int64_t[]){batch, hid}, 2, JAX_DTYPE_F32, "h_next");
        jax_mingru_step(net->gru, &x_proj, h_in, &h_next, temp_arena);

        if (h_out) *h_out = h_next;

        JaxTensor actor_out;
        jax_tensor_create(temp_arena, &actor_out, (int64_t[]){batch, act_dim}, 2, JAX_DTYPE_F32, "actor_logits");
        jax_mlp_layer(&h_next, &net->layers[1].param->weight, &net->layers[1].param->bias, &actor_out, JAX_ACT_NONE, temp_arena);
        net->layers[1].z_pre = actor_out;
        net->layers[1].act_storage = 1;

        if (net->act_discrete) {
            float* logits = (float*)actor_out.data;
            float* probs = (float*)actions->data;
            for (int b = 0; b < batch; ++b) {
                float max_logit = -INFINITY;
                for (int a = 0; a < act_dim; ++a)
                    if (logits[b * act_dim + a] > max_logit) max_logit = logits[b * act_dim + a];
                float sum_exp = 0.0f;
                for (int a = 0; a < act_dim; ++a)
                    sum_exp += expf(logits[b * act_dim + a] - max_logit);
                for (int a = 0; a < act_dim; ++a)
                    probs[b * act_dim + a] = expf(logits[b * act_dim + a] - max_logit) / sum_exp;
                int sampled = 0; float max_p = 0.0f;
                for (int a = 0; a < act_dim; ++a)
                    if (probs[b * act_dim + a] > max_p) { max_p = probs[b * act_dim + a]; sampled = a; }
                for (int a = 0; a < act_dim; ++a) probs[b * act_dim + a] = 0.0f;
                probs[b * act_dim + sampled] = 1.0f;
                ((float*)logprobs->data)[b] = logf(max_p + 1e-8f);
            }
        } else {
            float* mu = (float*)actor_out.data;
            float* act = (float*)actions->data;
            float* lp = (float*)logprobs->data;
            float ls = net->logstd ? 0.0f : net->logstd_fixed;
            float var = expf(2.0f * ls);
            for (int b = 0; b < batch; ++b) {
                for (int a = 0; a < act_dim; ++a) {
                    float u1 = (float)(rand() + 1) / (float)((unsigned)RAND_MAX + 2);
                    float u2 = (float)(rand() + 1) / (float)((unsigned)RAND_MAX + 2);
                    float eps = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                    float std = expf(ls);
                    act[b * act_dim + a] = mu[b * act_dim + a] + std * eps;
                    float diff = act[b * act_dim + a] - mu[b * act_dim + a];
                    lp[b] += -0.5f * (diff * diff / var + logf(2.0f * 3.14159265f * var));
                }
            }
        }

        if (values) memset(values->data, 0, batch * sizeof(float));
    }

    ((JaxPolicyNet*)net)->fwd_stored = 1;
}

void jax_policy_sample(JaxPolicyNet* net, JaxTensor* actions, JaxTensor* logprobs,
                        uint64_t* rng_state) {
    (void)rng_state;
    int batch = (int)actions->shape[0];
    int act_dim = net->act_dim;

    if (net->act_discrete) {
        float* probs = (float*)actions->data;
        for (int b = 0; b < batch; ++b) {
            float max_logit = -INFINITY;
            for (int a = 0; a < act_dim; ++a) {
                if (probs[b * act_dim + a] > max_logit)
                    max_logit = probs[b * act_dim + a];
            }
            float sum_exp = 0.0f;
            for (int a = 0; a < act_dim; ++a)
                sum_exp += expf(probs[b * act_dim + a] - max_logit);
            for (int a = 0; a < act_dim; ++a)
                probs[b * act_dim + a] = expf(probs[b * act_dim + a] - max_logit) / sum_exp;

            float max_val = -INFINITY;
            int sampled = 0;
            for (int a = 0; a < act_dim; ++a) {
                float gumbel = -logf(-logf((float)rand() / RAND_MAX + 1e-8f));
                float score = logf(probs[b * act_dim + a] + 1e-8f) + gumbel;
                if (score > max_val) { max_val = score; sampled = a; }
            }

            for (int a = 0; a < act_dim; ++a) probs[b * act_dim + a] = 0;
            probs[b * act_dim + sampled] = 1.0f;
            ((float*)logprobs->data)[b] = logf(probs[b * act_dim + sampled] + 1e-8f);
        }
    }
}

void jax_policy_deterministic(JaxPolicyNet* net, JaxTensor* actions) {
    int batch = (int)actions->shape[0];
    int act_dim = net->act_dim;
    float* probs = (float*)actions->data;

    if (net->act_discrete) {
        for (int b = 0; b < batch; ++b) {
            float max_p = -INFINITY;
            int argmax = 0;
            for (int a = 0; a < act_dim; ++a) {
                if (probs[b * act_dim + a] > max_p) {
                    max_p = probs[b * act_dim + a];
                    argmax = a;
                }
            }
            for (int a = 0; a < act_dim; ++a) probs[b * act_dim + a] = 0;
            probs[b * act_dim + argmax] = 1.0f;
        }
    }
}

int jax_policy_get_params(const JaxPolicyNet* net, float* out, int max_params) {
    int idx = 0;
    for (int i = 0; i < net->num_layers; ++i) {
        JaxLayer* l = &net->layers[i];
        int n = l->param->weight.shape[0] * l->param->weight.shape[1];
        if (out && idx + n > max_params) return -1;
        if (out) {
            float* w = (float*)l->param->weight.data;
            memcpy(out + idx, w, n * sizeof(float));
        }
        idx += n;
    }
    if (net->gru) {
        JaxParam* params[] = { &net->gru->Wz, &net->gru->Uz, &net->gru->bz,
                                &net->gru->Wr, &net->gru->Ur, &net->gru->br,
                                &net->gru->Wn, &net->gru->Un, &net->gru->bn };
        for (int i = 0; i < 9; ++i) {
            int n = params[i]->weight.shape[0] * params[i]->weight.shape[1];
            if (out && idx + n > max_params) return -1;
            if (out) {
                float* w = (float*)params[i]->weight.data;
                memcpy(out + idx, w, n * sizeof(float));
            }
            idx += n;
        }
    }
    return idx;
}

int jax_policy_set_params(JaxPolicyNet* net, const float* in, int num_params) {
    int idx = 0;
    for (int i = 0; i < net->num_layers; ++i) {
        JaxLayer* l = &net->layers[i];
        int n = l->param->weight.shape[0] * l->param->weight.shape[1];
        if (idx + n > num_params) return -1;
        float* w = (float*)l->param->weight.data;
        memcpy(w, in + idx, n * sizeof(float));
        idx += n;
    }
    if (net->gru) {
        JaxParam* params[] = { &net->gru->Wz, &net->gru->Uz, &net->gru->bz,
                                &net->gru->Wr, &net->gru->Ur, &net->gru->br,
                                &net->gru->Wn, &net->gru->Un, &net->gru->bn };
        for (int i = 0; i < 9; ++i) {
            int n = params[i]->weight.shape[0] * params[i]->weight.shape[1];
            if (idx + n > num_params) return -1;
            float* w = (float*)params[i]->weight.data;
            memcpy(w, in + idx, n * sizeof(float));
            idx += n;
        }
    }
    return 0;
}

void jax_orthogonal_init_params(JaxPolicyNet* net, float gain) {
    for (int i = 0; i < net->num_layers; ++i) {
        JaxLayer* l = &net->layers[i];
        jax_orthogonal_init(&l->param->weight, gain);
        if (l->param->bias.data) jax_tensor_fill(&l->param->bias, 0.0f);
    }
    if (net->gru) {
        JaxParam* params[] = { &net->gru->Wz, &net->gru->Uz, &net->gru->bz,
                                &net->gru->Wr, &net->gru->Ur, &net->gru->br,
                                &net->gru->Wn, &net->gru->Un, &net->gru->bn };
        for (int i = 0; i < 9; ++i) {
            jax_orthogonal_init(&params[i]->weight, gain);
            if (params[i]->bias.data) jax_tensor_fill(&params[i]->bias, 0.0f);
        }
    }
}

/* ===================================================================
 * Value Network
 * =================================================================== */

int jax_value_create(JaxValueNet* vnet, JaxArena* param_arena,
                      int obs_dim, const int* hid_sizes, int num_hid) {
    if (!vnet || !param_arena) return -1;

    vnet->param_arena = param_arena;
    vnet->num_layers = num_hid + 1;
    vnet->fwd_stored = 0;
    vnet->layers = JAX_ARENA_ALLOC(param_arena, JaxLayer, vnet->num_layers);
    if (!vnet->layers) return -1;

    int prev = obs_dim;
    for (int i = 0; i < num_hid; ++i) {
        JaxLayer* l = &vnet->layers[i];
        l->in_features = prev;
        l->out_features = hid_sizes[i];
        l->act = JAX_ACT_RELU;
        l->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
        if (!l->param) return -1;
        if (jax_param_create(param_arena, l->param, hid_sizes[i], prev, "value.hid") != 0) return -1;
        l->act_storage = 0;
        prev = hid_sizes[i];
    }

    JaxLayer* out = &vnet->layers[vnet->num_layers - 1];
    out->in_features = prev;
    out->out_features = 1;
    out->act = JAX_ACT_NONE;
    out->param = JAX_ARENA_ALLOC(param_arena, JaxParam, 1);
    if (!out->param) return -1;
    if (jax_param_create(param_arena, out->param, 1, prev, "value.out") != 0) return -1;
    out->act_storage = 0;

    return 0;
}

void jax_value_forward(const JaxValueNet* vnet,
                        const JaxTensor* obs,
                        JaxTensor* values,
                        JaxArena* temp_arena) {
    JaxTensor x = *obs;
    JaxTensor layer_out;
    int batch = (int)obs->shape[0];

    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxLayer* l = (JaxLayer*)&vnet->layers[i];
        jax_tensor_create(temp_arena, &layer_out, (int64_t[]){batch, l->out_features}, 2, JAX_DTYPE_F32, "value_layer");
        jax_mlp_layer(&x, &l->param->weight, &l->param->bias, &layer_out, l->act, temp_arena);
        if (i < vnet->num_layers - 1) {
            l->a_post = layer_out;
        } else {
            l->z_pre = layer_out;
        }
        l->act_storage = 1;
        x = layer_out;
    }

    float* v = (float*)x.data;
    for (int b = 0; b < batch; ++b) ((float*)values->data)[b] = v[b];

    ((JaxValueNet*)vnet)->fwd_stored = 1;
}

void jax_value_orthogonal_init(JaxValueNet* vnet, float gain) {
    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxLayer* l = &vnet->layers[i];
        jax_orthogonal_init(&l->param->weight, gain);
        if (l->param->bias.data) jax_tensor_fill(&l->param->bias, 0.0f);
    }
}

/* ===================================================================
 * Backward Pass -- Analytical Gradients
 * =================================================================== */

int jax_policy_backward(JaxPolicyNet* net,
                         const JaxTensor* obs,
                         const JaxTensor* actions,
                         const JaxTensor* old_logprobs,
                         const JaxTensor* advantages,
                         float clip_coef,
                         float policy_grad_scale,
                         JaxArena* temp_arena) {
    if (!net || !net->fwd_stored) return -1;

    if (net->act_discrete) {
        return jax_policy_backward_discrete(net, obs, actions, old_logprobs, advantages,
                                             clip_coef, policy_grad_scale, temp_arena);
    } else {
        return jax_policy_backward_continuous(net, obs, actions, old_logprobs, advantages,
                                              clip_coef, policy_grad_scale, temp_arena);
    }
}

int jax_policy_backward_discrete(JaxPolicyNet* net,
                                  const JaxTensor* obs,
                                  const JaxTensor* actions,
                                  const JaxTensor* old_logprobs,
                                  const JaxTensor* advantages,
                                  float clip_coef,
                                  float policy_grad_scale,
                                  JaxArena* temp_arena) {

    int mb = (int)obs->shape[0];
    int act_dim = net->act_dim;
    int last = net->num_layers - 1;

    float* dlogit = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * act_dim);
    if (!dlogit) return -1;

    float* old_lp = (float*)old_logprobs->data;
    float* adv = (float*)advantages->data;
    float* chosen = (float*)actions->data;

    JaxLayer* actor = &net->layers[last];
    float* logits = (float*)actor->z_pre.data;

    for (int i = 0; i < mb; ++i) {
        float max_logit = -INFINITY;
        for (int a = 0; a < act_dim; ++a)
            if (logits[i * act_dim + a] > max_logit) max_logit = logits[i * act_dim + a];
        float sum_exp = 0.0f;
        for (int a = 0; a < act_dim; ++a)
            sum_exp += expf(logits[i * act_dim + a] - max_logit);

        float pi_new[64];
        for (int a = 0; a < act_dim; ++a)
            pi_new[a] = expf(logits[i * act_dim + a] - max_logit) / sum_exp;

        int chosen_a = 0;
        for (int a = 0; a < act_dim; ++a)
            if (chosen[i * act_dim + a] > 0.5f) { chosen_a = a; break; }

        float log_pi_new_chosen = logf(pi_new[chosen_a] + 1e-8f);
        float ratio = expf(log_pi_new_chosen - old_lp[i]);
        float clipped = fmaxf(fminf(ratio, 1.0f + clip_coef), 1.0f - clip_coef);

        int is_clipped = (ratio > 1.0f + clip_coef || ratio < 1.0f - clip_coef);
        float signal = is_clipped ? 0.0f : ratio * adv[i] * policy_grad_scale / (float)mb;

        for (int a = 0; a < act_dim; ++a) {
            float indicator = (a == chosen_a) ? 1.0f : 0.0f;
            dlogit[i * act_dim + a] = signal * (indicator - pi_new[a]);
        }
    }

    float* dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * actor->in_features);
    if (!dx) return -1;

    float* w_out = (float*)actor->param->weight.data;
    float* grad_w_out = (float*)actor->param->grad.data;

    float* h_prev;
    int prev_feat;
    if (last > 0) {
        JaxLayer* prev_layer = &net->layers[last - 1];
        h_prev = (float*)prev_layer->a_post.data;
        prev_feat = prev_layer->out_features;
    } else {
        h_prev = (float*)obs->data;
        prev_feat = net->obs_dim;
    }

    for (int d = 0; d < act_dim; ++d) {
        for (int k = 0; k < prev_feat; ++k) {
            float g = 0.0f;
            for (int i = 0; i < mb; ++i)
                g += dlogit[i * act_dim + d] * h_prev[i * prev_feat + k];
            grad_w_out[d * prev_feat + k] += g;
        }
    }

    for (int i = 0; i < mb; ++i) {
        for (int k = 0; k < prev_feat; ++k) {
            float g = 0.0f;
            for (int d = 0; d < act_dim; ++d)
                g += dlogit[i * act_dim + d] * w_out[d * prev_feat + k];
            dx[i * prev_feat + k] = g;
        }
    }

    for (int li = last - 1; li >= 0; --li) {
        JaxLayer* l = &net->layers[li];
        int in_f = l->in_features;
        int out_f = l->out_features;
        float* w = (float*)l->param->weight.data;
        float* grad_w = (float*)l->param->grad.data;

        if (l->act == JAX_ACT_RELU && l->act_storage) {
            for (int i = 0; i < mb * out_f; ++i) {
                if (((float*)l->a_post.data)[i] <= 0.0f)
                    dx[i] = 0.0f;
            }
        }

        float* layer_input;
        int in_feat;
        if (li > 0) {
            JaxLayer* prev_l = &net->layers[li - 1];
            layer_input = (float*)prev_l->a_post.data;
            in_feat = prev_l->out_features;
        } else {
            layer_input = (float*)obs->data;
            in_feat = net->obs_dim;
        }

        for (int of = 0; of < out_f; ++of) {
            for (int k = 0; k < in_feat; ++k) {
                float g = 0.0f;
                for (int i = 0; i < mb; ++i)
                    g += dx[i * out_f + of] * layer_input[i * in_feat + k];
                grad_w[of * in_feat + k] += g;
            }
        }

        if (li > 0) {
            float* new_dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * in_feat);
            if (!new_dx) return -1;
            for (int i = 0; i < mb; ++i) {
                for (int k = 0; k < in_feat; ++k) {
                    float g = 0.0f;
                    for (int of = 0; of < out_f; ++of)
                        g += dx[i * out_f + of] * w[of * in_feat + k];
                    new_dx[i * in_feat + k] = g;
                }
            }
            dx = new_dx;
        }
    }

    return 0;
}

int jax_policy_backward_continuous(JaxPolicyNet* net,
                                   const JaxTensor* obs,
                                   const JaxTensor* actions,
                                   const JaxTensor* old_logprobs,
                                   const JaxTensor* advantages,
                                   float clip_coef,
                                   float policy_grad_scale,
                                   JaxArena* temp_arena) {
    if (!net || !net->fwd_stored) return -1;

    int mb = (int)obs->shape[0];
    int act_dim = net->act_dim;
    int last = net->num_layers - 1;

    float* old_lp = (float*)old_logprobs->data;
    float* adv = (float*)advantages->data;
    float* act = (float*)actions->data;

    JaxLayer* actor = &net->layers[last];
    float* mu = (float*)actor->z_pre.data;

    float ls = net->logstd ? 0.0f : net->logstd_fixed;
    float var = expf(2.0f * ls);

    float* dmu = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * act_dim);
    if (!dmu) return -1;

    for (int i = 0; i < mb; ++i) {
        float new_lp = 0.0f;
        for (int a = 0; a < act_dim; ++a) {
            float diff = act[i * act_dim + a] - mu[i * act_dim + a];
            new_lp += -0.5f * (diff * diff / var + logf(2.0f * 3.14159265f * var));
        }
        float ratio = expf(new_lp - old_lp[i]);
        float clipped = fmaxf(fminf(ratio, 1.0f + clip_coef), 1.0f - clip_coef);
        int is_clipped = (ratio > 1.0f + clip_coef || ratio < 1.0f - clip_coef);
        float signal = is_clipped ? 0.0f : ratio * adv[i] * policy_grad_scale / (float)mb;

        for (int a = 0; a < act_dim; ++a) {
            dmu[i * act_dim + a] = signal * (act[i * act_dim + a] - mu[i * act_dim + a]) / var;
        }
    }

    float* dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * actor->in_features);
    if (!dx) return -1;

    float* w_out = (float*)actor->param->weight.data;
    float* grad_w_out = (float*)actor->param->grad.data;

    float* h_prev;
    int prev_feat;
    if (last > 0) {
        h_prev = (float*)net->layers[last - 1].a_post.data;
        prev_feat = net->layers[last - 1].out_features;
    } else {
        h_prev = (float*)obs->data;
        prev_feat = net->obs_dim;
    }

    for (int d = 0; d < act_dim; ++d) {
        for (int k = 0; k < prev_feat; ++k) {
            float g = 0.0f;
            for (int i = 0; i < mb; ++i)
                g += dmu[i * act_dim + d] * h_prev[i * prev_feat + k];
            grad_w_out[d * prev_feat + k] += g;
        }
    }

    for (int i = 0; i < mb; ++i) {
        for (int k = 0; k < prev_feat; ++k) {
            float g = 0.0f;
            for (int d = 0; d < act_dim; ++d)
                g += dmu[i * act_dim + d] * w_out[d * prev_feat + k];
            dx[i * prev_feat + k] = g;
        }
    }

    for (int li = last - 1; li >= 0; --li) {
        JaxLayer* l = &net->layers[li];
        int in_f = l->in_features;
        int out_f = l->out_features;
        float* w = (float*)l->param->weight.data;
        float* grad_w = (float*)l->param->grad.data;

        if (l->act == JAX_ACT_RELU && l->act_storage) {
            for (int i = 0; i < mb * out_f; ++i) {
                if (((float*)l->a_post.data)[i] <= 0.0f)
                    dx[i] = 0.0f;
            }
        }

        float* layer_input;
        int in_feat;
        if (li > 0) {
            layer_input = (float*)net->layers[li - 1].a_post.data;
            in_feat = net->layers[li - 1].out_features;
        } else {
            layer_input = (float*)obs->data;
            in_feat = net->obs_dim;
        }

        for (int of = 0; of < out_f; ++of) {
            for (int k = 0; k < in_feat; ++k) {
                float g = 0.0f;
                for (int i = 0; i < mb; ++i)
                    g += dx[i * out_f + of] * layer_input[i * in_feat + k];
                grad_w[of * in_feat + k] += g;
            }
        }

        if (li > 0) {
            float* new_dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * in_feat);
            if (!new_dx) return -1;
            for (int i = 0; i < mb; ++i) {
                for (int k = 0; k < in_feat; ++k) {
                    float g = 0.0f;
                    for (int of = 0; of < out_f; ++of)
                        g += dx[i * out_f + of] * w[of * in_feat + k];
                    new_dx[i * in_feat + k] = g;
                }
            }
            dx = new_dx;
        }
    }

    return 0;
}

int jax_value_backward(JaxValueNet* vnet,
                        const JaxTensor* obs,
                        const JaxTensor* values,
                        const JaxTensor* targets,
                        float vf_coef,
                        JaxArena* temp_arena) {
    if (!vnet || !vnet->fwd_stored) return -1;

    int mb = (int)obs->shape[0];
    int last = vnet->num_layers - 1;

    float* v_pred = (float*)values->data;
    float* v_target = (float*)targets->data;

    float* dv = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb);
    if (!dv) return -1;
    for (int i = 0; i < mb; ++i)
        dv[i] = (v_pred[i] - v_target[i]) * 2.0f * vf_coef / (float)mb;

    JaxLayer* out_layer = &vnet->layers[last];
    float* w_out = (float*)out_layer->param->weight.data;
    float* grad_w_out = (float*)out_layer->param->grad.data;
    int out_f = out_layer->out_features;
    int in_f = out_layer->in_features;

    float* h_prev;
    int prev_feat;
    if (last > 0) {
        h_prev = (float*)vnet->layers[last - 1].a_post.data;
        prev_feat = vnet->layers[last - 1].out_features;
    } else {
        h_prev = (float*)obs->data;
        prev_feat = obs->shape[1];
    }

    for (int k = 0; k < prev_feat; ++k) {
        float g = 0.0f;
        for (int i = 0; i < mb; ++i)
            g += dv[i] * h_prev[i * prev_feat + k];
        grad_w_out[k] += g;
    }

    float* dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * prev_feat);
    if (!dx) return -1;
    for (int i = 0; i < mb; ++i)
        for (int k = 0; k < prev_feat; ++k)
            dx[i * prev_feat + k] = dv[i] * w_out[k];

    for (int li = last - 1; li >= 0; --li) {
        JaxLayer* l = &vnet->layers[li];
        int li_out = l->out_features;
        int li_in = l->in_features;
        float* w = (float*)l->param->weight.data;
        float* grad_w = (float*)l->param->grad.data;

        if (l->act == JAX_ACT_RELU && l->act_storage) {
            for (int i = 0; i < mb * li_out; ++i) {
                if (((float*)l->a_post.data)[i] <= 0.0f)
                    dx[i] = 0.0f;
            }
        }

        float* layer_input;
        int in_feat;
        if (li > 0) {
            layer_input = (float*)vnet->layers[li - 1].a_post.data;
            in_feat = vnet->layers[li - 1].out_features;
        } else {
            layer_input = (float*)obs->data;
            in_feat = (int)obs->shape[1];
        }

        for (int of = 0; of < li_out; ++of) {
            for (int k = 0; k < in_feat; ++k) {
                float g = 0.0f;
                for (int i = 0; i < mb; ++i)
                    g += dx[i * li_out + of] * layer_input[i * in_feat + k];
                grad_w[of * in_feat + k] += g;
            }
        }

        if (li > 0) {
            float* new_dx = (float*)JAX_ARENA_ALLOC(temp_arena, float, mb * in_feat);
            if (!new_dx) return -1;
            for (int i = 0; i < mb; ++i) {
                for (int k = 0; k < in_feat; ++k) {
                    float g = 0.0f;
                    for (int of = 0; of < li_out; ++of)
                        g += dx[i * li_out + of] * w[of * in_feat + k];
                    new_dx[i * in_feat + k] = g;
                }
            }
            dx = new_dx;
        }
    }

    return 0;
}

void jax_policy_zero_grad(JaxPolicyNet* net) {
    if (!net || !net->layers) return;
    for (int i = 0; i < net->num_layers; ++i) {
        JaxParam* p = net->layers[i].param;
        if (p && p->grad.data) {
            int n = (int)jax_tensor_numel(&p->grad);
            float* g = (float*)p->grad.data;
            for (int j = 0; j < n; ++j) g[j] = 0.0f;
        }
    }
}

void jax_value_zero_grad(JaxValueNet* vnet) {
    if (!vnet || !vnet->layers) return;
    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxParam* p = vnet->layers[i].param;
        if (p && p->grad.data) {
            int n = (int)jax_tensor_numel(&p->grad);
            float* g = (float*)p->grad.data;
            for (int j = 0; j < n; ++j) g[j] = 0.0f;
        }
    }
}

/* ===================================================================
 * Checkpointing (stub)
 * =================================================================== */
int jax_checkpoint_save(const JaxPolicyNet* net, const char* path) {
    (void)net; (void)path;
    return 0;
}

int jax_checkpoint_load(JaxPolicyNet* net, const char* path) {
    (void)net; (void)path;
    return 0;
}
