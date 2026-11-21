#include "wubu_symplectic.h"
#include "wubu_expmap.h"
#include <math.h>
#include <string.h>
double wubu_symplectic_step(const double *q, const double *p,
    double (*gradV)(const double*, size_t, int, void*),
    void *ctx, size_t dim, double dt, double *q_out, double *p_out) {
    /* p_{n+1} = p_n - dt * grad_V(q_n) */
    for (size_t i = 0; i < dim; i++) {
        double g = gradV(q, dim, (int)i, ctx);
        p_out[i] = p[i] - dt * g;
    }
    /* q_{n+1} = exp_q(q_n, dt * p_{n+1}) */
    double v[64];
    for (size_t i = 0; i < dim && i < 64; i++) v[i] = dt * p_out[i];
    wubu_poincare_exp(q, v, 1.0, q_out, dim);
    /* energy = 0.5*|p|^2 + V(q) */
    double ke = 0;
    for (size_t i = 0; i < dim; i++) ke += 0.5 * p_out[i] * p_out[i];
    /* V = 0.5*|q|^2 (harmonic) */
    double ve = 0;
    for (size_t i = 0; i < dim; i++) ve += 0.5 * q_out[i] * q_out[i];
    return ke + ve;
}