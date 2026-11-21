#include "wubu_hyperbolic_conv.h"
#include <math.h>
#include <string.h>

void wubu_mobius_matvec(const double *M, const double *x, double *out, size_t in, size_t outd, double c) {
    /* Mobius linear transformation (hyperbolic matrix-vector):
       M ⊗ x = (1/(1 + ...)) * ... 
       Simplified: apply matrix, then project to ball with proper scaling. */
    for (size_t j = 0; j < outd; j++) {
        double s = 0;
        for (size_t i = 0; i < in; i++) s += M[j*in + i] * x[i];
        out[j] = s;
    }
    /* Project to Poincare ball: scale if outside */
    double r = 0;
    for (size_t j = 0; j < outd; j++) r += out[j]*out[j];
    r = sqrt(r);
    double bound = 0.99 / sqrt(c > 0 ? c : 1.0);
    if (r >= bound && r > 1e-12) {
        double sc = bound / r;
        for (size_t j = 0; j < outd; j++) out[j] *= sc;
    }
}

void wubu_hyp_tanh(const double *x, double *out, size_t dim) {
    for (size_t i = 0; i < dim; i++) out[i] = tanh(x[i]);
    double r = 0; for (size_t i = 0; i < dim; i++) r += out[i]*out[i];
    r = sqrt(r);
    if (r >= 0.99) { double sc = 0.99/r; for (size_t i = 0; i < dim; i++) out[i] *= sc; }
}
