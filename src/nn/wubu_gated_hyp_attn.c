#include "wubu_gated_hyp_attn.h"
#include "wubu_hyperbolic_attn.h"
#include <math.h>
#include <stdlib.h>

void wubu_gated_hyp_attn(const double *x, const double *Wg, const double **keys, const double **vals, int nkeys, size_t dim, double *out) {
    /* gate */
    double g = 0;
    for (size_t i = 0; i < dim; i++) g += Wg[i]*x[i];
    g = 1.0/(1.0+exp(-g));
    /* attention weights */
    double *w = malloc(nkeys*sizeof(double));
    wubu_hyp_attn_weights(x, keys, nkeys, dim, w);
    /* attn = sum w_i * vals_i */
    for (size_t d = 0; d < dim; d++) {
        double a = 0;
        for (int i = 0; i < nkeys; i++) a += w[i]*vals[i][d];
        out[d] = g*a + (1-g)*x[d];
    }
    /* project to ball */
    double r = 0; for (size_t d = 0; d < dim; d++) r += out[d]*out[d];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t d = 0; d < dim; d++) out[d] *= s; }
    free(w);
}