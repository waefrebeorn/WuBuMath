#include "wubu_hyp_linear_attn.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
void wubu_hyp_linear_attn(const double *q, const double *k, const double *v, int n, size_t dim, double *out) {
    /* Simplified: standard attention with linear kernel (elu+1) */
    /* Compute K^T V (dim x dim) */
    double *KV = calloc(dim*dim, sizeof(double));
    double *Z = calloc(n, sizeof(double)); /* normalizer */
    for (int i = 0; i < n; i++) {
        for (size_t d1 = 0; d1 < dim; d1++) {
            double kfeat = (k[i*dim+d1] > -1) ? k[i*dim+d1]+1 : 0; /* elu+1 */
            for (size_t d2 = 0; d2 < dim; d2++) {
                KV[d1*dim+d2] += kfeat * v[i*dim+d2];
            }
            Z[i] += kfeat;
        }
    }
    for (int i = 0; i < n; i++) {
        for (size_t d1 = 0; d1 < dim; d1++) {
            double qfeat = (q[i*dim+d1] > -1) ? q[i*dim+d1]+1 : 0;
            double s = 0;
            for (size_t d2 = 0; d2 < dim; d2++) s += qfeat * KV[d1*dim+d2];
            out[i*dim+d1] = s / (Z[i] > 1e-12 ? Z[i] : 1);
        }
    }
    free(KV); free(Z);
}