#include "wubu_hyp_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
void wubu_hyp_attention(const double *Q, const double *K, const double *V, int n, size_t dim, double *out) {
    /* scaled dot-product attention */
    memset(out, 0, n*dim*sizeof(double));
    for (int i = 0; i < n; i++) {
        /* compute attention weights */
        double *w = malloc(n*sizeof(double));
        double maxw = -1e18;
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (size_t d = 0; d < dim; d++) s += Q[i*dim+d]*K[j*dim+d];
            w[j] = s / sqrt((double)dim);
            if (w[j] > maxw) maxw = w[j];
        }
        double sum = 0;
        for (int j = 0; j < n; j++) { w[j] = exp(w[j]-maxw); sum += w[j]; }
        for (int j = 0; j < n; j++) w[j] /= sum;
        /* weighted sum of V */
        for (int j = 0; j < n; j++) {
            for (size_t d = 0; d < dim; d++) out[i*dim+d] += w[j]*V[j*dim+d];
        }
        free(w);
    }
}