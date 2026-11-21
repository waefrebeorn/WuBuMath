#include "wubu_hyp_ffn.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
void wubu_hyp_ffn(const double *x, const double *W1, const double *W2, size_t dim, size_t hidden, double *out) {
    /* hidden layer */
    double *h = malloc(hidden*sizeof(double));
    for (size_t i = 0; i < hidden; i++) {
        double s = 0;
        for (size_t d = 0; d < dim; d++) s += W1[i*dim+d]*x[d];
        h[i] = s > 0 ? s : 0; /* relu */
    }
    /* output */
    for (size_t d = 0; d < dim; d++) {
        double s = 0;
        for (size_t i = 0; i < hidden; i++) s += W2[d*hidden+i]*h[i];
        out[d] = s;
    }
    free(h);
}