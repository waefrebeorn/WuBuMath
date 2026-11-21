#include "wubu_hyp_pool.h"
#include <math.h>

void wubu_hyp_pool(const double *pts, int n, size_t dim, double *out) {
    /* Euclidean mean then project to ball (approx) */
    for (size_t d = 0; d < dim; d++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += pts[i*dim + d];
        out[d] = s / (n > 0 ? n : 1);
    }
    double r = 0; for (size_t d = 0; d < dim; d++) r += out[d]*out[d];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t d = 0; d < dim; d++) out[d] *= s; }
}