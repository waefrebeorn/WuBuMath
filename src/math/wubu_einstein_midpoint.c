#include "wubu_einstein_midpoint.h"
#include <math.h>
#include <string.h>

void wubu_einstein_midpoint(const double *points, const double *weights, int npts, size_t dim, double *out) {
    /* sum gamma_i * w_i * x_i / sum gamma_i * w_i, gamma = x_t (time component) */
    double denom = 0;
    for (size_t d = 0; d <= dim; d++) out[d] = 0;
    for (int i = 0; i < npts; i++) {
        const double *p = &points[i*(dim+1)];
        double gamma = p[0]; /* time component */
        double w = weights[i] * gamma;
        denom += w;
        for (size_t d = 0; d <= dim; d++) out[d] += w * p[d];
    }
    if (fabs(denom) > 1e-12) for (size_t d = 0; d <= dim; d++) out[d] /= denom;
}