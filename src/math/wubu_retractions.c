#include "wubu_retractions.h"
#include <math.h>

void wubu_retract_sphere(double *x, const double *v, size_t dim) {
    for (size_t i = 0; i < dim; i++) x[i] += v[i];
    double r = 0; for (size_t i = 0; i < dim; i++) r += x[i]*x[i];
    r = sqrt(r);
    if (r > 1e-12) for (size_t i = 0; i < dim; i++) x[i] /= r;
}

void wubu_retract_stiefel(double *x, size_t dim) {
    /* Gram-Schmidt on single vector = normalize */
    double r = 0; for (size_t i = 0; i < dim; i++) r += x[i]*x[i];
    r = sqrt(r);
    if (r > 1e-12) for (size_t i = 0; i < dim; i++) x[i] /= r;
}