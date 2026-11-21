#include "wubu_hyp_residual.h"
#include "wubu_mobius_add.h"
#include <math.h>

void wubu_hyp_residual(const double *x, const double *fx, double *out, size_t dim) {
    /* Mobius addition x ⊕ fx then project */
    wubu_mobius_add(x, fx, out, dim, 1.0);
    double r = 0; for (size_t i = 0; i < dim; i++) r += out[i]*out[i];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t i = 0; i < dim; i++) out[i] *= s; }
}