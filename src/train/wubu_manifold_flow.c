#include "wubu_manifold_flow.h"
#include "wubu_poincare_explog.h"
#include <math.h>
#include <string.h>

static double pnrm(const double *x, size_t dim) {
    double s = 0; for (size_t i = 0; i < dim; i++) s += x[i]*x[i]; return sqrt(s);
}

void wubu_poincare_interp(const double *x, const double *y, double t, double *out, size_t dim) {
    /* Geodesic interpolation on Poincare ball:
       gamma(t) = exp_x(t * log_x(y)) */
    double v[64];
    wubu_poincare_log(x, y, v, dim, 1.0);
    for (size_t i = 0; i < dim && i < 64; i++) v[i] *= t;
    wubu_poincare_exp(x, v, out, dim, 1.0);
}

void wubu_flow_velocity(const double *x, const double *y, double sigma, double *v, size_t dim) {
    /* Velocity field: v = (y - x) / sigma, projected to tangent */
    wubu_poincare_log(x, y, v, dim, 1.0);
    if (sigma > 1e-12) for (size_t i = 0; i < dim; i++) v[i] /= sigma;
}
