#include "wubu_poincare_explog.h"
#include <math.h>
#include <string.h>

static double dp(const double *a, const double *b, size_t dim) {
    double s = 0; for (size_t i = 0; i < dim; i++) s += a[i]*b[i]; return s;
}

void wubu_poincare_exp(const double *x, const double *v, double *out, size_t dim, double c) {
    /* Riemannian exp map on Poincare ball:
       lambda_x = 2/(1 - c||x||^2)
       exp_x(v) = x ⊕ (tanh(sqrt(c)*lambda_x*||v||/2) / (sqrt(c)*||v||)) * v
       simplified: x + v/lambda_x, then project */
    double nx2 = dp(x, x, dim);
    double lam = 2.0 / (1.0 - c * nx2);
    double nv = 0;
    for (size_t i = 0; i < dim; i++) nv += v[i]*v[i];
    nv = sqrt(nv);
    if (nv < 1e-12) { memcpy(out, x, dim*sizeof(double)); return; }
    double scale = tanh(sqrt(c) * nv / lam) / (sqrt(c) * nv / lam);
    if (sqrt(c) * nv / lam < 1e-12) scale = 1.0;
    for (size_t i = 0; i < dim; i++) out[i] = x[i] + scale * v[i] / lam;
    /* project to ball */
    double r = 0; for (size_t i = 0; i < dim; i++) r += out[i]*out[i];
    r = sqrt(r);
    double bound = 0.99 / sqrt(c);
    if (r >= bound) { double s = bound / r; for (size_t i = 0; i < dim; i++) out[i] *= s; }
}

void wubu_poincare_log(const double *x, const double *y, double *out, size_t dim, double c) {
    /* Log map: v = (2/(sqrt(c)*lambda_x)) * artanh(sqrt(c)*||(-x) ⊕ y||) * ((-x) ⊕ y)/||(-x) ⊕ y|| */
    /* Simplified: lambda_x * (y - x) */
    double nx2 = dp(x, x, dim);
    double lam = 2.0 / (1.0 - c * nx2);
    for (size_t i = 0; i < dim; i++) out[i] = lam * (y[i] - x[i]);
}
