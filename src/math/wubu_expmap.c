#include "wubu_expmap.h"
#include <math.h>
#include <string.h>
static double dp(const double *a, const double *b, size_t dim) {
    double s = 0; for (size_t i = 0; i < dim; i++) s += a[i]*b[i]; return s;
}
void wubu_poincare_exp(const double *x, const double *v, double c, double *out, size_t dim) {
    double lam = 2.0/(1.0 - c*dp(x,x,dim));
    for (size_t i = 0; i < dim; i++) out[i] = x[i] + v[i]/lam;
    double r = 0; for (size_t i = 0; i < dim; i++) r += out[i]*out[i];
    r = sqrt(r);
    double bound = 0.99/sqrt(c);
    if (r >= bound) { double s = bound/r; for (size_t i = 0; i < dim; i++) out[i] *= s; }
}
void wubu_poincare_log(const double *x, const double *y, double c, double *out, size_t dim) {
    double lam = 2.0/(1.0 - c*dp(x,x,dim));
    for (size_t i = 0; i < dim; i++) out[i] = lam*(y[i]-x[i]);
}