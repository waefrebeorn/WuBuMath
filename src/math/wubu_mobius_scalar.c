#include "wubu_mobius_scalar.h"
#include <math.h>

void wubu_mobius_scalar(double r, const double *x, double *out, size_t dim, double c) {
    double nx = 0;
    for (size_t i = 0; i < dim; i++) nx += x[i]*x[i];
    nx = sqrt(nx);
    if (nx < 1e-12) { for (size_t i=0;i<dim;i++) out[i]=0; return; }
    double factor = tanh(r * atanh(sqrt(c)*nx)) / (sqrt(c)*nx);
    for (size_t i = 0; i < dim; i++) out[i] = (factor / sqrt(c)) * x[i]; /* approx */
    /* project to ball */
    double r2 = 0; for (size_t i=0;i<dim;i++) r2 += out[i]*out[i];
    r2 = sqrt(r2);
    if (r2 >= 0.99) { double s=0.99/r2; for(size_t i=0;i<dim;i++) out[i]*=s; }
}