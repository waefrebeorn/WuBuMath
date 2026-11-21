#include "wubu_sphere_ops.h"
#include <math.h>

void wubu_sphere_tangent_proj(const double *x, const double *v, double *out, size_t dim) {
    double dot = 0;
    for (size_t i = 0; i < dim; i++) dot += x[i]*v[i];
    for (size_t i = 0; i < dim; i++) out[i] = v[i] - dot*x[i];
}

void wubu_sphere_exp(const double *x, const double *v, double *out, size_t dim) {
    double nv = 0;
    for (size_t i = 0; i < dim; i++) nv += v[i]*v[i];
    nv = sqrt(nv);
    if (nv < 1e-12) { for (size_t i=0;i<dim;i++) out[i]=x[i]; return; }
    for (size_t i = 0; i < dim; i++) {
        out[i] = cos(nv)*x[i] + (sin(nv)/nv)*v[i];
    }
}