#include "wubu_pv_space.h"
#include <math.h>
#include <string.h>

void wubu_pv_add(const double *u, const double *v, double *out, size_t dim) {
    /* Einstein velocity addition:
       u ⊕ v = (u + v)/(1 + u·v) + (γ_u/(1+γ_u)) * (u·v/(1+u·v)) * u
       where γ_u = 1/sqrt(1 - ||u||^2) */
    double uu = 0, uv = 0;
    for (size_t i = 0; i < dim; i++) { uu += u[i]*u[i]; uv += u[i]*v[i]; }
    if (uu >= 1.0) uu = 0.99; /* clamp to unit ball */
    double gamma = 1.0 / sqrt(1.0 - uu);
    double denom = 1.0 + uv;
    if (fabs(denom) < 1e-12) denom = 1e-12;
    for (size_t i = 0; i < dim; i++) {
        out[i] = (u[i] + v[i]) / denom + (gamma / (1.0 + gamma)) * (uv / denom) * u[i];
    }
    /* project to ball */
    double r = 0;
    for (size_t i = 0; i < dim; i++) r += out[i]*out[i];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t i = 0; i < dim; i++) out[i] *= s; }
}
