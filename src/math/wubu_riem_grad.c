#include "wubu_riem_grad.h"

void wubu_lorentz_grad_proj(const double *x, const double *g, double *out, size_t dim) {
    /* tangent condition: <x, out>_L = 0. out = g + <g,x>_L x (since <x,x>_L=-1) */
    double ip = -x[0]*g[0];
    for (size_t i = 1; i <= dim; i++) ip += x[i]*g[i];
    for (size_t i = 0; i <= dim; i++) out[i] = g[i] + ip*x[i];
}