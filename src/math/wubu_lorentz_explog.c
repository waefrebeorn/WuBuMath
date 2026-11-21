#include "wubu_lorentz_explog.h"
#include <math.h>

void wubu_lorentz_exp(const double *x, const double *v, double *out, size_t dim) {
    /* v must be tangent: <x,v>_L = 0. out = cosh(||v||) x + sinh(||v||)/||v|| v */
    double nv = 0;
    /* Lorentz norm of v: sqrt(<v,v>_L) with sign handling; assume spacelike */
    double vv = -v[0]*v[0];
    for (size_t i = 1; i <= dim; i++) vv += v[i]*v[i];
    nv = sqrt(fabs(vv));
    if (nv < 1e-12) { for (size_t i = 0; i <= dim; i++) out[i] = x[i]; return; }
    double c = cosh(nv), s = sinh(nv)/nv;
    for (size_t i = 0; i <= dim; i++) out[i] = c*x[i] + s*v[i];
}

void wubu_lorentz_log(const double *x, const double *y, double *out, size_t dim) {
    double ip = -x[0]*y[0];
    for (size_t i = 1; i <= dim; i++) ip += x[i]*y[i];
    double d = acosh(fabs(ip) > 1 ? fabs(ip) : 1);
    /* out = d / sinh(d) * (y - cosh(d) x) approximated */
    double s = (fabs(sinh(d)) > 1e-12) ? d/sinh(d) : 1.0;
    for (size_t i = 0; i <= dim; i++) out[i] = s*(y[i] - cosh(d)*x[i]);
}