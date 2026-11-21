#include "wubu_cancelfree.h"
#include <math.h>
double wubu_cf_poincare_dist(const double *u, const double *v, size_t dim, double c) {
    /* cancellation-free: use 1 - 2*<u,v>/(...) form */
    double nu=0, nv=0, uv=0;
    for (size_t i=0;i<dim;i++){ nu+=u[i]*u[i]; nv+=v[i]*v[i]; uv+=u[i]*v[i]; }
    double num = (1 - 2*uv + nu*nv) ; /* stable form */
    double denom = (1 - nu)*(1 - nv);
    if (fabs(denom)<1e-12) denom=1e-12;
    double arg = 1 + 2*c*num/denom;
    if (arg<1) arg=1;
    return acosh(sqrt(arg))/sqrt(c>0?c:1);
}
double wubu_cf_lorentz_dist(const double *u, const double *v, size_t dim) {
    /* cancellation-free Lorentz distance */
    double ip = -u[0]*v[0];
    for (size_t i=1;i<=dim;i++) ip += u[i]*v[i];
    double arg = -ip;
    if (arg < 1) arg = 1;
    return acosh(arg);
}