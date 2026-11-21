#include "wubu_hyp_dist.h"
#include <math.h>

double wubu_poincare_dist(const double *u, const double *v, size_t dim, double c) {
    double nu=0,nv=0,diff=0;
    for (size_t i=0;i<dim;i++){ nu+=u[i]*u[i]; nv+=v[i]*v[i]; double d=u[i]-v[i]; diff+=d*d; }
    double denom = (1 - c*nu)*(1 - c*nv);
    if (fabs(denom)<1e-12) denom=1e-12;
    double arg = 1 + 2*c*diff/denom;
    if (arg < 1) arg = 1;
    return acosh(sqrt(arg))/sqrt(c>0?c:1);
}

double wubu_lorentz_dist(const double *u, const double *v, size_t dim) {
    /* u,v are (dim+1)-vectors in Minkowski space; distance = arcosh(-<u,v>_L) */
    double ip = -u[0]*v[0];
    for (size_t i=1;i<=dim;i++) ip += u[i]*v[i];
    double arg = -ip;
    if (arg < 1) arg = 1;
    return acosh(arg);
}