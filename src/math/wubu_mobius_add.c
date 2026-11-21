#include "wubu_mobius_add.h"
#include <math.h>

void wubu_mobius_add(const double *u, const double *v, double *out, size_t dim, double c) {
    double nu=0,nv=0,uv=0;
    for (size_t i=0;i<dim;i++){ nu+=u[i]*u[i]; nv+=v[i]*v[i]; uv+=u[i]*v[i]; }
    double denom = 1 + 2*c*uv + c*c*nu*nv;
    if (fabs(denom)<1e-12) denom=1e-12;
    for (size_t i=0;i<dim;i++) {
        out[i] = ((1+2*c*uv+c*nv)*u[i] + (1-c*nu)*v[i]) / denom;
    }
}