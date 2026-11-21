#include "wubu_klein_model.h"
#include <math.h>
double wubu_klein_dist(const double *u, const double *v, size_t dim) {
    double uv=0, nu=0, nv=0;
    for (size_t i=0;i<dim;i++){ uv+=u[i]*v[i]; nu+=u[i]*u[i]; nv+=v[i]*v[i]; }
    double denom = sqrt((1-nu)*(1-nv));
    if (fabs(denom)<1e-12) denom=1e-12;
    double arg = (1-uv)/denom;
    if (arg<1) arg=1;
    return acosh(arg);
}