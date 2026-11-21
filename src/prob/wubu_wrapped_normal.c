#include "wubu_wrapped_normal.h"
#include "wubu_poincare_explog.h"
#include <math.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
void wubu_wrapped_normal_sample(const double *mu, double sigma, double *out, size_t dim, unsigned int *rng) {
    /* Gaussian in tangent space at origin, then exp */
    double *v = malloc(dim*sizeof(double));
    for (size_t i=0;i<dim;i++){
        /* Box-Muller */
        *rng = *rng*1103515245u+12345u; double r1=(double)*rng/0xFFFFFFFFu;
        *rng = *rng*1103515245u+12345u; double r2=(double)*rng/0xFFFFFFFFu;
        v[i] = sigma*sqrt(-2*log(r1+1e-12))*cos(2*M_PI*r2);
    }
    /* exp at origin (origin = 0 in ball): out = tanh(||v||/2)*v/||v|| */
    double nv=0; for (size_t i=0;i<dim;i++) nv+=v[i]*v[i]; nv=sqrt(nv);
    if (nv<1e-12){ for(size_t i=0;i<dim;i++) out[i]=mu[i]; free(v); return; }
    double t = tanh(nv/2.0);
    for (size_t i=0;i<dim;i++) out[i] = mu[i] + t*v[i]/nv;
    /* project to ball */
    double r=0; for(size_t i=0;i<dim;i++) r+=out[i]*out[i]; r=sqrt(r);
    if (r>=0.99){ double s=0.99/r; for(size_t i=0;i<dim;i++) out[i]*=s; }
    free(v);
}