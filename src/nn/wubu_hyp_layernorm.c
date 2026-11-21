#include "wubu_hyp_layernorm.h"
#include <math.h>

void wubu_hyp_layernorm(const double *x, double *out, size_t dim, double eps) {
    double mean=0;
    for (size_t i=0;i<dim;i++) mean += x[i];
    mean /= dim;
    double var=0;
    for (size_t i=0;i<dim;i++){ double d=x[i]-mean; var+=d*d; }
    var /= dim;
    double std = sqrt(var+eps);
    for (size_t i=0;i<dim;i++) out[i] = (x[i]-mean)/std;
    /* project to ball */
    double r=0; for(size_t i=0;i<dim;i++) r+=out[i]*out[i]; r=sqrt(r);
    if (r>=0.99){ double s=0.99/r; for(size_t i=0;i<dim;i++) out[i]*=s; }
}