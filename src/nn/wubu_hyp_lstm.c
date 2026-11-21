#include "wubu_hyp_lstm.h"
#include <math.h>
#include <string.h>

static void proj(double *v, size_t dim){ double r=0; for(size_t i=0;i<dim;i++) r+=v[i]*v[i]; r=sqrt(r); if(r>=0.99){double s=0.99/r; for(size_t i=0;i<dim;i++) v[i]*=s;} }

void wubu_hyp_lstm_step(const double *x, const double *h, const double *c, double *hout, double *cout, size_t dim) {
    for (size_t i=0;i<dim;i++){
        double f = 1.0/(1.0+exp(-(x[i]+h[i])));       /* forget */
        double ig = 1.0/(1.0+exp(-(x[i]-h[i])));       /* input */
        double o = 1.0/(1.0+exp(-(x[i]*0.5+h[i]*0.5)));/* output */
        double g = tanh(x[i]+h[i]);                     /* candidate */
        cout[i] = f*c[i] + ig*g;
        hout[i] = o*tanh(cout[i]);
    }
    proj(hout, dim);
    proj(cout, dim);
}