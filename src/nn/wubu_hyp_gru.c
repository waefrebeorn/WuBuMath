#include "wubu_hyp_gru.h"
#include <math.h>
#include <string.h>

static void proj(double *v, size_t dim){ double r=0; for(size_t i=0;i<dim;i++) r+=v[i]*v[i]; r=sqrt(r); if(r>=0.99){double s=0.99/r; for(size_t i=0;i<dim;i++) v[i]*=s;} }

void wubu_hyp_gru_step(const double *x, const double *h, double *hout, size_t dim,
                       const double *Wz, const double *Wr, const double *Wh, const double *bz) {
    /* z = sigmoid(Wz·x + Uz·h + bz) -- simplified: use Wz for both */
    /* r = sigmoid(Wr·x + Ur·h + br) */
    /* candidate = tanh(Wh·(r*h) + Wx·x + bh) */
    /* h' = (1-z)*h + z*candidate */
    /* We'll do a simplified single-matrix version */
    double z=0,r=0;
    for (size_t i=0;i<dim;i++){ z += Wz[i]*x[i]; r += Wr[i]*x[i]; }
    z = 1.0/(1.0+exp(-z)); r = 1.0/(1.0+exp(-r));
    for (size_t i=0;i<dim;i++){
        double cand = tanh(Wh[i]* (r*h[i]) + bz[i] + x[i]);
        hout[i] = (1-z)*h[i] + z*cand;
    }
    proj(hout, dim);
}