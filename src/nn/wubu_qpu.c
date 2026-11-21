#include "wubu_qpu.h"
#include <math.h>

void wubu_quat_mul(const double *a, const double *b, double *out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

void wubu_qpu_forward(const double *qs, const double *weights, int n, double *out) {
    /* out starts as identity */
    out[0]=1; out[1]=0; out[2]=0; out[3]=0;
    double tmp[4];
    for (int i=0;i<n;i++){
        /* scale quaternion by weight (scalar part) */
        double q[4] = { qs[i*4]*weights[i], qs[i*4+1]*weights[i],
                        qs[i*4+2]*weights[i], qs[i*4+3]*weights[i] };
        wubu_quat_mul(out, q, tmp);
        for(int j=0;j<4;j++) out[j]=tmp[j];
    }
    /* normalize */
    double nrm = 0; for(int j=0;j<4;j++) nrm += out[j]*out[j];
    nrm = (nrm>0)? 1.0/sqrt(nrm) : 1.0;
    for(int j=0;j<4;j++) out[j]*=nrm;
}
