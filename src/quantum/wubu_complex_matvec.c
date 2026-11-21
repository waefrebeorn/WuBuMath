#include "wubu_complex_matvec.h"

void wubu_complex_matvec(const double *g, const double *v, double *out) {
    /* g: re00,im00,re01,im01,re10,im10,re11,im11
       v: re0,im0,re1,im1 */
    double a0 = g[0]*v[0] - g[1]*v[1] + g[2]*v[2] - g[3]*v[3];
    double b0 = g[0]*v[1] + g[1]*v[0] + g[2]*v[3] + g[3]*v[2];
    double a1 = g[4]*v[0] - g[5]*v[1] + g[6]*v[2] - g[7]*v[3];
    double b1 = g[4]*v[1] + g[5]*v[0] + g[6]*v[3] + g[7]*v[2];
    out[0]=a0; out[1]=b0; out[2]=a1; out[3]=b1;
}