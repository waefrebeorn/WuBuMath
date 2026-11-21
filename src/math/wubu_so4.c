#include "wubu_so4.h"
#include <math.h>

static void qmul(const double*a,const double*b,double*o){
    o[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    o[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    o[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    o[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}
static void qconj(const double*q,double*o){ o[0]=q[0]; o[1]=-q[1]; o[2]=-q[2]; o[3]=-q[3]; }

void wubu_so4_rotate(const double *qL, const double *qR, const double *x, double *out) {
    /* x as quaternion (0,x0,x1,x2) -> rotate -> extract vector part */
    double qx[4]={0,x[0],x[1],x[2]};
    double tmp[4], res[4], qc[4];
    qconj(qR, qc);
    qmul(qL, qx, tmp);
    qmul(tmp, qc, res);
    /* vector part */
    out[0]=res[1]; out[1]=res[2]; out[2]=res[3];
    /* 4th component for 4D: use res[0] as scalar (w) */
    if (/* 4D */ 1) out[3]=res[0];
}