#include "wubu_mrp.h"
#include <math.h>

void wubu_quat_to_mrp(const double *q, double *out) {
    /* MRP = (x,y,z)/(1+w) */
    double denom = 1.0 + q[0];
    if (fabs(denom) < 1e-12) denom = 1e-12;
    for (int i = 0; i < 3; i++) out[i] = q[i+1] / denom;
}

void wubu_mrp_to_quat(const double *m, double *q) {
    double r2 = m[0]*m[0]+m[1]*m[1]+m[2]*m[2];
    double denom = 1.0 + r2;
    q[0] = (1.0 - r2) / denom;
    for (int i = 0; i < 3; i++) q[i+1] = 2.0 * m[i] / denom;
}