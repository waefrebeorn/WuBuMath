/* wubu_mrp.h */
#ifndef WUBU_MRP_H
#define WUBU_MRP_H
#include <stddef.h>
/* Modified Rodrigues Parameters: stereographic projection S^3 -> R^3.
   q: unit quaternion (w,x,y,z). out: 3-vector. */
void wubu_quat_to_mrp(const double *q, double *out);
void wubu_mrp_to_quat(const double *m, double *q);
#endif