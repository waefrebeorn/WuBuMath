/* wubu_so4.h */
#ifndef WUBU_SO4_H
#define WUBU_SO4_H
#include <stddef.h>
/* SO(4) rotation via two quaternions (left and right): x' = qL * x * qR.
   qL, qR: unit quaternions (4 doubles). x: 4D point. out: 4D. */
void wubu_so4_rotate(const double *qL, const double *qR, const double *x, double *out);
#endif