/* wubu_qpu.h -- Quaternion Product Unit (CVPR 2020) */
#ifndef WUBU_QPU_H
#define WUBU_QPU_H
#include <stddef.h>
/* Hamilton product of two quaternions (w,x,y,z) */
void wubu_quat_mul(const double *a, const double *b, double *out);
/* Weighted chain of Hamilton products: out = q1 * q2 * ... * qn with weights */
void wubu_qpu_forward(const double *qs, const double *weights, int n, double *out);
#endif