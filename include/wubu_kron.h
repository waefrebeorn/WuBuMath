/* wubu_kron.h */
#ifndef WUBU_KRON_H
#define WUBU_KRON_H
#include <stddef.h>
/* Kronecker product of two complex 2x2 (8 doubles each) -> 4x4 (32 doubles) */
void wubu_kron2(const double *A, const double *B, double *out);
#endif