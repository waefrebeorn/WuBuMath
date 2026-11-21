/* wubu_complex_matvec.h */
#ifndef WUBU_COMPLEX_MATVEC_H
#define WUBU_COMPLEX_MATVEC_H
#include <stddef.h>
/* Apply complex 2x2 gate (8 doubles) to complex 2-vector (4 doubles). */
void wubu_complex_matvec(const double *g, const double *v, double *out);
#endif