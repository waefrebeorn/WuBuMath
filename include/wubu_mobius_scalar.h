/* wubu_mobius_scalar.h */
#ifndef WUBU_MOBIUS_SCALAR_H
#define WUBU_MOBIUS_SCALAR_H
#include <stddef.h>
/* Mobius scalar multiplication: r ⊗_c x */
void wubu_mobius_scalar(double r, const double *x, double *out, size_t dim, double c);
#endif