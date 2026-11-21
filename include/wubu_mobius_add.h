/* wubu_mobius_add.h */
#ifndef WUBU_MOBIUS_ADD_H
#define WUBU_MOBIUS_ADD_H
#include <stddef.h>
/* Mobius addition in Poincare ball: u ⊕_c v */
void wubu_mobius_add(const double *u, const double *v, double *out, size_t dim, double c);
#endif