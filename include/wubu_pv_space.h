/* wubu_pv_space.h */
#ifndef WUBU_PV_SPACE_H
#define WUBU_PV_SPACE_H
#include <stddef.h>
/* Proper Velocity: unconstrained hyperbolic via relativistic velocity addition.
   pv_add: u ⊕ v in PV space (Einstein addition). */
void wubu_pv_add(const double *u, const double *v, double *out, size_t dim);
#endif