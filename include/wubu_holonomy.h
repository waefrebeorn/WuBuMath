/* wubu_holonomy.h */
#ifndef WUBU_HOLONOMY_H
#define WUBU_HOLONOMY_H
#include <stddef.h>
/* Holonomy: parallel transport vector v around a loop (approx: identity for flat, nontrivial for curved).
   Simplified: apply a rotation matrix R (dim x dim) to v. */
void wubu_holonomy_apply(const double *R, const double *v, double *out, size_t dim);
#endif