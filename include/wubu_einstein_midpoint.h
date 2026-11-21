/* wubu_einstein_midpoint.h */
#ifndef WUBU_EINSTEIN_MIDPOINT_H
#define WUBU_EINSTEIN_MIDPOINT_H
#include <stddef.h>
/* Einstein midpoint in Lorentz model: weighted average of points (each dim+1 with time component).
   points: npts x (dim+1). weights: npts. out: (dim+1). */
void wubu_einstein_midpoint(const double *points, const double *weights, int npts, size_t dim, double *out);
#endif