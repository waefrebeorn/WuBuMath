/* wubu_ptc.h */
#ifndef WUBU_PTC_H
#define WUBU_PTC_H
#include <stddef.h>
/* Parallel transport convolution: average tangent vectors at x, transported to origin.
   inputs: nvec vectors (dim each) at point x; output averaged vector at origin. */
void wubu_ptc_average(const double *vecs, int nvec, size_t dim, double *out);
#endif