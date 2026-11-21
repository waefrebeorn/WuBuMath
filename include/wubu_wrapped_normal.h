/* wubu_wrapped_normal.h */
#ifndef WUBU_WRAPPED_NORMAL_H
#define WUBU_WRAPPED_NORMAL_H
#include <stddef.h>
/* Sample from wrapped normal on Poincare ball:
   v ~ N(0, sigma²I) in tangent space at origin, then exp_0(v). */
void wubu_wrapped_normal_sample(const double *mu, double sigma, double *out, size_t dim, unsigned int *rng);
#endif