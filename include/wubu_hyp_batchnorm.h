/* wubu_hyp_batchnorm.h */
#ifndef WUBU_HYP_BATCHNORM_H
#define WUBU_HYP_BATCHNORM_H
#include <stddef.h>
/* Hyperbolic batchnorm: normalize features (Euclidean approx) */
void wubu_hyp_batchnorm(const double *in, double *out, int n, size_t dim, double eps);
#endif