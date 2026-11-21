/* wubu_hyp_linear.h */
#ifndef WUBU_HYP_LINEAR_H
#define WUBU_HYP_LINEAR_H
#include <stddef.h>
/* Linear layer: out = W*x + b */
void wubu_hyp_linear(const double *x, const double *W, const double *b, size_t in_dim, size_t out_dim, double *out);
#endif