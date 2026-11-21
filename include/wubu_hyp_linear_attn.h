/* wubu_hyp_linear_attn.h */
#ifndef WUBU_HYP_LINEAR_ATTN_H
#define WUBU_HYP_LINEAR_ATTN_H
#include <stddef.h>
/* Hyperbolic linear attention (Hypformer-style): O(n) via kernel trick.
   q,k,v: n x dim. out: n x dim. */
void wubu_hyp_linear_attn(const double *q, const double *k, const double *v, int n, size_t dim, double *out);
#endif