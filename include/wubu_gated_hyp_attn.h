/* wubu_gated_hyp_attn.h */
#ifndef WUBU_GATED_HYP_ATTN_H
#define WUBU_GATED_HYP_ATTN_H
#include <stddef.h>
/* Gated hyperbolic attention: gate = sigmoid(Wg·x); attn = hyp_attn; out = gate*attn + (1-gate)*x */
void wubu_gated_hyp_attn(const double *x, const double *Wg, const double **keys, const double **vals, int nkeys, size_t dim, double *out);
#endif