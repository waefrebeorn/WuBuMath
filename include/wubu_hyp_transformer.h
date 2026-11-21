/* wubu_hyp_transformer.h */
#ifndef WUBU_HYP_TRANSFORMER_H
#define WUBU_HYP_TRANSFORMER_H
#include <stddef.h>
/* Hyperbolic transformer block: gated attn + FFN + residual + layernorm */
void wubu_hyp_transformer_block(const double *x, const double *Wg, const double **keys, const double **vals,
    int nkeys, size_t dim, const double *Wffn, double *out);
#endif