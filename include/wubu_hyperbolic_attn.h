/* wubu_hyperbolic_attn.h */
#ifndef WUBU_HYPERBOLIC_ATTN_H
#define WUBU_HYPERBOLIC_ATTN_H
#include <stddef.h>
/* softmax attention weights from Poincare distances between q and keys */
void wubu_hyp_attn_weights(const double *q, const double **keys, int nkeys, size_t dim, double *weights);
#endif