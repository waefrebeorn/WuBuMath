/* wubu_hyp_attention.h */
#ifndef WUBU_HYP_ATTENTION_H
#define WUBU_HYP_ATTENTION_H
#include <stddef.h>
/* Standard multi-head attention (Euclidean). Simplified: single head scaled dot-product. */
void wubu_hyp_attention(const double *Q, const double *K, const double *V, int n, size_t dim, double *out);
#endif