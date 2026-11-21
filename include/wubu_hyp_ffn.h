/* wubu_hyp_ffn.h */
#ifndef WUBU_HYP_FFN_H
#define WUBU_HYP_FFN_H
#include <stddef.h>
/* Feed-forward network: x -> W1 -> relu -> W2 */
void wubu_hyp_ffn(const double *x, const double *W1, const double *W2, size_t dim, size_t hidden, double *out);
#endif