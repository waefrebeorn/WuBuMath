/* wubu_hyp_softmax.h */
#ifndef WUBU_HYP_SOFTMAX_H
#define WUBU_HYP_SOFTMAX_H
#include <stddef.h>
/* Softmax over array of size n */
void wubu_hyp_softmax(const double *in, int n, double *out);
/* Cross-entropy loss for single target */
double wubu_hyp_cross_entropy(const double *probs, int target, int n);
#endif