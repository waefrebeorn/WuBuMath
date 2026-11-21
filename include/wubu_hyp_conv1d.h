/* wubu_hyp_conv1d.h */
#ifndef WUBU_HYP_CONV1D_H
#define WUBU_HYP_CONV1D_H
#include <stddef.h>
/* 1D convolution: out[t] = sum_k in[t+k]*kernel[k] */
void wubu_hyp_conv1d(const double *in, int n, const double *kernel, int k, double *out);
#endif