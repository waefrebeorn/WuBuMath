/* wubu_hyp_vq.h */
#ifndef WUBU_HYP_VQ_H
#define WUBU_HYP_VQ_H
#include <stddef.h>
/* Vector quantization on Poincare ball: find nearest codebook entry */
int wubu_hyp_vq(const double *x, const double *codebook, int ncode, size_t dim);
#endif