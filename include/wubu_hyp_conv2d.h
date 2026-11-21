/* wubu_hyp_conv2d.h */
#ifndef WUBU_HYP_CONV2D_H
#define WUBU_HYP_CONV2D_H
#include <stddef.h>
/* 2D hyperbolic convolution: apply kernel to spatial feature map (simplified) */
void wubu_hyp_conv2d(const double *in, int H, int W, int C, const double *kernel, int kH, int kW, double *out);
#endif