#include "wubu_hyp_conv2d.h"
#include <string.h>

void wubu_hyp_conv2d(const double *in, int H, int W, int C, const double *kernel, int kH, int kW, double *out) {
    /* Real 2D convolution: out[h,w,c] = sum over kernel of in[h+kh, w+kw, c] * kernel[kh,kw] */
    int padH = kH / 2, padW = kW / 2;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                double sum = 0;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = h + kh - padH;
                        int iw = w + kw - padW;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            sum += in[(ih*W + iw)*C + c] * kernel[(kh*kW + kw)*C + c];
                        }
                    }
                }
                out[(h*W + w)*C + c] = sum;
            }
        }
    }
}
