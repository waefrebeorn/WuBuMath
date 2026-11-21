/* wubu_rd_quant.h -- rate-distortion quantizer */
#ifndef WUBU_RD_QUANT_H
#define WUBU_RD_QUANT_H
#include <stddef.h>
/* choose quantization index minimizing eta*(w-q)^2 + lambda*rate */
int wubu_rd_quantize(double w, const double *levels, int nlevels,
                     const double *rates, double eta, double lambda);
#endif