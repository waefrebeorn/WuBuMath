/* GROUP 7: Variable-size transforms */
#ifndef WUBU_TRANSFORM_H
#define WUBU_TRANSFORM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_tr_forward(const int16_t* input,int16_t* output,int size);
void wubu_tr_inverse(const int16_t* coeff,int16_t* output,int size);
void wubu_tr_skip(const int16_t* input,int16_t* output,int size);
void wubu_tr_quantize_m(const int16_t* coeffs,const uint8_t* qmat,
                          int16_t* output,int size,int qp);
void wubu_tr_dequantize_m(const int16_t* quantized,const uint8_t* qmat,
                            int16_t* output,int size,int qp);
const uint8_t* wubu_tr_get_qmat(int size,int intra);
#ifdef __cplusplus
}
#endif
#endif
