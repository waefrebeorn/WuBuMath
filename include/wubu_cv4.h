/* GAP-C069: codec v4 — proper bit-packing + subpixel SLERP */
#ifndef WUBU_CV4_H
#define WUBU_CV4_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_cv4_encode(const uint8_t* frames,const float* quats,
                      int n_frames,int D,int W,int H,float angle_step,
                      int quality_bits,FILE* out);
void wubu_cv4_decode(FILE* in,uint8_t* frames_out,
                      int n_frames,int W,int H,float angle_step,
                      int quality_bits);
#ifdef __cplusplus
}
#endif
#endif
