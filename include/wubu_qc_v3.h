/* GAP-C068: quaternion codec v3 with subpixel refinement */
#ifndef WUBU_QC_V3_H
#define WUBU_QC_V3_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_cv3_encode(const uint8_t* frames,const float* quats,
                      int n_frames,int D,int n_keys,
                      int W,int H,float angle_step,FILE* out);
#ifdef __cplusplus
}
#endif
#endif
