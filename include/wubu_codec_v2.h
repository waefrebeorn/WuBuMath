/* GAP-C067: complete quaternion codec v2 */
#ifndef WUBU_CODEC_V2_H
#define WUBU_CODEC_V2_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_cv2_encode(const float* quats,int n_frames,int D,
                      int n_keys,int W,int H,float angle_step,
                      const uint8_t* reference_frame,FILE* out);
#ifdef __cplusplus
}
#endif
#endif
