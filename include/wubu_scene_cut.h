/* GAP-C051: scene-cut detection + adaptive KEY insertion */
#ifndef WUBU_SCENE_CUT_H
#define WUBU_SCENE_CUT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_sc_classify(const float* quat_frames,int n_frames,int D,
                       float cut_thresh,float skip_thresh,
                       uint8_t* out_type);
void wubu_sc_stats(const uint8_t* types,int n_frames,
                    int* n_key,int* n_inter,int* n_skip);
long wubu_sc_estimate_bytes(int n_key,long key_bytes,
                             int n_inter,long inter_bytes,int n_skip);
#ifdef __cplusplus
}
#endif
#endif
