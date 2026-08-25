/* GAP-B001+B007: infrared band */
#ifndef WUBU_INFRARED_H
#define WUBU_INFRARED_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int  wubu_ir_pack(const float* audio,int n_samples,int canvas_w,
                   int ir_rows,uint8_t* canvas);
int  wubu_ir_unpack(const uint8_t* canvas,int canvas_w,int ir_rows,
                     float* audio,int max_samples);
uint32_t wubu_ir_checksum(const uint8_t* canvas,int canvas_w,int ir_rows);
#ifdef __cplusplus
}
#endif
#endif
