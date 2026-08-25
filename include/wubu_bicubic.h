/* GAP-C055: bicubic interpolation for rotation */
#ifndef WUBU_BICUBIC_H
#define WUBU_BICUBIC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_bc_rotate(const uint8_t* src,uint8_t* dst,
                     int W,int H,float angle);
#ifdef __cplusplus
}
#endif
#endif
