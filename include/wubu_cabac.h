/* GROUP 3-6: CABAC arithmetic coder */
#ifndef WUBU_CABAC_H
#define WUBU_CABAC_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t state;  /* 0..63 probability state index */
    uint8_t mps;    /* most probable symbol (0 or 1) */
} CabacContext;

typedef struct {
    uint8_t* buf;
    size_t cap;
    size_t pos;
    uint32_t low;
    int range;
    int bits_outstanding;
    int first_bit;
} WuBuCabacEnc;

void wubu_cabac_init_encoder(WuBuCabacEnc* e,uint8_t* buf,size_t cap);
void wubu_cabac_encode_bin(WuBuCabacEnc* e,CabacContext* ctx,int bin_val);
void wubu_cabac_encode_bypass(WuBuCabacEnc* e,int bin_val);
void wubu_cabac_encode_terminate(WuBuCabacEnc* e,int bin_val);
void wubu_cabac_init_contexts(CabacContext* contexts,int n_contexts,
                               int init_value,int qp);
#ifdef __cplusplus
}
#endif
#endif
