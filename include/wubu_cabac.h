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

typedef struct {
    const uint8_t* buf;
    size_t cap;
    size_t pos;
    uint32_t range;
    uint32_t offset;
    uint32_t value;
    int bits_read;
} WuBuCabacDec;

void wubu_cabac_init_decoder(WuBuCabacDec* d,const uint8_t* buf,size_t cap);
int  wubu_cabac_decode_bin(WuBuCabacDec* d,CabacContext* ctx);
int  wubu_cabac_decode_bypass(WuBuCabacDec* d);
int  wubu_cabac_decode_terminate(WuBuCabacDec* d);
#ifdef __cplusplus
}
#endif
#endif
