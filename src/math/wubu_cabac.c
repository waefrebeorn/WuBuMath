/*
 * wubu_cabac.c -- Binary arithmetic coder with context modeling
 * v5 (C1 CLOSED): encoder + decoder, exact round-trip.
 *
 * Formulation: LZMA-proven range coder discipline.
 *   - range kept in [2^24, 2^28); renorm shifts one BYTE at a time
 *   - encoder: 33-bit low; ShiftLow() emits top byte only when no future
 *     carry can alter it; ambiguous 0xFF bytes counted in cache_size and
 *     resolved when the first non-ambiguous byte appears (carry adds 1)
 *   - decoder: 32-bit code window mirrors byte-for-byte
 *   - context bins: H.264-style rLPS=(range>>14)*pstate[state]
 *   - bypass bins: LZMA EncodeDirectBits/DecodeDirectBits style
 */
#include "wubu_cabac.h"
#include <stdlib.h>
#include <string.h>

/* probability state machine */
static const uint8_t cabac_trans_lps[64]={
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9,11,11,12,
   13,13,15,15,16,17,18,18,19,20,22,24,24,25,27,28,
   29,31,32,34,35,37,38,40,42,43,45,46,48,49,51,52,
   53,54,55,56,57,58,59,60,61,62,62,63,63,63
};
static const uint8_t cabac_trans_mps[64]={
    1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,
   17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
   33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
   50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57
};

/* LPS probabilities scaled to [2,2693] representing pLPS*65536/32 */
static const uint16_t cabac_pstate[64]={
    2,   6,   9,  12,  15,  18,  21,  24,
   29,  33,  38,  43,  49,  55,  62,  69,
   77,  86,  95, 105, 116, 128, 140, 153,
  167, 182, 197, 214, 231, 250, 270, 291,
  313, 337, 363, 390, 419, 450, 483, 519,
  557, 597, 641, 687, 737, 791, 849, 911,
  978,1049,1125,1206,1291,1382,1479,1582,
 1692,1809,1934,2067,2209,2360,2521,2693
};

#define RC_TOP (1u<<24)

/* ===== Encoder ===== */

typedef struct {
    uint64_t low;
    uint32_t range;
    uint8_t  cache;
    uint64_t cache_size;
    int      first_byte;
    uint8_t* buf;
    size_t   cap;
    size_t   pos;
} EncCore;

static EncCore g_enc;

static void enc_out(EncCore* c,uint8_t b){
    if(c->pos<c->cap)c->buf[c->pos]=b;
    c->pos++;
}

/* emit top byte of low when no pending carry can change it */
static void enc_shift_low(EncCore* c){
    if((uint32_t)c->low<(uint32_t)0xFF000000u||(int)(c->low>>32)!=0){
        uint8_t carry=(uint8_t)(c->low>>32);
        uint8_t temp=c->cache;
        if(c->first_byte){
            /* nothing before cache to patch */
        }
        do{
            enc_out(c,(uint8_t)(temp+carry));
            temp=0xFF;
        }while(--c->cache_size!=0);
        c->cache=(uint8_t)((uint32_t)c->low>>24);
        c->first_byte=0;
    }
    c->cache_size++;
    c->low=(uint32_t)c->low<<8;
}

void wubu_cabac_init_encoder(WuBuCabacEnc* e,uint8_t* buf,size_t cap){
    memset(e,0,sizeof(WuBuCabacEnc));
    e->buf=buf;e->cap=cap;e->pos=0;
}

void wubu_cabac_start(WuBuCabacEnc* e,uint8_t* buf,size_t cap){
    wubu_cabac_init_encoder(e,buf,cap);
    memset(&g_enc,0,sizeof(g_enc));
    g_enc.buf=buf;g_enc.cap=cap;
    g_enc.low=0;
    g_enc.range=0xFFFFFFFFu;
    g_enc.cache_size=1;
    g_enc.first_byte=1;
    /* No forced shift: the initial cache=0 flushes as the stream's first
       byte when the first real shift fires. Decoder skips exactly 1 byte. */
}

void wubu_cabac_encode_bin(WuBuCabacEnc* e,CabacContext* ctx,int bin_val){
    uint32_t rLPS=(g_enc.range>>14)*(uint32_t)cabac_pstate[ctx->state];
    uint32_t rMPS=g_enc.range-rLPS;
    if(bin_val!=ctx->mps){
        g_enc.low+=rMPS;
        g_enc.range=rLPS?rLPS:1;
        if(ctx->state==0)ctx->mps=!ctx->mps;
        ctx->state=cabac_trans_lps[ctx->state];
    }else{
        g_enc.range=rMPS?rMPS:1;
        ctx->state=cabac_trans_mps[ctx->state];
    }
    while(g_enc.range<RC_TOP){
        enc_shift_low(&g_enc);
        g_enc.range<<=8;
    }
}

void wubu_cabac_encode_bypass(WuBuCabacEnc* e,int bin_val){
    uint32_t half=g_enc.range>>1;
    if(bin_val)g_enc.low+=half;
    g_enc.range=half?half:1;
    while(g_enc.range<RC_TOP){
        enc_shift_low(&g_enc);
        g_enc.range<<=8;
    }
}

long wubu_cabac_finish(WuBuCabacEnc* e){
    /* emit 5 zero-bytes worth of shifts: flushes low fully, carry-safe */
    for(int i=0;i<5;i++)enc_shift_low(&g_enc);
    e->pos=g_enc.pos;
    return (long)e->pos;
}

void wubu_cabac_init_contexts(CabacContext* contexts,int n_contexts,
                               int init_value,int qp){
    for(int i=0;i<n_contexts;i++){
        contexts[i].state=init_value%64;
        contexts[i].mps=0;
    }
}

/* ===== Decoder ===== */

typedef struct {
    uint32_t code;
    uint32_t range;
    const uint8_t* buf;
    size_t cap;
    size_t pos;
} DecCore;

static DecCore g_dec;

static uint8_t dec_in(void){
    uint8_t b=0;
    if(g_dec.pos<g_dec.cap)b=g_dec.buf[g_dec.pos++];
    return b;
}

void wubu_cabac_decoder_start(WuBuCabacDec* d,const uint8_t* buf,size_t bytes){
    memset(d,0,sizeof(*d));
    d->buf=(uint8_t*)buf;d->cap=bytes;d->pos=0;
    memset(&g_dec,0,sizeof(g_dec));
    g_dec.buf=buf;g_dec.cap=bytes;
    g_dec.range=0xFFFFFFFFu;
    g_dec.code=0;
    dec_in();  /* skip first byte: encoder's initial cache always flushes as 0x00 */
    for(int i=0;i<4;i++)
        g_dec.code=(g_dec.code<<8)|dec_in();
}

static void dec_normalize(void){
    while(g_dec.range<RC_TOP){
        g_dec.code=(g_dec.code<<8)|dec_in();
        g_dec.range<<=8;
    }
}

int wubu_cabac_decode_bin(WuBuCabacDec* d,CabacContext* ctx){
    uint32_t rLPS=(g_dec.range>>14)*(uint32_t)cabac_pstate[ctx->state];
    uint32_t rMPS=g_dec.range-rLPS;
    int bin;
    if(g_dec.code<rMPS){
        bin=ctx->mps;
        g_dec.range=rMPS?rMPS:1;
        ctx->state=cabac_trans_mps[ctx->state];
    }else{
        bin=!ctx->mps;
        g_dec.code-=rMPS;
        g_dec.range=rLPS?rLPS:1;
        if(ctx->state==0)ctx->mps=!ctx->mps;
        ctx->state=cabac_trans_lps[ctx->state];
    }
    dec_normalize();
    return bin;
}

int wubu_cabac_decode_bypass(WuBuCabacDec* d){
    uint32_t half=g_dec.range>>1;
    int bin;
    if(g_dec.code>=half){
        bin=1;
        g_dec.code-=half;
    }else{
        bin=0;
    }
    g_dec.range=half?half:1;
    dec_normalize();
    return bin;
}

long wubu_cabac_decoder_finish(WuBuCabacDec* d){
    d->pos=g_dec.pos;
    return (long)d->pos;
}
