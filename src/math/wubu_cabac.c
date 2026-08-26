/*
 * wubu_cabac.c -- GROUP 3-6: CABAC arithmetic coder
 * The entropy engine that gives H.264/HEVC its compression advantage.
 *
 * Architecture:
 * - 64-state probability state machine (LPS probability 0.01875 to 0.5)
 * - MPS/LPS distinction per context
 * - Multiplication-free range subdivision via quantized range table
 * - Renormalization when range drops below 2^8
 * - Bypass mode for equiprobable bins
 *
 * This is the single highest-impact module for beating x264 on compression.
 */
#include "wubu_cabac.h"
#include <stdlib.h>
#include <string.h>

/* ===== Probability State Machine Tables ===== */

/* LPS probability table (64 states), from H.264 Table 9-44 */
static const uint16_t cabac_lpstab[64]={
    2,   6,   9,  12,  15,  18,  21,  24,
   29,  33,  38,  43,  49,  55,  62,  69,
   77,  86,  95, 105, 116, 128, 140, 153,
  167, 182, 197, 214, 231, 250, 270, 291,
  313, 337, 363, 390, 419, 450, 483, 519,
  557, 597, 641, 687, 737, 791, 849, 911,
  978,1049,1125,1206,1291,1382,1479,1582,
 1692,1809,1934,2067,2209,2360,2521,2693
};

/* State transition tables from H.264 Table 9-45 */
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

/* Range subdivision: pre-computed products of pLPS × range/4 */
/* We use the simplified approach: rLPS = (range >> 6) * lpstab[state] >> 10ish */
/* Pre-computed rangeLPS table from H.264 spec Table 9-43
 * rangeTabLPS[state][qIdx] where qIdx = quantized range index (0-3) */
static const uint8_t cabac_range_tab[64][4]={
 {128,176,208,240},{128,167,197,227},{128,158,187,216},{123,150,178,205},
 {116,142,169,195},{111,135,160,185},{105,128,152,175},{100,122,144,166},
 { 95,116,137,158},{ 90,110,130,150},{ 86,104,124,143},{ 82, 99,117,135},
 { 78, 94,112,129},{ 74, 90,106,123},{ 71, 85,101,117},{ 67, 81, 96,111},
 { 64, 77, 91,106},{ 61, 73, 87,101},{ 58, 70, 82, 96},{ 55, 67, 79, 91},
 { 53, 63, 75, 87},{ 50, 60, 71, 83},{ 48, 57, 68, 79},{ 46, 55, 64, 75},
 { 44, 52, 62, 72},{ 42, 50, 59, 69},{ 40, 48, 56, 66},{ 38, 46, 54, 63},
 { 37, 44, 52, 61},{ 35, 42, 50, 58},{ 33, 40, 47, 55},{ 32, 38, 45, 53},
 { 30, 36, 43, 51},{ 29, 34, 41, 48},{ 27, 33, 39, 46},{ 26, 31, 37, 44},
 { 25, 30, 36, 42},{ 23, 28, 34, 41},{ 22, 27, 32, 39},{ 21, 26, 31, 37},
 { 20, 24, 30, 36},{ 19, 23, 28, 34},{ 18, 22, 27, 33},{ 17, 21, 26, 31},
 { 16, 20, 25, 30},{ 15, 19, 23, 29},{ 14, 18, 22, 28},{ 14, 17, 21, 27},
 { 13, 16, 20, 26},{ 12, 15, 19, 25},{ 11, 14, 18, 24},{ 11, 13, 17, 23},
 { 10, 13, 16, 22},{ 10, 12, 15, 21},{  9, 11, 15, 20},{  9, 11, 14, 19},
 {  8, 10, 14, 19},{  8,  9, 13, 18},{  7,  9, 12, 17},{  7,  9, 12, 17},
 {  7,  8, 11, 16},{  6,  8, 11, 16},{  6,  7, 10, 15},{  6,  7, 10, 15}
};

static int cabac_rLPS(int range,int state){
    /* normalize range to [256,510] and get qIdx from bits 8-9 */
    int nbits=0;
    int r=range;
    while(r<256){r<<=1;nbits++;}
    int q_idx=(r>>9)&3;
    return cabac_range_tab[state][q_idx]<<nbits;
}

/* ===== Encoder ===== */

void wubu_cabac_init_encoder(WuBuCabacEnc* e,uint8_t* buf,size_t cap){
    memset(e,0,sizeof(WuBuCabacEnc));
    e->buf=buf;e->cap=cap;e->pos=0;
    e->low=0;
    e->range=510;  /* initial range = 510 (H.264 spec) */
    e->bits_outstanding=0;
    e->first_bit=1;
}

/* output a bit with carry propagation */
static void ce_putbit(WuBuCabacEnc* e,int bit){
    /* write into byte at current bit position */
    size_t byte=e->pos/8;
    int shift=7-(int)(e->pos%8);
    if(byte<e->cap&&bit)
        e->buf[byte]|=(1<<shift);
    else if(byte<e->cap)
        e->buf[byte]&=~(1<<shift);
    e->pos++;

    while(e->bits_outstanding>0){
        byte=e->pos/8;
        shift=7-(int)(e->pos%8);
        if(byte<e->cap){
            if(bit)e->buf[byte]&=~(1<<shift);
            else   e->buf[byte]|=(1<<shift);
        }
        e->pos++;
        e->bits_outstanding--;
    }
}

static void ce_renorm(WuBuCabacEnc* e){
    int safety=0;
    while(e->range<256&&safety++<32){
        if(e->low<256){
            /* 0 → output 0 */
            ce_putbit(e,0);
        }else if(e->low>=512){
            e->low-=512;
            ce_putbit(e,1);
        }else{
            e->low-=256;
            e->bits_outstanding++;
        }
        e->range<<=1;
        e->low=(e->low<<1)&0x3FF; /* keep 10-bit precision */
    }
}

/* encode one binary decision using context model */
void wubu_cabac_encode_bin(WuBuCabacEnc* e,CabacContext* ctx,int bin_val){
    int rLPS=cabac_rLPS(e->range,ctx->state);
    e->range-=rLPS;

    if(bin_val!=ctx->mps){
        /* LPS path */
        e->low+=e->range;
        e->range=rLPS;
        if(ctx->state==0)ctx->mps=!ctx->mps;  /* MPS flip at state 0 */
        ctx->state=cabac_trans_lps[ctx->state];
    }else{
        ctx->state=cabac_trans_mps[ctx->state];
    }
    ce_renorm(e);
}

/* encode bypass bin (equiprobable, no context) */
void wubu_cabac_encode_bypass(WuBuCabacEnc* e,int bin_val){
    e->low<<=1;
    if(bin_val)e->low+=e->range;
    ce_renorm(e);
}

/* encode terminate flag (end of slice) */
void wubu_cabac_encode_terminate(WuBuCabacEnc* e,int bin_val){
    e->range-=2;
    if(bin_val){
        e->low+=e->range;
        e->range=2;
        ce_renorm(e);
        /* flush: encode 1 followed by outstanding bits */
        ce_putbit(e,(e->low>>9)&1);
        e->bits_outstanding++;  /* flush all as opposite */
        ce_putbit(e,(e->low>>8)&1);
        /* pad to byte boundary */
        while((e->pos%8)!=0){if(e->pos<e->cap)e->buf[e->pos]=1;e->pos++;}
    }else{
        ce_renorm(e);
    }
}

/* initialize context models for a slice */
void wubu_cabac_init_contexts(CabacContext* contexts,int n_contexts,
                               int init_value,int qp){
    for(int i=0;i<n_contexts;i++){
        int preCtxState=(init_value*abs(qp-26))>>1; /* simplified */
        int slope_idx=(preCtxState>>4)&3;
        int offset=preCtxState&15;

        /* map to state and MPS */
        int val=(slope_idx<<4)|offset;
        if(val<=63){contexts[i].state=val;contexts[i].mps=0;}
        else{contexts[i].state=val%64;contexts[i].mps=1;}
    }
}

/* ===== Decoder ===== */



static uint32_t cd_read_bits(WuBuCabacDec* d,int n){
    uint32_t val=0;
    for(int i=0;i<n;i++){
        val<<=1;
        if(d->pos<d->cap)
            val|=(d->buf[d->pos]>>(7-(d->pos%8)))&1;
        d->pos++;
    }
    return val;
}

void wubu_cabac_init_decoder(WuBuCabacDec* d,const uint8_t* buf,size_t cap){
    d->buf=buf;d->cap=cap;d->pos=0;
    d->range=510;
    /* read first byte directly as offset (H.264 spec: offset = first 9 bits) */
    d->offset=0;
    /* read bits one at a time, skipping leading zeros and the marker '1' */
    int count=0;
    while(d->pos<d->cap*8){
        uint32_t b=cd_read_bits(d,1);
        if(b)break; /* found the '1' marker */
        count++;
        if(count>16)break; /* safety */
    }
    /* read remaining 9 bits of offset */
    d->offset=cd_read_bits(d,9);
}

/* renormalize decoder: shift range left, read new bits into offset */
static void cd_renorm(WuBuCabacDec* d){
    int safety=0;
    while(d->range<256&&safety++<32){
        d->range<<=1;
        d->offset=(d->offset<<1)|cd_read_bits(d,1);
    }
}

int wubu_cabac_decode_bin(WuBuCabacDec* d,CabacContext* ctx){
    int rLPS=cabac_rLPS((int)d->range,ctx->state);
    d->range-=rLPS;

    int bin_val;
    if(d->offset>=d->range){
        /* LPS path */
        bin_val=!ctx->mps;
        d->offset-=d->range;
        d->range=rLPS;
        if(ctx->state==0)ctx->mps=!ctx->mps;
        ctx->state=cabac_trans_lps[ctx->state];
    }else{
        bin_val=ctx->mps;
        ctx->state=cabac_trans_mps[ctx->state];
    }
    cd_renorm(d);
    return bin_val;
}

int wubu_cabac_decode_bypass(WuBuCabacDec* d){
    d->offset<<=1;
    d->offset|=cd_read_bits(d,1);
    
    if(d->offset>=d->range){
        d->offset-=d->range;
        return 1;
    }
    return 0;
}

int wubu_cabac_decode_terminate(WuBuCabacDec* d){
    d->range-=2;
    if(d->offset>=d->range){
        return 1; /* terminate detected */
    }
    cd_renorm(d);
    return 0;
}
