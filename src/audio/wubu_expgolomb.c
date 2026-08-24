/*
 * wubu_expgolomb.c -- GAP-E008: Exponential-Golomb bit-level coding
 *
 * The H.264/HEVC standard entropy code for residual magnitudes:
 *   codeword = [L-1 zeros][1][L-1 LSBs of value]
 * where L = floor(log2(v+1)) + 1. Value 0 encodes as single '1'.
 *
 * Bit-level reader/writer with MSB-first packing — the same convention
 * as all ITU codecs, compatible with our beam-canvas bitstream.
 */
#include "wubu_expgolomb.h"
#include <string.h>
#include <math.h>

void wubu_bw_init(WubuBitWriter* bw,uint8_t* buf,size_t cap){
    bw->buf=buf;bw->cap=cap;bw->pos=0;bw->bit=0;
    memset(buf,0,cap);
}
static int bw_put_bit(WubuBitWriter* bw,int b){
    if(bw->pos>=bw->cap)return -1;
    if(b)bw->buf[bw->pos]|=(uint8_t)(0x80>>bw->bit);
    if(++bw->bit==8){bw->bit=0;bw->pos++;}
    return 0;
}
int wubu_eg_put(WubuBitWriter* bw,uint32_t v){
    uint32_t vp=v+1;
    int L=0;uint32_t t=vp;
    while(t){L++;t>>=1;}
    /* L-1 leading zeros */
    for(int i=0;i<L-1;i++)
        if(bw_put_bit(bw,0))return -1;
    /* then binary of vp (L bits, MSB first) */
    for(int i=L-1;i>=0;i--)
        if(bw_put_bit(bw,(vp>>i)&1))return -1;
    return 0;
}

void wubu_br_init(WubuBitReader* br,const uint8_t* buf,size_t len){
    br->buf=buf;br->len=len;br->pos=0;br->bit=0;
}
static int br_get_bit(WubuBitReader* br){
    if(br->pos>=br->len)return -1;
    int b=(br->buf[br->pos]>>(7-br->bit))&1;
    if(++br->bit==8){br->bit=0;br->pos++;}
    return b;
}
int wubu_eg_get(WubuBitReader* br,uint32_t* out){
    int zeros=0;
    while(br_get_bit(br)==0){
        zeros++;
        if(zeros>32)return -1;
    }
    /* we just consumed the leading 1 */
    uint32_t v=1;
    for(int i=0;i<zeros;i++){
        int b=br_get_bit(br);
        if(b<0)return -1;
        v=(v<<1)|(uint32_t)b;
    }
    *out=v-1;
    return 0;
}
