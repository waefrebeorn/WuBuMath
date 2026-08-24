/*
 * wubu_delta.c -- GAP-E007: Delta + zigzag variable-length encoding
 *
 * The entropy-coding backbone for the beam canvas's index streams:
 *   1. delta: store differences between consecutive values (smaller ints)
 *   2. zigzag: map signed→unsigned so small |deltas| get short codes
 *   3. varint: variable-length bytes — 7 bits per byte, high bit = continue
 *
 * This is the same pipeline as Protobuf/varint + WebP lossless integer
 * coding. Round-trip exactness is the gate.
 */
#include "wubu_delta.h"
#include <stdlib.h>
#include <string.h>

/* zigzag: 0->0, -1->1, 1->2, -2->3, 2->4 ... */
uint32_t wubu_zigzag_encode(int32_t v){
    return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}
int32_t wubu_zigzag_decode(uint32_t v){
    return (int32_t)(v >> 1) ^ -(int32_t)(v & 1);
}

/* delta transform in-place */
void wubu_delta_encode(int32_t* vals,int n){
    for(int i=n-1;i>0;i--)vals[i]-=vals[i-1];
}
void wubu_delta_decode(int32_t* vals,int n){
    for(int i=1;i<n;i++)vals[i]+=vals[i-1];
}

/* varint encode: returns bytes written */
size_t wubu_varint_put(uint8_t* buf,uint32_t v){
    size_t n=0;
    while(v>=0x80){
        buf[n++]=(uint8_t)(v|0x80);
        v>>=7;
    }
    buf[n++]=(uint8_t)v;
    return n;
}

/* varint decode: returns bytes read; sets *out */
size_t wubu_varint_get(const uint8_t* buf,size_t max,uint32_t* out){
    uint32_t v=0;int shift=0;size_t n=0;
    while(n<max){
        uint8_t b=buf[n++];
        v|=(uint32_t)(b&0x7F)<<shift;
        if(!(b&0x80))break;
        shift+=7;
    }
    *out=v;
    return n;
}

/* full pipeline: deltas → zigzag → varint stream.
 * Returns total bytes written to buf. */
size_t wubu_delta_compress(const int32_t* src,int n,uint8_t* buf,size_t buf_cap){
    if(n<=0)return 0;
    int32_t* tmp=malloc(sizeof(int32_t)*(size_t)n);
    memcpy(tmp,src,sizeof(int32_t)*(size_t)n);
    wubu_delta_encode(tmp,n);
    size_t off=0;
    /* first value raw varint (it's the base) */
    for(int i=0;i<n;i++){
        uint32_t zz=wubu_zigzag_encode(tmp[i]);
        if(off+5>buf_cap){free(tmp);return 0;}  /* overflow guard */
        off+=wubu_varint_put(buf+off,zz);
    }
    free(tmp);
    return off;
}

/* inverse. Returns 0 on success, -1 on malformed input. */
int wubu_delta_decompress(const uint8_t* buf,size_t buf_len,
                           int32_t* dst,int n){
    size_t off=0;
    int32_t prev=0;
    for(int i=0;i<n;i++){
        uint32_t zz;
        size_t used=wubu_varint_get(buf+off,buf_len-off,&zz);
        if(used==0)return -1;
        off+=used;
        int32_t d=wubu_zigzag_decode(zz);
        prev=(i==0)?d:prev+d;
        dst[i]=prev;
    }
    return 0;
}
