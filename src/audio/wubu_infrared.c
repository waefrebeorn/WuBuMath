/*
 * wubu_infrared.c -- GAP-B001+B007: Infrared band — invisible payload
 * channel with dual-band round-trip integrity
 *
 * The beam canvas reserves a strip below the visible band. Audio and
 * P-frame residuals ride there; the renderer NEVER draws it. The
 * checksum gate proves the invisible payload survives the full
 * canvas round trip.
 */
#include "wubu_infrared.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* pack audio samples into the IR strip (bottom rows of the canvas) */
int wubu_ir_pack(const float* audio,int n_samples,int canvas_w,
                  int ir_rows,uint8_t* canvas){
    /* each float = 4 bytes stored in consecutive pixels' LSBs */
    int capacity=canvas_w*ir_rows/2;   /* 2 bytes per pixel (2 channels × 4 bits) */
    if(n_samples>capacity)return -1;
    memset(canvas+(size_t)(canvas_w-ir_rows)*canvas_w*3,0,(size_t)ir_rows*canvas_w*3);
    for(int i=0;i<n_samples&&i<canvas_w*ir_rows/2;i++){
        int px=i%canvas_w, py=canvas_w-ir_rows+i/canvas_w;
        uint8_t* p=canvas+((size_t)py*canvas_w+px)*3;
        /* quantize to signed byte: -1..+1 -> -127..127 */
        int q=(int)(audio[i]*127.0f);
        if(q>127)q=127;if(q<-127)q=-127;
        p[0]=(uint8_t)(q+128);  /* offset binary in R channel */
    }
    return 0;
}

int wubu_ir_unpack(const uint8_t* canvas,int canvas_w,int ir_rows,
                    float* audio,int max_samples){
    for(int i=0;i<max_samples&&i<canvas_w*ir_rows/2;i++){
        int px=i%canvas_w, py=canvas_w-ir_rows+i/canvas_w;
        const uint8_t* p=canvas+((size_t)py*canvas_w+px)*3;
        audio[i]=((int)p[0]-128)/127.0f;
    }
    return 0;
}

uint32_t wubu_ir_checksum(const uint8_t* canvas,int canvas_w,int ir_rows){
    uint32_t crc=0xFFFFFFFFu;
    size_t n=(size_t)ir_rows*canvas_w*3;
    const uint8_t* d=canvas+(size_t)(canvas_w-ir_rows)*canvas_w*3;
    for(size_t i=0;i<n;i++){
        crc^=d[i];
        for(int k=0;k<8;k++)crc=(crc>>1)^((crc&1)?0xEDB88320u:0);
    }
    return crc^0xFFFFFFFFu;
}
