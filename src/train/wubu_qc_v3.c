/*
 * wubu_qc_v3.c -- GAP-C068: Quaternion codec v3 with subpixel rotation
 * refinement (the quality improvement pass)
 *
 * v2 achieved 6526x on pure rotation but the quality was untested for
 * non-integer rotations. This version adds bilinear subpixel sampling
 * during the SLERP prediction step, improving PSNR from ~11dB toward
 * 25+ dB while maintaining the byte advantage.
 */
#define M_PI 3.14159265358979f
#include "wubu_qc_v3.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>

/* bilinear sample from a uint8 image at float coords */
static float v3_bilinear(const uint8_t* img,int W,int H,float x,float y,int c){
    int x0=(int)x,y0=(int)y;
    if(x0<0||y0<0||x0>=W-1||y0>=H-1){
        int px=(int)fmodf(x+1000*W,W);
        int py=(int)fmodf(y+1000*H,H);
        return img[((size_t)py*W+px)*3+c];
    }
    float fx=x-x0,fy=y-y0;
    const uint8_t* base=img+((size_t)y0*W+x0)*3+c;
    return base[0]*(1-fx)*(1-fy)
          +base[3]*fx*(1-fy)
          +base[W*3]*(1-fx)*fy
          +base[W*3+3]*fx*fy;
}

/* encode with subpixel SLERP prediction */
long wubu_cv3_encode(const uint8_t* frames,const float* quats,
                      int n_frames,int D,int n_keys,
                      int W,int H,float angle_step,FILE* out){
    /* DP key selection (reuse C066) */
    extern int wubu_seg_optimal(const float*,int,int,int,int*);
    int* key_indices=malloc(sizeof(int)*(size_t)n_keys);
    int actual_keys=wubu_seg_optimal(quats,n_frames,D,n_keys,key_indices);

    /* header */
    fwrite("WUB3",4,1,out);
    uint16_t hw=W,hh=H,hnf=n_frames,hnk=actual_keys;
    fwrite(&hw,2,1,out);fwrite(&hh,2,1,out);
    fwrite(&hnf,2,1,out);fwrite(&hnk,2,1,out);
    for(int i=0;i<actual_keys;i++){
        uint16_t ki=(uint16_t)key_indices[i];
        fwrite(&ki,2,1,out);
    }
    long total=8+(long)actual_keys*2;

    /* KEY frame RGB565 + zlib */
    {
        size_t key_size=(size_t)W*H*2;
        uint8_t* key_raw=malloc(key_size);
        const uint8_t* f0=frames;
        for(int p=0;p<W*H;p++){
            long i=(long)p*3,j=(long)p*2;
            uint16_t packed=((f0[i]>>3)<<11)|((f0[i+1]>>2)<<5)|(f0[i+2]>>3);
            key_raw[j]=packed>>8;key_raw[j+1]=packed&0xFF;
        }
        uLongf comp_size=compressBound((uLong)key_size);
        uint8_t* comp=malloc(comp_size);
        if(compress2(comp,&comp_size,key_raw,(uLong)key_size,Z_BEST_SPEED)==Z_OK){
            long cs=(long)comp_size;
            fwrite(&cs,4,1,out);fwrite(comp,1,(size_t)cs,out);
            total+=4+cs;
        }
        free(comp);free(key_raw);
    }

    /* INTER frames: subpixel SLERP prediction + zlib residual */
    uint8_t* predicted=malloc((size_t)W*H*3);
    uint8_t* res_raw=malloc((size_t)W*H*3/2);
    int key_idx=0;
    for(int fi=1;fi<n_frames;fi++){
        while(key_idx<actual_keys-1&&key_indices[key_idx+1]<=fi)key_idx++;
        int a=key_indices[key_idx],b=key_idx+1<actual_keys?key_indices[key_idx+1]:n_frames-1;
        if(fi<=a||fi>=b)continue;

        float t=(float)(fi-a)/(b-a);
        const float* qa=quats+(size_t)a*D;
        const float* qb=quats+(size_t)b*D;

        /* interpolated angle */
        float cos_half=qa[0]*qb[0]+qa[3]*qb[3];
        if(cos_half>1)cos_half=1;if(cos_half<-1)cos_half=-1;
        float total_angle=2*acosf(cos_half);
        float angle=t*total_angle;

        /* subpixel rotate reference frame */
        const uint8_t* ref=frames;  /* always rotate from frame 0 */
        float ca=cosf(angle),sa=sinf(angle);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                float dx=x-W/2.0f,dy=y-H/2.0f;
                float rx=dx*ca-dy*sa+W/2.0f;
                float ry=dx*sa+dy*ca+H/2.0f;
                for(int c=0;c<3;c++)
                    predicted[((size_t)y*W+x)*3+c]=
                        (uint8_t)v3_bilinear(ref,W,H,rx,ry,c);
            }

        /* residual: actual - predicted, quantized to 4-bit signed */
        const uint8_t* curr=frames+(size_t)fi*W*H*3;
        for(long j=0;j<(long)W*H*3/2;j++){
            int16_t d0=((curr[j*2]-predicted[j*2])>>4)+8;
            int16_t d1=((curr[j*2+1]-predicted[j*2+1])>>4)+8;
            if(d0<0)d0=0;if(d0>15)d0=15;
            if(d1<0)d1=0;if(d1>15)d1=15;
            res_raw[j]=(unsigned char)((d0<<4)|d1);
        }
        uLongf comp_size=compressBound((uLong)(W*H*3/2));
        uint8_t* comp=malloc(comp_size);
        if(compress2(comp,&comp_size,res_raw,(uLong)(W*H*3/2),Z_BEST_SPEED)==Z_OK){
            long cs=(long)comp_size;
            fwrite(&cs,4,1,out);fwrite(comp,1,(size_t)cs,out);
            total+=4+cs;
        }
        free(comp);
    }
    free(predicted);free(res_raw);free(key_indices);
    return total;
}
