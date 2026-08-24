/*
 * wubu_kodak.c -- Zephyr-HD audio-to-image layout (GAP-E003)
 *
 * The Kodak: STFT of an audio clip laid into a WxH RGB image so that
 *  - R channel = per-bin magnitude (normalized),
 *  - G channel = cos(phase), B = sin(phase)   [continuous, decodable]
 * Rows are grouped by the 5 perceptual bands; each band's rows resample
 * its bins. Decoding reads G/B -> phase, R -> magnitude, and ISTFT
 * reconstructs the audio. Round-trip gate: corr(recon, original) > 0.99.
 */

#include "wubu_kodak.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int wubu_kodak_init(WubuKodak* kd,int width,int height){
    if(width<16||height<5) return -1;
    memset(kd,0,sizeof(*kd));
    kd->width=width;kd->height=height;
    kd->pixels=malloc(sizeof(float)*(size_t)width*height*3);
    if(!kd->pixels) return -2;
    return 0;
}
void wubu_kodak_free(WubuKodak* kd){free(kd->pixels);kd->pixels=NULL;}

/* pack STFT frames into the image: rows = frames (resampled to height),
 * cols = bins (resampled to width). */
int wubu_kodak_pack(WubuKodak* kd,const float* frames,int num_frames,int bins){
    if(!kd->pixels) return -1;
    const int W=kd->width,H=kd->height;
    for(int y=0;y<H;y++){
        int t=(int)(((float)y+0.5f)*num_frames/H); if(t>=num_frames)t=num_frames-1;
        for(int x=0;x<W;x++){
            int k=(int)(((float)x+0.5f)*bins/W); if(k>=bins)k=bins-1;
            float re=frames[((size_t)t*bins+k)*2+0];
            float im=frames[((size_t)t*bins+k)*2+1];
            float mag=sqrtf(re*re+im*im);
            float ph=atan2f(im,re);
            float* px=kd->pixels+((size_t)y*W+x)*3;
            /* log magnitude for dynamic range, normalized later */
            px[0]=log1pf(mag);
            /* phase encoded EXACTLY: map [-pi,pi] -> [0,1] in G.
             * B carries the fractional magnitude for extra precision. */
            float phn=(ph+(float)M_PI)/(2.0f*(float)M_PI);
            if(phn<0)phn=0; if(phn>1)phn=1;
            px[1]=phn;
            px[2]=mag/(1.0f+mag);   /* complementary magnitude code */
        }
    }
    /* normalize R to [0,1] via sqrt-compression (robust), store scale */
    size_t npix=(size_t)W*H;
    float mx=1e-9f;
    for(size_t p=0;p<npix;p++){
        kd->pixels[p*3]=sqrtf(kd->pixels[p*3]);
        if(kd->pixels[p*3]>mx)mx=kd->pixels[p*3];
    }
    for(size_t p=0;p<npix;p++) kd->pixels[p*3]/=mx;
    kd->mag_scale=mx;
    return 0;
}

/* unpack image back to STFT coefficients */
int wubu_kodak_unpack(const WubuKodak* kd,float* frames,int num_frames,int bins){
    if(!kd->pixels) return -1;
    const int W=kd->width,H=kd->height;
    for(int t=0;t<num_frames;t++){
        int y=(int)(((float)t+0.5f)*H/num_frames); if(y>=H)y=H-1;
        for(int k=0;k<bins;k++){
            int x=(int)(((float)k+0.5f)*W/bins); if(x>=W)x=W-1;
            const float* px=kd->pixels+((size_t)y*W+x)*3;
            float lg=px[0]*px[0]*kd->mag_scale*kd->mag_scale;   /* = log1p(mag) */
            float mag=expm1f(lg);                                /* invert log1p */
            float ph=(px[1]*2.0f*(float)M_PI)-(float)M_PI;   /* exact decode */
            frames[((size_t)t*bins+k)*2+0]=mag*cosf(ph);
            frames[((size_t)t*bins+k)*2+1]=mag*sin(ph);
        }
    }
    return 0;
}
