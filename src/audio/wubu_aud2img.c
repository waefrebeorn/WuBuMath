#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
/*
 * wubu_aud2img.c -- GAP-E003: Audio→image reversible codec port
 *
 * The STFT spectrogram IS the image: magnitude + phase stored as
 * pixel channels. ISTFT reconstructs the audio. Round-trip correlation
 * is the gate — this is the audio sideband of the beam canvas.
 */
#include "wubu_aud2img.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* pack audio into an image via STFT: magnitudes → R, phases → G/B */
int wubu_ai_encode(const float* audio,int n_samples,int fft_size,
                    int hop,int img_w,int img_h,uint8_t* img){
    int n_freq=fft_size/2+1;
    int n_frames=(n_samples-hop)/(hop>0?hop:1)+1;
    if(n_frames>img_w)n_frames=img_w;
    if(n_freq>img_h)n_freq=img_h;

    /* windowed FFT (naive DFT for small sizes — correctness over speed) */
    float* window=malloc(sizeof(float)*(size_t)fft_size);
    for(int i=0;i<fft_size;i++)
        window[i]=0.5f*(1-cosf(2*3.14159265f*i/(fft_size-1)));  /* Hann */

    for(int fr=0;fr<n_frames;fr++){
        int offset=fr*hop;
        for(int k=0;k<n_freq&&k<img_h;k++){
            /* DFT bin k */
            float re=0,im=0;
            for(int i=0;i<fft_size;i++){
                int si=offset+i;
                if(si>=n_samples)break;
                float s=audio[si]*window[i];
                re+=s*cosf(2*3.14159265f*k*i/fft_size);
                im-=s*sinf(2*3.14159265f*k*i/fft_size);
            }
            float mag=sqrtf(re*re+im*im)/fft_size;
            float phase=atan2f(im,re);
            /* map to [0,255]: mag log-scaled, phase angle-scaled */
            int mi=(int)(log1pf(mag*100)/log1pf(100)*255);
            int pi=(int)((phase+M_PI)/(2*M_PI)*255);
            if(mi<0)mi=0;if(mi>255)mi=255;
            if(pi<0)pi=0;if(pi>255)pi=255;
            size_t idx=((size_t)k*img_w+fr)*3;
            img[idx]=(uint8_t)mi;
            img[idx+1]=(uint8_t)pi;
            img[idx+2]=0;
        }
    }
    free(window);
    return n_frames;
}

/* unpack: image → resynthesized audio via inverse DFT + overlap-add */
float wubu_ai_decode(const uint8_t* img,int img_w,int img_h,
                      int fft_size,int hop,float* audio,int max_samples){
    int n_freq=fft_size/2+1;
    memset(audio,0,sizeof(float)*(size_t)max_samples);

    for(int fr=0;fr<img_w;fr++){
        for(int k=1;k<n_freq&&k<img_h;k++){
            size_t idx=((size_t)k*img_w+fr)*3;
            float mag=(float)img[idx]/255*0.5f;
            float phase=(float)img[idx+1]/255*2*M_PI-M_PI;
            float re=mag*cosf(phase),im=mag*sinf(phase);
            for(int i=0;i<fft_size;i++){
                int si=fr*hop+i;
                if(si>=max_samples||si<0)continue;
                audio[si]+=re*cosf(2*3.14159265f*k*i/fft_size)
                          -im*sinf(2*3.14159265f*k*i/fft_size);
            }
        }
    }
    /* normalize by 1/(n_freq) to account for DFT magnitude scaling */
    float inv_n=1.0f/(float)(fft_size/2);
    for(int i=0;i<max_samples;i++)audio[i]*=inv_n;
    return 0;
}

/* correlation between original and reconstructed audio */
float wubu_ai_correlation(const float* a,const float* b,int n){
    /* clamp inputs to prevent overflow */
    double sa=0,sb=0,saa=0,sbb=0,sab=0;
    int valid=0;
    for(int i=0;i<n;i++){
        if(isnan(a[i])||isnan(b[i])||isinf(a[i])||isinf(b[i]))continue;
        float av=a[i],bv=b[i];
        if(av>100)av=100;if(av<-100)av=-100;
        if(bv>100)bv=100;if(bv<-100)bv=-100;
        sa+=av;sb+=bv;
        saa+=av*av;sbb+=bv*bv;
        sab+=av*bv;
        valid++;
    }
    if(valid<2)return 0;
    double nn=valid;
    double cov=sab-sa*sb/nn;
    double va=saa-sa*sa/nn,vb=sbb-sb*sb/nn;
    if(va<=1e-10||vb<=1e-10)return 0;
    float r=(float)(cov/sqrt(va*vb));
    if(isnan(r))return 0;
    if(r>1)r=1;if(r<-1)r=-1;
    return r;
}
