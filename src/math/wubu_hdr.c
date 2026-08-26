/*
 * wubu_hdr.c -- GROUP 24: HDR & Wide Color Gamut
 *
 * G24.01: PQ (SMPTE ST 2084) EOTF/Inverse-EOTF
 * G24.02: HLG (ARIB STD-B67) transfer function
 * G24.03: 10-bit pipeline helpers
 * G24.05: MaxCLL/MaxFALL metadata computation
 * G24.07: HDR→SDR tone mapping curves
 */
#define M_PI 3.14159265358979f
#include "wubu_hdr.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== PQ (SMPTE ST 2084) ===== */

/*
 * PQ EOTF: converts 10/12-bit signal to absolute luminance in nits.
 * Full spec formula:
 *   V = normalized [0,1]
 *   L = 10000 * ((max(V^(1/m2) - c1, 0) / (c2 - c3*V^(1/m2)))^(1/m1))
 * where m1=2610/16384, m2=2523/4096*128, c1=3424/4096, c2=2413/4096*32, c3=2392/4096*32
 */

double wubu_pq_eotf_full(int pq_value,int bit_depth){
    double m1=2610.0/16384.0;
    double m2=2523.0/4096.0*128;
    double c1=3424.0/4096.0;
    double c2=2413.0/4096.0*32;
    double c3=2392.0/4096.0*32;
    
    double max_val=(1<<bit_depth)-1;
    double V=pq_value/max_val;
    
    if(V<=0)return 0;
    
    double Vp=pow(V,1/m2);
    double num=Vp-c1;
    double den=c2-c3*Vp;
    
    if(den<=0)return 10000; /* clamp to max luminance */
    
    return 10000.0*pow(num/den,1/m1);
}

/* inverse: nits → PQ code value */
int wubu_pq_inverse_eotf(double nits,int bit_depth){
    double m1=2610.0/16384.0;
    double m2=2523.0/4096.0*128;
    double c1=3424.0/4096.0;
    double c2=2413.0/4096.0*32;
    double c3=2392.0/4096.0*32;
    
    if(nits<=0)return 0;
    if(nits>=10000)return (1<<bit_depth)-1;
    
    double Y=nits/10000.0;
    double Yp=pow(Y,m1);
    
    double num=c1+c2*Yp;
    double den=1+c3*Yp;
    
    double V=pow(num/den,m2);
    return (int)(V*((1<<bit_depth)-1)+0.5);
}

/* ===== HLG (ARIB STD-B67) ===== */

/*
 * HLG OETF (scene linear → signal):
 *   for x <= 1/12: sqrt(3*x)
 *   for x > 1/12: a*log(12x-b)+c
 * where a=0.17883277, b=0.28466892, c=0.55991073
 */
double wubu_hlg_oetf(double scene_linear){
    const double a=0.17883277,b=0.28466892,c=0.55991073;
    if(scene_linear<=0)return 0;
    
    if(scene_linear<=1.0/12)
        return sqrt(3*scene_linear);
    else
        return a*log(12*scene_linear-b)+c;
}

/* HLG inverse OETF */
double wubu_hlg_inverse(double signal){
    const double a=0.17883277,b=0.28466892,c=0.55991073;
    if(signal<=0)return 0;
    
    if(signal<=0.5)
        return signal*signal/3;
    else
        return (exp((signal-c)/a)+b)/12;
}

/* ===== G24.05: HDR Metadata ===== */

void wubu_hdr_compute_metadata(const uint16_t* luma,long n_pixels,int bit_depth,
                                 int* max_cll,int* max_fall){
    /* convert each pixel to nits using PQ */
    long frame_count=1; /* single frame for now */
    double max_nits=0;
    double max_frame_avg=0;
    
    /* compute per-pixel luminance */
    double sum=0;
    for(long i=0;i<n_pixels;i++){
        double nits=wubu_pq_eotf_full(luma[i],bit_depth);
        if(nits>max_nits)max_nits=nits;
        sum+=nits;
    }
    
    max_frame_avg=sum/n_pixels;
    
    *max_cll=(int)(max_nits+0.5);
    *max_fall=(int)(max_frame_avg+0.5);
}

/* ===== G24.07: HDR→SDR tone mapping ===== */

/* simple knee-based tone mapping curve */
void wubu_hdr_to_sdr_tonemap(const uint16_t* hdr_luma,long n_pixels,
                               int hdr_bit_depth,float hdr_max_nits,
                               uint8_t* sdr_output){
    float sdr_max=203.0f; /* reference white for SDR */
    
    for(long i=0;i<n_pixels;i++){
        double nits=wubu_pq_eotf_full(hdr_luma[i],hdr_bit_depth);
        
        /* knee curve: preserve up to sdr_max, compress above */
        double mapped;
        if(nits<=sdr_max)
            mapped=nits*(255.0/sdr_max); /* linear region */
        else{
            /* soft knee compression above SDR white */
            double ratio=sdr_max/nits;
            double compressed=sdr_max+(nits-sdr_max)*ratio*ratio*0.5;
            mapped=compressed*(255.0/sdr_max);
            if(mapped>255)mapped=255;
        }
        
        sdr_output[i]=(uint8_t)(mapped<0?0:(mapped>255?255:mapped));
    }
}
