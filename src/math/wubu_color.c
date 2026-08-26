/*
 * wubu_color.c -- GROUP 17: Color space pipeline
 *
 * G17.01: BT.601 ↔ BT.709 conversion matrices
 * G17.02: BT.2020 for HDR/WCG
 * G17.06: HDR→SDR tone mapping (Reinhard)
 */
#define M_PI 3.14159265358979f
#include "wubu_color.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Conversion matrices ===== */

typedef struct {
    /* Y = m[0]*R + m[1]*G + m[2]*B */
    double kr,kg,kb;
} ColorMatrix;

static const ColorMatrix cm_601 ={0.299,  0.587,  0.114};
static const ColorMatrix cm_709 ={0.2126, 0.7152, 0.0722};
static const ColorMatrix cm_2020={0.2627, 0.6780, 0.0593};

/* RGB→YUV using specified matrix */
void wubu_rgb_to_yuv_cm(const uint8_t* rgb,uint8_t* y,uint8_t* cb,uint8_t* cr,
                          long n_pixels,wubu_color_std_t std){
    const ColorMatrix* cm;
    switch(std){
        case WUBU_CS_BT601: cm=&cm_601;break;
        case WUBU_CS_BT709: cm=&cm_709;break;
        case WUBU_CS_BT2020:cm=&cm_2020;break;
        default: cm=&cm_601;
    }
    
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i],G=rgb[i+1],B=rgb[i+2];
        double Y=cm->kr*R+cm->kg*G+cm->kb*B;
        y[j]=(uint8_t)(Y<0?0:(Y>255?255:Y));
    }
}

/* ===== BT.709 full conversion (with chroma) ===== */

void wubu_rgb_to_709(const uint8_t* rgb,uint8_t* y,uint8_t* u,uint8_t* v,
                      long n_pixels){
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i]/255.0,G=rgb[i+1]/255.0,B=rgb[i+2]/255.0;
        
        double Y=0.2126*R+0.7152*G+0.0722*B;
        double Cb=-0.1146*R-0.3854*G+0.5*B;
        double Cr=0.5*R-0.4542*G-0.0458*B;
        
        y[j]=(uint8_t)((Y*255.0+0.5));
        u[j]=(uint8_t)((Cb*127.0+128.0+0.5));
        v[j]=(uint8_t)((Cr*127.0+128.0+0.5));
    }
}

/* ===== BT.2020 for HDR ===== */

void wubu_rgb_to_2020(const uint8_t* rgb,uint16_t* y,uint16_t* u,uint16_t* v,
                       long n_pixels){
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i]/255.0,G=rgb[i+1]/255.0,B=rgb[i+2]/255.0;
        
        double Y=0.2627*R+0.6780*G+0.0593*B;
        double Cb=(B-Y)/1.8814;
        double Cr=(R-Y)/1.4746;
        
        /* output as 10-bit for HDR */
        y[j]=(uint16_t)(Y*1023.0+0.5);
        u[j]=(uint16_t)(Cb*511.0+512.0+0.5);
        v[j]=(uint16_t)(Cr*511.0+512.0+0.5);
    }
}

/* ===== G17.06: Reinhard tone mapping ===== */

/* Simple global tone mapping: maps [0,max_luma] to [0,255] with knee curve */
void wubu_tonemap_reinhard(const float* hdr_rgb,uint8_t* sdr_rgb,
                             long n_pixels,float max_luminance){
    for(long i=0;i<n_pixels;i++){
        /* compute luminance */
        float L=0.2126f*hdr_rgb[i*3]+0.7152f*hdr_rgb[i*3+1]+0.0722f*hdr_rgb[i*3+2];
        
        /* Reinhard: L_out = L / (1 + L/L_white) */
        float L_white=max_luminance*0.8f; /* white point at 80% of max */
        float L_mapped=L/(1.0f+L/L_white);
        
        /* normalize to [0,1] then scale to [0,255] */
        float scale=L_mapped/(L>0.001f?L:0.001f); /* ratio preserves color ratios */
        
        for(int c=0;c<3;c++){
            float val=hdr_rgb[i*3+c]*scale;
            if(val>255)val=255;
            if(val<0)val=0;
            sdr_rgb[i*3+c]=(uint8_t)(val+0.5f);
        }
    }
}

/* PQ (SMPTE ST 2084) EOTF approximation for testing */
float wubu_pq_eotf(uint16_t pq_value){
    /* simplified PQ inverse EOTF for luminance in nits */
    double x=(double)pq_value/1023.0;
    /* approximate: L ≈ 10000 * x^4 for quick test purposes */
    return (float)(10000.0*pow(x,4.0));
}
