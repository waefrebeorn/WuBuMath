/*
 * wubu_bicubic.c -- GAP-C055: Bicubic interpolation for quaternion
 * rotation reconstruction (upgrade from bilinear's 26.6dB)
 *
 * Research source: PixInsight analysis — Lanczos-3 gives best rotation
 * quality; bicubic is close and cheaper. We implement bicubic (a=-0.5,
 * Catmull-Rom) which should push PSNR from 26.6 toward 35+ dB.
 */
#define M_PI 3.14159265358979f
#include "wubu_bicubic.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Catmull-Rom bicubic kernel (a = -0.5) */
static float bc_kernel(float x){
    x=fabsf(x);
    if(x<=1.0f)
        return 1.5f*x*x*x-2.5f*x*x+1.0f;
    if(x<2.0f)
        return -0.5f*x*x*x+2.5f*x*x-4.0f*x+2.0f;
    return 0;
}

static float bc_sample(const float* img,int W,int H,float x,float y){
    int xi=(int)x,yi=(int)y;
    float fx=x-xi,fy=y-yi;
    float result=0;
    for(int m=-1;m<=2;m++){
        for(int n2=-1;n2<=2;n2++){
            int sy=yi+m,sx=xi+n2;
            if(sy<0||sy>=H||sx<0||sx>=W)continue;
            float wgt=bc_kernel(m-fy)*bc_kernel(n2-fx);
            result+=wgt*img[(size_t)sy*W+sx];
        }
    }
    return result;
}

void wubu_bc_rotate(const uint8_t* src,uint8_t* dst,
                     int W,int H,float angle){
    size_t np=(size_t)W*H;
    /* convert to float per channel */
    float* ch[3];
    for(int c=0;c<3;c++)ch[c]=malloc(np*sizeof(float));
    for(size_t i=0;i<np;i++)
        for(int c=0;c<3;c++)ch[c][i]=src[i*3+c];

    float ca=cosf(-angle),sa=sinf(-angle);
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++){
            float dx=x-W/2.0f,dy=y-H/2.0f;
            float rx=dx*ca-dy*sa+W/2.0f;
            float ry=dx*sa+dy*ca+H/2.0f;
            if(rx>=1&&rx<W-2&&ry>=1&&ry<H-2){
                for(int c=0;c<3;c++)
                    dst[((size_t)y*W+x)*3+c]=
                        (uint8_t)(bc_sample(ch[c],W,H,rx,ry)+0.5f);
            }else{
                int px=(int)fmodf(rx+1000*W,W);
                int py=(int)fmodf(ry+1000*H,H);
                for(int c=0;c<3;c++)
                    dst[((size_t)y*W+x)*3+c]=src[((size_t)py*W+px)*3+c];
            }
        }

    for(int c=0;c<3;c++)free(ch[c]);
}
