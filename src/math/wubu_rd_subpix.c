/*
 * wubu_rd_subpix.c -- GAP-C054: Bilinear subpixel rotation for quaternion
 * reconstruction — closes the quality gap in the RD curve
 *
 * The C053 RD curve showed quaternion at 11.3dB vs Euclidean at 50.8dB.
 * The gap: nearest-neighbor rotation sampling. Bilinear interpolation
 * should push PSNR to 35+ dB while keeping the byte advantage.
 */
#define M_PI 3.14159265358979f
#include "wubu_rd_subpix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* bilinear sample from a float image */
static float sp_bilinear(const float* img,int W,int H,float x,float y){
    int x0=(int)x,y0=(int)y;
    float fx=x-x0,fy=y-y0;
    if(x0<0||y0<0||x0>=W-1||y0>=H-1)return 0;
    return img[y0*W+x0]*(1-fx)*(1-fy)
          +img[y0*W+x0+1]*fx*(1-fy)
          +img[(y0+1)*W+x0]*(1-fx)*fy
          +img[(y0+1)*W+x0+1]*fx*fy;
}

void wubu_sp_rotate(const uint8_t* src,uint8_t* dst,
                     int W,int H,float angle){
    /* convert to float per channel */
    size_t np=(size_t)W*H;
    float* ch[3];
    for(int c=0;c<3;c++)ch[c]=malloc(np*sizeof(float));
    float* out=malloc(np*sizeof(float));

    for(size_t i=0;i<np;i++)
        for(int c=0;c<3;c++)ch[c][i]=src[i*3+c];

    /* inverse rotate: for each dest pixel, sample from src */
    float ca=cosf(-angle),sa=sinf(-angle);
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++){
            float dx=x-W/2.0f,dy=y-H/2.0f;
            float rx=dx*ca-dy*sa+W/2.0f;
            float ry=dx*sa+dy*ca+H/2.0f;
            if(rx>=0&&rx<W-1&&ry>=0&&ry<H-1)
                for(int c=0;c<3;c++)
                    dst[((size_t)y*W+x)*3+c]=
                        (uint8_t)sp_bilinear(ch[c],W,H,rx,ry);
            else{
                int px=(int)(rx+100*W)%W;
                int py=(int)(ry+100*H)%H;
                for(int c2=0;c2<3;c2++)
                    dst[((size_t)y*W+x)*3+c2]=src[((size_t)py*W+px)*3+c2];
            }
        }

    for(int c=0;c<3;c++){free(ch[c]);}
    free(out);
}
