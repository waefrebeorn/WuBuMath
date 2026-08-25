/*
 * wubu_deblock.c -- GAP-C073: Deblocking filter for DCT-based codec
 * (removes blocking artifacts at 8x8 block boundaries)
 *
 * After quantized DCT reconstruction, block edges show discontinuities
 * ("blocking artifacts"). This adaptive filter smooths across boundaries
 * when the difference is small (artifact) but preserves real edges.
 *
 * Simplified H.264-style: for each vertical/horizontal 8x8 boundary,
 * measure the difference between pixels on either side. If below a
 * threshold, blend them. If above, leave alone (it's an edge).
 */
#include "wubu_deblock.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* apply deblocking to one channel of a frame */
void wubu_db_filter(uint8_t* img,int W,int H,int quality){
    /* threshold scales with quality (higher quality = gentler filtering) */
    int alpha=(quality<3)?24:(quality<6)?16:(quality<9)?8:4;
    int beta=alpha/2;

    /* vertical boundaries (between columns x=7,15,23,...) */
    for(int by=0;by<H;by+=8){
        for(int bx=8;bx<W;bx+=8){
            for(int r=0;r<8;r++){
                int y=by+r;
                if(y>=H)break;
                /* pixels on either side of boundary */
                int p2=img[((size_t)y*W+bx-2)*3];
                int p1=img[((size_t)y*W+bx-1)*3];
                int q1=img[((size_t)y*W+bx)*3];
                int q2=img[((size_t)y*W+bx+1<W?bx+1:W-1)*3];

                int d=p1-q1;
                if(abs(d)>=alpha)continue; /* real edge, don't touch */

                /* smoothness check on both sides */
                if(abs(p2-p1)<beta&&abs(q2-q1)<beta){
                    /* blend */
                    int delta=(d+2)>>2;
                    int v1=p1-delta,v2=q1+delta;
                    img[((size_t)y*W+bx-1)*3]=(uint8_t)(v1<0?0:(v1>255?255:v1));
                    if(bx<W)img[((size_t)y*W+bx)*3]=(uint8_t)(v2<0?0:(v2>255?255:v2));
                }
            }
        }
    }

    /* horizontal boundaries (between rows y=7,15,23,...) */
    for(int bx=0;bx<W;bx+=8){
        for(int by=8;by<H;by+=8){
            for(int c=0;c<8;c++){
                int x=bx+c;
                if(x>=W)break;
                int p2=(by-2>=0)?img[((size_t)(by-2)*W+x)*3]:img[(size_t)x*3];
                int p1=img[((size_t)(by-1)*W+x)*3];
                int q1=img[((size_t)by*W+x)*3];
                int q2=(by+1<H)?img[((size_t)(by+1)*W+x)*3]:p1;

                int d=p1-q1;
                if(abs(d)>=alpha)continue;
                if(abs(p2-p1)<beta&&abs(q2-q1)<beta){
                    int delta=(d+2)>>2;
                    int v1=p1-delta,v2=q1+delta;
                    img[((size_t)(by-1)*W+x)*3]=(uint8_t)(v1<0?0:(v1>255?255:v1));
                    img[((size_t)by*W+x)*3]=(uint8_t)(v2<0?0:(v2>255?255:v2));
                }
            }
        }
    }
}
