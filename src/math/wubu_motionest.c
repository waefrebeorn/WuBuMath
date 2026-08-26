/*
 * wubu_motionest.c -- GAP-C075: Block matching motion estimation
 * (SAD-based integer-pel search with diamond refinement)
 *
 * The missing piece for translational motion: instead of assuming
 * pure rotation, find the (dx,dy) offset that minimizes SAD between
 * the current block and the reference frame. This handles pan, tilt,
 * and object motion that rotation-only prediction cannot.
 *
 * Search strategy: small diamond search (fast, near-optimal) followed
 * by optional full search within a limited window.
 */
#include "wubu_motionest.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* SAD between a block in curr at (bx,by) and ref at (bx+dx,by+dy) */
long wubu_me_sad(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,int dx,int dy){
    long sad=0;
    for(int r=0;r<bs;r++){
        int cy=by+r;
        int ry=cy+dy;
        if(ry<0||ry>=H){return 1LL<<30;}  /* out of bounds = max cost */
        for(int c=0;c<bs;c++){
            int cx=bx+c;
            int rx=cx+dx;
            if(rx<0||rx>=W){sad+=(1<<20);continue;}
            sad+=abs(curr[((size_t)cy*W+cx)*3]-ref[((size_t)ry*W+rx)*3]);
        }
    }
    return sad;
}

/* find best motion vector for one block using diamond + local search */
long wubu_me_block(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy){
    /* start at (0,0), do diamond search pattern */
    int best_dx=0,best_dy=0;
    long best_sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,0,0);

    /* diamond search: check 4 neighbors iteratively */
    static const int dia_dx[]={0,-1,0,1,0};
    static const int dia_dy[]={-1,0,1,0,-1};

    int improved=1;
    while(improved){
        improved=0;
        for(int d=0;d<4;d++){
            int ndx=best_dx+dia_dx[d];
            int ndy=best_dy+dia_dy[d];
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){
                best_sad=sad;best_dx=ndx;best_dy=ndy;
                improved=1;
            }
        }
    }

    /* local refinement: ±1 around best */
    for(int dy=-1;dy<=1;dy++)
        for(int dx=-1;dx<=1;dx++){
            if(dx==0&&dy==0)continue;
            int ndx=best_dx+dx,ndy=best_dy+dy;
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){
                best_sad=sad;best_dx=ndx;best_dy=ndy;
            }
        }

    *out_dx=best_dx;
    *out_dy=best_dy;
}

/* estimate motion vectors for all blocks in a frame */
int wubu_me_frame(const uint8_t* curr,const uint8_t* ref,
                   int W,int H,int bs,int search_range,
                   int* out_mvs /* [n_blocks_x*n_blocks_y][2] */){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx,dy;
            wubu_me_block(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy);
            out_mvs[count*2]=dx;
            out_mvs[count*2+1]=dy;
            count++;
        }
    return count;
}

/* apply motion compensation: build predicted frame from reference + MVs */
void wubu_me_compensate(const uint8_t* ref,int W,int H,int bs,
                          const int* mvs,uint8_t* predicted){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx=mvs[count*2],dy=mvs[count*2+1];
            count++;
            for(int r=0;r<bs;r++){
                int sy=by*bs+r,ty=sy+dy;
                if(ty<0||ty>=H)continue;
                for(int c=0;c<bs;c++){
                    int sx=bx*bs+c,tx=sx+dx;
                    if(tx<0||tx>=W)continue;
                    for(int ch=0;ch<3;ch++)
                        predicted[((size_t)sy*W+sx)*3+ch]=
                            ref[((size_t)ty*W+tx)*3+ch];
                }
            }
        }
}
