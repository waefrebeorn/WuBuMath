/*
 * wubu_partitions.c -- GROUP 1: Sub-block partitions + search patterns
 *
 * G1.09: Sub-block ME (4x4, 8x4, 4x8, 16x8, 8x16 partitions)
 * G1.15: Early termination based on block variance
 * G1.16-17: Diamond and hexagon search patterns
 */
#include "wubu_partitions.h"
#include "wubu_motionest.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Block variance for early termination ===== */

long wubu_block_variance(const uint8_t* img,int W,int H,
                          int bx,int by,int bs){
    long sum=0,sum_sq=0,n=0;
    for(int r=0;r<bs;r++){
        int y=by+r;if(y>=H)break;
        for(int c=0;c<bs;c++){
            int x=bx+c;if(x>=W)break;
            long v=img[(size_t)y*W+x];
            sum+=v;sum_sq+=v*v;n++;
        }
    }
    if(n==0)return 0;
    long mean=sum/n;
    return sum_sq/n-mean*mean;
}

/* ===== Partition enumeration for a block ===== */

int wubu_enum_partitions(int block_size,wubu_partition_t* parts){
    int count=0;
    switch(block_size){
        case 4:
            parts[count++]=(wubu_partition_t){0,0,4,4,PART_4x4};
            break;
        case 8:
            parts[count++]=(wubu_partition_t){0,0,8,8,PART_8x8};
            parts[count++]=(wubu_partition_t){0,0,8,4,PART_8x4};
            parts[count++]=(wubu_partition_t){0,4,8,4,PART_8x4};
            parts[count++]=(wubu_partition_t){0,0,4,8,PART_4x8};
            parts[count++]=(wubu_partition_t){4,0,4,8,PART_4x8};
            break;
        case 16:
            /* all partition modes for a 16×16 macroblock */
            parts[count++]=(wubu_partition_t){0,0,16,16,PART_16x16};
            /* 16x8 */
            parts[count++]=(wubu_partition_t){0,0,16,8,PART_16x8};
            parts[count++]=(wubu_partition_t){0,8,16,8,PART_16x8};
            /* 8x16 */
            parts[count++]=(wubu_partition_t){0,0,8,16,PART_8x16};
            parts[count++]=(wubu_partition_t){8,0,8,16,PART_8x16};
            /* 8x8 sub-blocks */
            for(int i=0;i<4;i++){
                int sx=8*(i%2),sy=8*(i/2);
                parts[count++]=(wubu_partition_t){sx,sy,8,8,PART_8x8};
            }
            break;
        default:
            parts[count++]=(wubu_partition_t){0,0,block_size,block_size,PART_16x16};
    }
    return count;
}

/* ===== Diamond Search Pattern ===== */

/* classic small diamond pattern: center + 4 cardinal neighbors */
static const int dia_dx[5]={0,-1,0,1,0};
static const int dia_dy[5]={0,0,1,0,-1};

/* hexagon pattern: center + 6 surrounding points */
static const int hex_dx[7]={0,-2,-1,1,2,1,-1};
static const int hex_dy[7]={0, 0,-1,-1,0,1, 1};

/*
 * Diamond search ME with early termination.
 * Returns SAD of best match. Sets MV in quarter-pel units.
 */
long wubu_diamond_search(const uint8_t* curr,const uint8_t* ref,
                           int W,int H,int bx,int by,int bs,
                           int search_range,int* out_dx,int* out_dy){

    
    int best_dx=0,best_dy=0;
    long best_sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,0,0);
    
    if(best_sad==0)return 0; /* already perfect */

    /* diamond iterations until no improvement or out of range */
    for(int iter=0;iter<search_range;iter++){
        int improved=0;
        for(int d=1;d<5;d++){ /* skip index 0 (center) */
            int ndx=best_dx+dia_dx[d];
            int ndy=best_dy+dia_dy[d];
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){
                best_sad=sad;best_dx=ndx;best_dy=ndy;
                improved=1;
            }
        }
        if(!improved)break;
    }

    /* local ±1 refinement */
    for(int dy=-1;dy<=1;dy++)
        for(int dx=-1;dx<=1;dx++){
            if(dx==0&&dy==0)continue;
            int ndx=best_dx+dx,ndy=best_dy+dy;
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){best_sad=sad;best_dx=ndx;best_dy=ndy;}
        }

    *out_dx=best_dx;*out_dy=best_dy;
    return best_sad;
}

/*
 * Hexagon search ME — better for larger motion ranges
 */
long wubu_hexagon_search(const uint8_t* curr,const uint8_t* ref,
                           int W,int H,int bx,int by,int bs,
                           int search_range,int* out_dx,int* out_dy){


    int best_dx=0,best_dy=0;
    long best_sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,0,0);
    
    if(best_sad==0)return 0;

    /* large hexagon pattern first */
    for(int step=search_range/2;step>=1;step/=2){
        int improved=1;
        while(improved){
            improved=0;
            for(int d=1;d<7;d++){ /* skip center */
                int ndx=best_dx+hex_dx[d]*step;
                int ndy=best_dy+hex_dy[d]*step;
                if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
                
                long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
                if(sad<best_sad){
                    best_sad=sad;best_dx=ndx;best_dy=ndy;
                    improved=1;
                }
            }
        }
    }
    
    /* final ±1 refinement */
    for(int dy=-1;dy<=1;dy++)
        for(int dx=-1;dx<=1;dx++){
            if(dx==0&&dy==0)continue;
            int ndx=best_dx+dx,ndy=best_dy+dy;
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){best_sad=sad;best_dx=ndx;best_dy=ndy;}
        }
    
    *out_dx=best_dx;*out_dy=best_dy;
    return best_sad;
}
