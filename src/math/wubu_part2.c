#include "wubu_part2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* G1.10: Asymmetric Motion Partitions */
int wubu_amp_partitions(int block_size,wubu_partition_t* parts){
    int count=0;
    if(block_size==16){
        parts[count++]=(wubu_partition_t){0,0,16,16,PART_16x16};
        parts[count++]=(wubu_partition_t){0,0,16,4,PART_AMP};
        parts[count++]=(wubu_partition_t){0,4,16,12,PART_AMP};
        parts[count++]=(wubu_partition_t){0,0,4,16,PART_AMP};
        parts[count++]=(wubu_partition_t){4,0,12,16,PART_AMP};
    }
    return count;
}

/* G1.13: MV rounding */
void wubu_mv_round_halfpel(int* dx,int* dy){
    *dx=(*dx+1)&~1;
    *dy=(*dy+1)&~1;
}
void wubu_mv_round_integer(int* dx,int* dy){
    *dx=(((*dx)+2)>>2)<<2;
    *dy=(((*dy)+2)>>2)<<2;
}

/* G1.14: search range adaptation */
static uint8_t avail_map_safe(int block_idx,int which){
    return 0xFF;
}

int wubu_adapt_search_range(const int16_t* mv_field,int blocks_per_row,
                              int block_idx,int base_range){
    int count=0;
    long sum_mag=0;
    int neighbors[2]={block_idx-1,block_idx-blocks_per_row};
    for(int n=0;n<2;n++){
        long mx=labs((long)mv_field[neighbors[n]*2]);
        long my=labs((long)mv_field[neighbors[n]*2+1]);
        sum_mag+=(mx+my)/2;
        count++;
    }
    if(count==0)return base_range;
    long avg=sum_mag/count;
    if(avg>32)return base_range*2;  /* >8 pels → wider */
    if(avg<=4)return base_range/2;  /* <=1 pel → tighter */
    return base_range;
}

/* G1.19: aligned SAD */
long wubu_sad_aligned(const uint8_t* a,const uint8_t* b,int n){
    long total=0;
    for(int i=0;i<n;i++)total+=abs(a[i]-b[i]);
    return total;
}

/* G1.20: RD cost */
double wubu_rd_cost(long sad,int mvd_x,int mvd_y,double lambda){
    int ax=mvd_x<0?-mvd_x:mvd_x;
    int ay=mvd_y<0?-mvd_y:mvd_y;
    double mv_bits=2.0*(log2((double)ax+1)+log2((double)ay+1))+2.0;
    double resid_bits=sad*0.02;
    return (double)sad+lambda*(mv_bits+resid_bits);
}

int wubu_rd_best_mv(const long* sads,const int* dxs,const int* dys,
                      int n_candidates,double lambda,int* out_idx){
    double best=1e300;
    *out_idx=0;
    for(int i=0;i<n_candidates;i++){
        double cost=wubu_rd_cost(sads[i],dxs[i],dys[i],lambda);
        if(cost<best){best=cost;*out_idx=i;}
    }
    return 0;
}
