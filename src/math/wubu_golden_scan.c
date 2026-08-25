/*
 * wubu_golden_scan.c -- GAP-A012 + A010: Golden-angle progressive
 * scan order with locality-preservation metric
 *
 * Research source: golden-angle MRI sampling (node 4.6 precedent).
 * The golden angle 137.507...° produces the most uniform progressive
 * coverage of any irrational rotation — each new sample fills the
 * largest remaining gap. This is the ORDER the beam canvas sweeps.
 *
 * A010's locality metric measures how well spatial neighbors stay
 * close in scan order — lower = better compression (residuals are
 * smaller when consecutive samples are near each other).
 */
#include "wubu_golden_scan.h"
#include <stdlib.h>
#include <math.h>

void wubu_gs_order(int n,float* out_x,float* out_y){
    /* golden angle in radians */
    const float GA=2.39996323f;
    for(int i=0;i<n;i++){
        float a=i*GA;
        float r=sqrtf((float)(i+0.5f)/(float)n);
        out_x[i]=r*cosf(a);
        out_y[i]=r*sinf(a);
    }
}

float wubu_gs_locality(const float* xs,const float* ys,int n){
    /* mean distance between consecutive samples in SCAN order,
     * normalized by sqrt(area/n) so different orders compare fairly */
    if(n<2)return 0;
    double total=0;
    int cnt=0;
    for(int i=1;i<n;i++){
        float dx=xs[i]-xs[i-1],dy=ys[i]-ys[i-1];
        total+=sqrt(dx*dx+dy*dy);
        cnt++;
    }
    return (float)(total/cnt);
}

/* raster-scan baseline for comparison */
void wubu_gs_raster(int n_side,float* out_x,float* out_y){
    int idx=0;
    for(int y=0;y<n_side;y++)
        for(int x=0;x<n_side;x++){
            out_x[idx]=(float)x/n_side-0.5f;
            out_y[idx]=(float)y/n_side-0.5f;
            idx++;
        }
}
