/*
 * wubu_hilbert.c -- GAP-A011: Hilbert-curve scan order (the locality
 * baseline that golden-angle must beat) + A010's locality metric
 * applied across three scan strategies
 *
 * Research source: Chen 2022 — Hilbert curve provides the BEST spatial
 * correlation preservation of any space-filling curve (vs Z-order,
 * Gray-coded). This is the compression baseline: consecutive Hilbert
 * samples are always spatially adjacent → smallest deltas → best
 * entropy coding.
 */
#include "wubu_hilbert.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* iterative Hilbert curve d2xy — Wikipedia, with d/=4 per level */
static void hilbert_d2xy(int n,int* x,int* y,int d){
    int rx,ry,t,d2=d;
    *x=*y=0;
    for(int s=1;s<n;s*=2){
        rx=1&(d2/2);
        ry=1&(d2^rx);
        /* rotate */
        if(ry==0){
            if(rx==1){
                *x=s-1-*x;
                *y=s-1-*y;
            }
            t=*x;*x=*y;*y=t;
        }
        *x+=s*rx;
        *y+=s*ry;
        d2/=4;
    }
}

void wubu_hil_order(int side,float* out_x,float* out_y){
    int n=side*side;
    for(int d=0;d<n;d++){
        int x,y;
        hilbert_d2xy(side,&x,&y,d);
        out_x[d]=(float)x/side-0.5f;
        out_y[d]=(float)y/side-0.5f;
    }
}

float wubu_gs_locality(const float* xs,const float* ys,int n);

float wubu_hil_locality(const float* xs,const float* ys,int n){
    return wubu_gs_locality(xs,ys,n);
}
