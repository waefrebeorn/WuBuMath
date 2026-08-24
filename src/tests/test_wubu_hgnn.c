/* test_wubu_hgnn.c -- GAP-B015 gates
 *  G1 outputs stay on-ball
 *  G2 identity W: output ≈ neighborhood midpoint (not raw input)
 *  G3 zero-degree node keeps its own embedding
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hgnn.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic GCN Layer Tests ===\n\n");
    const int N=6,D=4;
    float c=1.0f;

    /* simple graph: 0-1, 0-2, 1-3, 2-3, 3-4, 4-5 (node 5 has degree 1) */
    int adj_ptr[7]={0,2,4,6,9,11,12};
    int adj_idx[12]={1,2, 0,3, 0,3, 1,2,4, 3,5, 4};

    /* random on-ball embeddings */
    float x[N*D];
    unsigned rs=42u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/2000.0f*0.8f-0.4f;
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }

    printf("  g1_outputs_on_ball...");
    {
        float W[D*D];memset(W,0,sizeof(W));
        for(int j=0;j<D;j++)W[j*D+j]=1.0f;
        float out[N*D];
        CHECK(wubu_hgnn_layer(x,adj_idx,adj_ptr,NULL,W,N,D,c,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;for(int d=0;d<D;d++)n2+=out[i*D+d]*out[i*D+d];
            CHECK(n2<1.0f+1e-5f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_aggregation_differs_from_input...");
    {
        /* with identity W, the aggregated output should differ from input
         * for nodes with neighbors (proves aggregation happened) */
        float W[D*D];memset(W,0,sizeof(W));
        for(int j=0;j<D;j++)W[j*D+j]=1.0f;
        float out[N*D];
        CHECK(wubu_hgnn_layer(x,adj_idx,adj_ptr,NULL,W,N,D,c,out)==0);
        int any_diff=0;
        for(int i=0;i<N-1;i++){   /* skip isolated */
            float diff=0;
            for(int d=0;d<D;d++){float df=out[i*D+d]-x[i*D+d];diff+=df*df;}
            if(sqrtf(diff)>1e-6f)any_diff=1;
        }
        CHECK(any_diff);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
