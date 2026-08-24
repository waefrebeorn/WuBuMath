/*
 * wubu_hstack.c -- GAP-C040: Multi-layer hyperbolic GCN stack with
 * residual connections (anti-oversmoothing)
 *
 * Research source: "Toward Deeper Hyperbolic GCNs" (arXiv:2310.02027)
 * + R-HGCN (AAAI 2024). The oversmoothing problem: naive stacked
 * aggregation pulls all node embeddings toward the gyromidpoint until
 * they're indistinguishable. The fix: Möbius-add residual connections —
 * each layer's update is x' = x ⊕ α·GCN(x) with learnable/small alpha,
 * preserving node identity while accumulating neighborhood info.
 *
 * Gates: (1) outputs on-ball through 4 layers, (2) node distinctness
 * (mean pairwise distance) does NOT collapse to zero — the anti-
 * oversmoothing invariant that justifies the whole architecture.
 */
#include "wubu_hstack.h"
#include "wubu_hgnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* one gyrolinear+projection pass (same math as B015's step 2) */
static int hs_linear(const float* W,const float* x,int N,int D,float c,
                     float* out){
    for(int i=0;i<N;i++){
        const float* xi=x+(size_t)i*D;
        float* oi=out+(size_t)i*D;
        for(int j=0;j<D;j++){
            float acc=0;
            for(int k=0;k<D;k++)acc+=W[(size_t)j*D+k]*xi[k];
            oi[j]=acc;
        }
        /* project */
        float n2=0;for(int d=0;d<D;d++)n2+=oi[d]*oi[d];
        if(n2>1.0f){float s=sqrtf(0.999f/n2);for(int d=0;d<D;d++)oi[d]*=s;}
    }
    return 0;
}

int wubu_hstack_forward(const float* x,const int* adj_idx,
                         const int* adj_ptr,const float* edge_weight,
                         const float* W,int L,int N,int D,float c,
                         float alpha,float* out){
    if(!x||!adj_idx||!adj_ptr||!W||!out||L<1)return -1;

    float* cur=malloc(sizeof(float)*(size_t)N*D);
    float* agg=malloc(sizeof(float)*(size_t)N*D);
    memcpy(cur,x,sizeof(float)*(size_t)N*D);

    for(int l=0;l<L;l++){
        const float* Wl=W+(size_t)l*D*D;
        /* aggregate neighbors into agg via wubu_hgnn_layer internals:
         * reuse hgnn's aggregation by calling its full layer then
         * treating output as pre-residual transform of it. Simpler:
         * call hgnn with identity W to get aggregated+projected, but we
         * want agg BEFORE linear... For gate purposes: run hgnn layer
         * (aggregate+linear), then blend: out = mobius_add(cur, alpha*agg).*/
        if(wubu_hgnn_layer(cur,adj_idx,adj_ptr,edge_weight,Wl,N,D,c,agg)!=0){
            free(cur);free(agg);return -2;
        }
        /* mobius residual: cur' = cur ⊕ (alpha ⊗ agg_dir) where
         * alpha scales the tangent contribution */
        for(int i=0;i<N;i++){
            float uu=0,vv=0,uv=0;
            float* ci=cur+(size_t)i*D;
            const float* ai=agg+(size_t)i*D;
            for(int d=0;d<D;d++){uu+=ci[d]*ci[d];vv+=ai[d]*ai[d];uv+=ci[d]*ai[d];}
            float num1=1+2*c*uv+c*alpha*alpha*vv;
            float num2=1-c*uu;
            float den=1+2*c*uv+c*c*uu*alpha*alpha*vv;
            if(den<1e-10f)den=1e-10f;
            for(int d=0;d<D;d++)
                ci[d]=(num1*ci[d]+num2*alpha*ai[d])/den;
            /* project */
            float n2=0;for(int d=0;d<D;d++)n2+=ci[d]*ci[d];
            if(n2>0.99998f){float s=sqrtf(0.99998f/n2);
                for(int d=0;d<D;d++)ci[d]*=s;}
        }
    }
    memcpy(out,cur,sizeof(float)*(size_t)N*D);
    free(cur);free(agg);
    return 0;
}

/* mean pairwise geodesic distance — the anti-oversmoothing metric */
float wubu_hstack_distinctness(const float* embs,int n,int D,float c){
    if(n<2)return 0;
    float total=0;int cnt=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float ab2=0,a2=0,b2=0;
            for(int d=0;d<D;d++){
                float df=embs[(size_t)i*D+d]-embs[(size_t)j*D+d];ab2+=df*df;
                a2+=embs[(size_t)i*D+d]*embs[(size_t)i*D+d];
                b2+=embs[(size_t)j*D+d]*embs[(size_t)j*D+d];
            }
            float den=(1-c*a2)*(1-c*b2);
            if(den<1e-9f)den=1e-9f;
            total+=acoshf(1+2*c*ab2/den)/sqrtf(c);
            cnt++;
        }
    return cnt>0?total/cnt:0;
}
