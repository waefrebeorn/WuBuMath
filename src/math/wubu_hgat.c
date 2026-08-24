/*
 * wubu_hattention_gnn.c -- GAP-C041: Hyperbolic attention-based GNN layer
 *
 * Research source: HGAT (arXiv:1912.03046) + HGCN's attention aggregation.
 * Unlike B015's uniform-weight gyromidpoint, this layer computes
 * ATTENTION weights from hyperbolic distances between center and neighbor:
 *
 *   w_ij = softmax_j( -d(x_i, x_j) / tau )
 *
 * so closer (more hierarchically related) neighbors contribute more to
 * the aggregated midpoint. This is the "local tangent attention" that
 * HGCN showed outperforms origin-tangent aggregation (Table 2: 90.8 vs
 * 82.0 on Disease).
 */
#include "wubu_hgat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hgat_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* midpoint of K points with given weights (Einstein, from hgnn) */
static void hgat_midpoint(const float* vals,const float* w,int K,int D,
                           float c,float* out){
    float num[64],den=1e-10f,tm[64];
    int dd=D<64?D:64;
    memset(num,0,sizeof(float)*(size_t)dd);
    for(int k=0;k<K;k++){
        const float* x=vals+(size_t)k*D;
        float n2=0;for(int d=0;d<dd;d++)n2+=x[d]*x[d];
        float gamma=2/(1-c*n2);if(gamma<1)gamma=1;
        for(int d=0;d<dd;d++)num[d]+=gamma*w[k]*x[d];
        den+=(gamma-1)*w[k];
    }
    float tn2=0;
    for(int d=0;d<dd;d++){tm[d]=num[d]/den;tn2+=tm[d]*tm[d];}
    float disc=1-c*tn2;if(disc<1e-9f)disc=1e-9f;
    float sc=1/(1+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=(d<dd)?tm[d]*sc:0;
}

int wubu_hgat_forward(const float* x,const int* adj_idx,
                       const int* adj_ptr,const float* W_self,
                       const float* alpha_vec,   /* [D] learned attn scale */
                       int N,int D,float c,float tau,float* out){
    if(!x||!adj_idx||!adj_ptr||!out)return -1;

    /* self transform once per node */
    float* self_t=malloc(sizeof(float)*(size_t)N*D);
    if(!self_t)return -2;
    for(int i=0;i<N;i++){
        const float* xi=x+(size_t)i*D;
        float n2=0;for(int d=0;d<D;d++)n2+=xi[d]*xi[d];
        float nv=sqrtf(n2);
        if(nv>1e-10f){
            float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
            float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
            for(int d=0;d<D;d++)
                self_t[(size_t)i*D+d]=zn*xi[d]*(alpha_vec?alpha_vec[d]:1.0f);
        }else{
            memset(self_t+(size_t)i*D,0,sizeof(float)*D);
        }
    }

    for(int i=0;i<N;i++){
        int start=adj_ptr[i],end=adj_ptr[i+1];
        int deg=end-start;
        if(deg<=0){
            memcpy(out+(size_t)i*D,x+(size_t)i*D,sizeof(float)*D);
            continue;
        }
        /* attention weights over neighbors: closer → bigger weight */
        float* w=malloc(sizeof(float)*(size_t)deg);
        float mx=-1e30f,z=0;
        for(int j=start;j<end;j++){
            int nb=adj_idx[j];
            float d=hgat_dist(x+(size_t)i*D,x+(size_t)nb*D,D,c);
            w[j-start]=-d/(tau*(alpha_vec?alpha_vec[0]:1.0f)+1e-6f);
            if(w[j-start]>mx)mx=w[j-start];
        }
        for(int j=0;j<deg;j++){w[j]=expf(w[j]-mx);z+=w[j];}
        for(int j=0;j<deg;j++)w[j]/=(z>0?z:1);

        /* gather neighbor embeddings + midpoint */
        float* neigh=malloc(sizeof(float)*(size_t)deg*D);
        for(int j=start;j<end;j++)
            memcpy(neigh+(size_t)(j-start)*D,x+(size_t)adj_idx[j]*D,
                   sizeof(float)*D);
        float mid[64];
        hgat_midpoint(neigh,w,deg,D,c,mid);

        /* residual: out = x_i ⊕ self_transform(x_i) ⊕ midpoint-ish blend.
         * Simplify: out = mobius_add(x_i, midpoint) — the neighborhood
         * info rides on the residual identity. */
        {
            float uu=0,vv=0,uv=0;
            const float* xi=x+(size_t)i*D;
            for(int d=0;d<D&&d<64;d++){
                uu+=xi[d]*xi[d];vv+=mid[d]*mid[d];uv+=xi[d]*mid[d];
            }
            float num1=1+2*c*uv+c*vv,num2=1-c*uu;
            float den=1+2*c*uv+c*c*uu*vv;
            if(den<1e-10f)den=1e-10f;
            for(int d=0;d<D&&d<64;d++)
                out[(size_t)i*D+d]=(num1*xi[d]+num2*mid[d])/den;
            for(int d=64;d<D;d++)out[(size_t)i*D+d]=0;
            /* project */
            float n2=0;for(int d=0;d<D;d++)n2+=out[(size_t)i*D+d]*out[(size_t)i*D+d];
            if(n2>0.99998f){float s=sqrtf(0.99998f/n2);
                for(int d=0;d<D;d++)out[(size_t)i*D+d]*=s;}
        }
        free(neigh);free(w);
    }
    free(self_t);
    return 0;
}
