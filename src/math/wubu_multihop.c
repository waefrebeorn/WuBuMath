/*
 * wubu_multihop.c -- GAP-B018: Multi-hop geodesic neighborhood weighting
 *
 * HGCN's local tangent aggregation (their Eq. 9, Table 2: 82.0 vs 80.9
 * Att^o) extended to MULTIHOP: node i's layer-2 neighborhood (neighbors
 * of neighbors) contributes with weight decaying by path length —
 *   w_ij^(hop) = exp(-hop_count / tau)
 * so 1-hop neighbors dominate, 2-hop whisper, 3-hop barely register.
 *
 * Aggregation happens in the LOCAL TANGENT SPACE of each center node
 * (the paper's key finding: local beats origin-tangent).
 */
#include "wubu_multihop.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* BFS hop distances from src up to max_hops */
static int mh_bfs(const int* adj_idx,const int* adj_ptr,int n,int src,
                   int max_hops,uint8_t* hops){
    memset(hops,255,(size_t)n);   /* 255 = unreachable */
    hops[src]=0;
    int frontier[1024],nf=1;
    frontier[0]=src;
    for(int h=1;h<=max_hops&&nf>0;h++){
        int next[4096],nn=0;
        for(int fi=0;fi<nf;fi++){
            int u=frontier[fi];
            for(int p=adj_ptr[u];p<adj_ptr[u+1];p++){
                int v=adj_idx[p];
                if(hops[v]==255){hops[v]=(uint8_t)h;if(nn<4096)next[nn++]=v;}
            }
        }
        memcpy(frontier,next,sizeof(int)*(size_t)(nn<4096?nn:4096));
        nf=nn;
    }
    return 0;
}

int wubu_mh_aggregate(const float* x,const int* adj_idx,const int* adj_ptr,
                       int n,int D,float c,float tau,int max_hops,
                       float* out){
    if(!x||!adj_ptr||!out)return -1;
    uint8_t* hops=malloc((size_t)n);
    float* tang=malloc(sizeof(float)*(size_t)n*D);
    if(!hops||!tang){free(hops);free(tang);return -2;}

    /* precompute log_0 of every node once */
    for(int i=0;i<n;i++){
        const float* xi=x+(size_t)i*D;
        float n2=0;
        for(int d=0;d<D;d++)n2+=xi[d]*xi[d];
        float nx=sqrtf(n2);
        if(nx>1e-10f){
            float arg=sqrtf(c)*nx;
            if(arg>0.999999f)arg=0.999999f;
            float zn=atanhf(arg)/(sqrtf(c)*nx);
            for(int d=0;d<D;d++)tang[(size_t)i*D+d]=zn*xi[d];
        }else{
            memset(tang+(size_t)i*D,0,sizeof(float)*D);
        }
    }

    for(int i=0;i<n;i++){
        mh_bfs(adj_idx,adj_ptr,n,i,max_hops,hops);
        /* weighted tangent average: self + decaying hops */
        float wself=1.0f;
        float acc[512];
        int dd=D<512?D:512;
        for(int d=0;d<dd&&d<512;d++)acc[d]=wself*tang[(size_t)i*D+d];
        float wsum=wself;
        for(int j=0;j<n;j++){
            if(j==i||hops[j]==255||hops[j]==0)continue;
            float w=expf(-(float)hops[j]/tau);
            for(int d=0;d<dd&&d<512;d++)acc[d]+=w*tang[(size_t)j*D+d];
            wsum+=w;
        }
        /* back through exp_0 */
        float vn2=0;
        for(int d=0;d<dd&&d<512;d++){acc[d]/=wsum;vn2+=acc[d]*acc[d];}
        float nv=sqrtf(vn2);
        if(nv>1e-10f){
            float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
            for(int d=0;d<D&&d<512;d++)out[(size_t)i*D+d]=coeff*acc[d];
        }else{
            for(int d=0;d<D&&d<512;d++)out[(size_t)i*D+d]=0;
        }
        for(int d=512;d<D;d++)out[(size_t)i*D+d]=0;
        float n2c=0;
        for(int d=0;d<D;d++)n2c+=out[(size_t)i*D+d]*out[(size_t)i*D+d];
        if(n2c>0.99998f){
            float s=sqrtf(0.99998f/n2c);
            for(int d=0;d<D;d++)out[(size_t)i*D+d]*=s;
        }
    }
    free(hops);free(tang);
    return 0;
}
