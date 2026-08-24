/*
 * wubu_hknn.c -- GAP-D010 companion: hyperbolic k-NN retrieval
 *
 * Brute-force k-nearest neighbor search on the Poincaré ball using
 * geodesic distance. This is the production retrieval engine that the
 * manifold-CLIP recall@k gates (D010) exercise. Simple, correct, and
 * sufficient for our corpus sizes; graph-based HNSW is a future gap.
 */
#include "wubu_hknn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float wubu_hknn_distance(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

int wubu_hknn_search(const float* db,int n,const float* query,
                     int D,float c,int k,
                     int* out_idx,float* out_dist){
    if(k<=0||k>n)return -1;
    /* compute all distances */
    float* dists=malloc(sizeof(float)*(size_t)n);
    if(!dists)return -2;
    for(int i=0;i<n;i++)
        dists[i]=wubu_hknn_distance(query,db+(size_t)i*D,D,c);
    /* GAP-D010 fix: sort (dist,idx) as a single pair array */
    typedef struct { float d; int i; } HKNNPair;
    HKNNPair* pairs=malloc(sizeof(HKNNPair)*(size_t)n);
    for(int i=0;i<n;i++){pairs[i].d=dists[i];pairs[i].i=i;}
    for(int a=0;a<k&&a<n;a++){
        int min_i=a;
        for(int b=a+1;b<n;b++)
            if(pairs[b].d<pairs[min_i].d)min_i=b;
        if(min_i!=a){
            HKNNPair tp=pairs[a];pairs[a]=pairs[min_i];pairs[min_i]=tp;
        }
    }
    for(int a=0;a<k;a++){
        out_idx[a]=pairs[a].i;
        out_dist[a]=pairs[a].d;
    }
    free(pairs);
    free(dists);
    return 0;
}

/* batch: compute recall@k against ground truth labels */
float wubu_hknn_recall(const float* db_emb,const float* query_emb,
                       const int* gt_labels,int n_queries,int D,float c,int k){
    /* db_emb and query_emb are the same set; exclude self-match */
    int hits=0;
    float* dists=malloc(sizeof(float)*(size_t)n_queries);
    int* idx=malloc(sizeof(int)*(size_t)n_queries);
    for(int qi=0;qi<n_queries;qi++){
        const float* q=query_emb+(size_t)qi*D;
        for(int j=0;j<n_queries;j++)
            dists[j]=wubu_hknn_distance(q,db_emb+(size_t)j*D,D,c);
        /* partial selection top-k excluding self */
        for(int a=0;a<k&&a<n_queries;a++){
            int min_i=-1;float min_d=1e30f;
            for(int b=0;b<n_queries;b++){
                if(b==qi)continue;
                int already=0;
                for(int prev=0;prev<a;prev++)if(idx[prev]==b){already=1;break;}
                if(already)continue;
                if(dists[b]<min_d){min_d=dists[b];min_i=b;}
            }
            idx[a]=min_i;
        }
        /* check if any of the top-k has same label */
        for(int a=0;a<k;a++){
            if(idx[a]>=0&&idx[a]<n_queries&&gt_labels[idx[a]]==gt_labels[qi]){
                hits++;break;
            }
        }
    }
    free(dists);free(idx);
    return (n_queries>0)?(float)hits/n_queries:0.0f;
}
