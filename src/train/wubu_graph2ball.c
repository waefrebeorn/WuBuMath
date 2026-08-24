/*
 * wubu_graph2ball.c -- GAP-D022: Graph-to-ball embedding via random walks
 * (node2vec-style, hyperbolic negative sampling)
 *
 * Pipeline:
 *   1. generate truncated random walks on the graph (uniform neighbor choice)
 *   2. for each walk window: skip-gram pairs (center, context)
 *   3. train Poincaré embeddings (D017) on positive pairs + random negatives
 *
 * The result: adjacent graph nodes land close on the ball; the hierarchy
 * of the graph emerges in the radial dimension automatically.
 */
#include "wubu_graph2ball.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* CSR adjacency assumed: adj_idx/adj_ptr like B015 */

int wubu_g2b_init(WubuG2B* g,int n,int D,float c,float lr,unsigned seed){
    g->n=n;g->D=D;g->c=c;g->lr=lr;
    return wubu_pe_init(&g->pe,n,D,c,lr,seed);
}
void wubu_g2b_free(WubuG2B* g){wubu_pe_free(&g->pe);}

static unsigned grs=1u;
static int gr_rand(int mod){
    grs=grs*1103515245u+12345u;
    return (int)((grs>>16)%((unsigned)mod?mod:1));
}

/* one random walk from start, length len; writes walk[] */
void wubu_g2b_walk(const int* adj_idx,const int* adj_ptr,
                   int n,int start,int len,int* walk){
    int cur=start;
    for(int s=0;s<len;s++){
        walk[s]=cur;
        int deg=adj_ptr[cur+1]-adj_ptr[cur];
        if(deg<=0){
            /* teleport */
            cur=gr_rand(n);
        }else{
            cur=adj_idx[adj_ptr[cur]+gr_rand(deg)];
        }
    }
}

/* train on `num_walks` walks per node, window size w, k negatives per pair */
float wubu_g2b_train(WubuG2B* g,const int* adj_idx,const int* adj_ptr,
                     int walk_len,int num_walks,int window,int k_neg){
    float total_loss=0;
    int pairs=0;
    int* walk=malloc(sizeof(int)*(size_t)walk_len);

    /* seed the RNG deterministically per call count */
    static unsigned call_count=0u;
    grs=++call_count*7919u+13u;

    for(int w=0;w<num_walks;w++){
        for(int start=0;start<g->n;start++){
            wubu_g2b_walk(adj_idx,adj_ptr,g->n,start,walk_len,walk);
            /* skip-gram pairs within window */
            for(int center=0;center<walk_len;center++){
                for(int off=1;off<=window&&center+off<walk_len;off++){
                    int u=walk[center],v=walk[center+off];
                    if(u==v)continue;
                    int neg[8];
                    for(int k=0;k<k_neg&&k<8;k++)
                        neg[k]=gr_rand(g->n);
                    total_loss+=wubu_pe_train_edge(&g->pe,u,v,neg,k_neg<8?k_neg:8);
                    pairs++;
                }
            }
        }
    }
    free(walk);
    return pairs>0?total_loss/pairs:0.0f;
}

/* average distance between adjacent graph nodes vs non-adjacent pairs */
float wubu_g2b_separation(const WubuG2B* g,const int* adj_idx,
                           const int* adj_ptr){
    int D=g->D;
    float edge_sum=0;int edges=0;
    for(int i=0;i<g->n;i++)
        for(int j=adj_ptr[i];j<adj_ptr[i+1];j++){
            int nb=adj_idx[j];
            if(nb>i){
                edge_sum+=wubu_pe_dist_public(&g->pe,i,nb);
                edges++;
            }
        }
    float non_sum=0;int non=0;
    /* sample some non-edges */
    for(int i=0;i<g->n&&non<edges;i++)
        for(int j=i+2;j<g->n&&non<edges;j++){
            /* check not an edge: linear scan (fine for tests) */
            int is_edge=0;
            for(int jj=adj_ptr[i];jj<adj_ptr[i+1];jj++)
                if(adj_idx[jj]==j){is_edge=1;break;}
            for(int jj=adj_ptr[j];jj<adj_ptr[j+1];jj++)
                if(adj_idx[jj]==i){is_edge=1;break;}
            if(is_edge)continue;
            non_sum+=wubu_pe_dist_public(&g->pe,i,j);
            non++;
        }
    if(edges==0||non==0)return -1;
    float edge_avg=edges>0?edge_sum/edges:0;
    float non_avg=non>0?non_sum/non:0;
    return non_avg-edge_avg;   /* positive = edges closer */
}

float wubu_pe_dist_public(const WubuPEmb* pe,int u,int v){
    if(u<0||v<0||u>=pe->n||v>=pe->n)return -1.0f;
    const float* a=pe->emb+(size_t)u*pe->D;
    const float* b=pe->emb+(size_t)v*pe->D;
    float c=pe->c;
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<pe->D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}
