/*
 * wubu_hier.c -- GAP-D015: Hyperbolic hierarchical classification
 *
 * Uses the tree embedding (A021) to place a class hierarchy on the ball,
 * then classifies by nearest ancestor path: an input is classified to the
 * leaf whose prototype is nearest in geodesic distance, with confidence
 * derived from the margin over sibling leaves.
 *
 * The key hyperbolic advantage: deeper classes (more specific) sit
 * farther from the origin, so the geodesic distance encodes both
 * semantic similarity AND hierarchy depth simultaneously.
 */
#include "wubu_hier.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_hier_init(WubuHier* h,const int* parent,int n_nodes,
                   const int* leaf_indices,int n_leaves,int D,float c){
    if(n_nodes<1||n_leaves<1||D<1)return -1;
    h->n=n_nodes;h->n_leaf=n_leaves;h->D=D;h->c=c;
    h->parent=malloc(sizeof(int)*(size_t)n_nodes);
    h->leaf_idx=malloc(sizeof(int)*(size_t)n_leaves);
    h->proto=malloc(sizeof(float)*(size_t)n_nodes*D);
    if(!h->parent||!h->leaf_idx||!h->proto)return -2;
    memcpy(h->parent,parent,sizeof(int)*(size_t)n_nodes);
    memcpy(h->leaf_idx,leaf_indices,sizeof(int)*(size_t)n_leaves);

    /* embed the hierarchy using our Sarkar construction */
    if(wubu_tree_embed(parent,n_nodes,D,1.0f,h->proto)!=0)return -3;
    return 0;
}

void wubu_hier_free(WubuHier* h){
    free(h->parent);free(h->leaf_idx);free(h->proto);
    h->parent=NULL;h->leaf_idx=NULL;h->proto=NULL;
}

static float hier_dist(const float* a,const float* b,int D,float c){
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

/* classify: returns index into leaf_indices of nearest leaf */
int wubu_hier_classify(const WubuHier* h,const float* x,int* out_conf){
    float best=1e30f;int bi=-1;
    float second=1e30f;
    for(int li=0;li<h->n_leaf;li++){
        int ni=h->leaf_idx[li];
        float d=hier_dist(x,h->proto+(size_t)ni*h->D,h->D,h->c);
        if(d<best){second=best;best=d;bi=li;}
        else if(d<second){second=d;}
    }
    /* margin = second-best minus best (positive = confident) */
    if(out_conf)*out_conf=(int)(1000.0f*(second-best));
    return bi;
}

/* fraction of correct classifications on labeled data */
float wubu_hier_accuracy(const WubuHier* h,const float* xs,
                          const int* true_leaf,int n){
    int hits=0;
    for(int i=0;i<n;i++){
        int pred=wubu_hier_classify(h,xs+(size_t)i*h->D,NULL);
        if(pred==true_leaf[i])hits++;
    }
    return (n>0)?(float)hits/n:0.0f;
}
