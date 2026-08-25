/*
 * wubu_htree_softmax.c -- GAP-C047: Hyperbolic hierarchical softmax
 * (log-depth classification via tree-path routing)
 *
 * Hierarchical softmax (Morin & Bengio 2005) hyperbolized: classes are
 * leaves of a binary tree; each internal node scores its two children
 * by geodesic affinity to the input, giving a log2(K)-depth decision
 * path instead of K prototypes. Probability of a leaf = product of
 * sigmoid branch probabilities along its root-to-leaf path.
 *
 * The hyperbolic fit: deep-tree class hierarchies embed with low
 * distortion on the ball, so geodesic affinity at each branch is
 * exactly the right local decision variable.
 */
#include "wubu_htree_softmax.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float hs_gd(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* P(go left) from child-distance difference */
static float hs_branch_left(const float* x,const float* lp,const float* rp,
                             int D,float c){
    float dl=hs_gd(x,lp,D,c);
    float dr=hs_gd(x,rp,D,c);
    return 1.0f/(1.0f+expf((dl-dr)/0.5f));
}

int wubu_hts_init(WubuHTS* t,int n_leaves,int D,float c){
    if(n_leaves<2)return -1;
    /* complete binary tree: nodes 0..n_leaves-2 internal, leaves after */
    int n_int=n_leaves-1;
    t->n_leaves=n_leaves;t->D=D;t->c=c;
    t->n_nodes=n_int+n_leaves;
    t->left=malloc(sizeof(int)*(size_t)t->n_nodes);
    t->right=malloc(sizeof(int)*(size_t)t->n_nodes);
    t->proto=malloc(sizeof(float)*(size_t)t->n_nodes*D);
    if(!t->left||!t->right||!t->proto)return -2;
    unsigned rs=42u;
    for(int i=0;i<t->n_nodes;i++)t->left[i]=t->right[i]=-1;
    /* build: internal node i has children 2i+1, 2i+2 (heap layout) */
    for(int i=0;i<n_int;i++){
        t->left[i]=2*i+1;
        t->right[i]=2*i+2;
    }
    for(int i=0;i<t->n_nodes*D;i++){
        rs=rs*1103515245u+12345u;
        t->proto[i]=((float)((rs>>16)%2000))/20000.0f-0.05f;
    }
    return 0;
}
void wubu_hts_free(WubuHTS* t){free(t->left);free(t->right);free(t->proto);}

float wubu_hts_leaf_prob(const WubuHTS* t,const float* x,int leaf){
    if(leaf<0||leaf>=t->n_leaves)return 0;
    /* leaf node index in array: n_int + leaf where n_int=n_leaves-1 */
    int n_int=t->n_leaves-1;
    int target=n_int+leaf;
    /* walk from root following the heap path to `target`,
     * multiplying branch probabilities */
    /* path bits: from most significant to least of (target+1) */
    int v=target+1;
    int bits[32],nb=0;
    while(v>1){bits[nb++]=v&1;v>>=1;}
    float p=1.0f;
    int node=0;
    for(int i=nb-1;i>=0;i--){
        if(node<0||node>=t->n_nodes)return 0;
        int ln=t->left[node],rn=t->right[node];
        if(ln<0||rn<0||ln>=t->n_nodes||rn>=t->n_nodes)return 0;
        float p_left=hs_branch_left(x,
            t->proto+(size_t)ln*t->D,
            t->proto+(size_t)rn*t->D,t->D,t->c);
        /* bit=1 -> RIGHT child; bit=0 -> LEFT */
        p*=bits[i]?(1.0f-p_left):p_left;
        node=bits[i]?rn:ln;
    }
    return p;
}

int wubu_hts_predict(const WubuHTS* t,const float* x){
    int best=0;float bp=-1;
    for(int l=0;l<t->n_leaves;l++){
        float p=wubu_hts_leaf_prob(t,x,l);
        if(p>bp){bp=p;best=l;}
    }
    return best;
}
