/*
 * wubu_hiermerge.c -- GAP-D016: Hyperbolic hierarchical clustering
 * (Chami et al. NeurIPS 2020 "From Trees to Continuous Embeddings and Back")
 *
 * The decoder half of A021's encoder: recover a binary tree FROM hyperbolic
 * embeddings. Algorithm (HMLC / hyperbolic agglomerative):
 *   1. start with every point as its own cluster, prototype = the point
 *   2. find the pair of clusters with smallest LCA distance
 *      (approximated here by their geodesic midpoint's depth — deeper
 *      pairs merge first, matching hyperbolic tree geometry)
 *   3. merge them; new prototype = gyromidpoint of the two children
 *   4. repeat until one cluster remains; record merges as a tree
 */
#include "wubu_hiermerge.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float wubu_hm_distance(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1+2*c*ab2/den;
    return acoshf(arg>1?arg:1)/sqrtf(c);
}

/* gyromidpoint of two points (Einstein half-way) */
static void hm_midpoint(const float* a,const float* b,int D,float c,float* out){
    /* mobius add of scaled halves: out = a ⊕ (0.5 ⊗ (⊖a ⊕ b)) */
    float neg[64],v[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd;d++)neg[d]=-a[d];
    for(int d=dd;d<D;d++)neg[d]=0;
    /* v = (-a) ⊕ b */
    {
        float uu=0,vv=0,uv=0;
        for(int d=0;d<dd;d++){uu+=neg[d]*neg[d];vv+=b[d]*b[d];uv+=neg[d]*b[d];}
        float num1=1+2*c*uv+c*vv,num2=1-c*uu;
        float den=1+2*c*uv+c*c*uu*vv;
        if(den<1e-10f)den=1e-10f;
        for(int d=0;d<dd;d++)v[d]=(num1*neg[d]+num2*b[d])/den;
    }
    /* scale v by 0.5: tanh(0.5*atanh(sqrt(c)|v|))/(sqrt(c)|v|)*v */
    float vn2=0;for(int d=0;d<dd;d++)vn2+=v[d]*v[d];
    float nv=sqrtf(vn2);
    float tv[64];memset(tv,0,sizeof(float)*64);
    if(nv>1e-10f){
        float coeff=tanhf(0.5f*atanhf(sqrtf(c)*nv))/(sqrtf(c)*nv);
        for(int d=0;d<dd;d++)tv[d]=coeff*v[d];
    }
    /* out = a ⊕ tv */
    {
        float uu=0,vv=0,uv=0;
        for(int d=0;d<dd;d++){uu+=a[d]*a[d];vv+=tv[d]*tv[d];uv+=a[d]*tv[d];}
        float num1=1+2*c*uv+c*vv,num2=1-c*uu;
        float den=1+2*c*uv+c*c*uu*vv;
        if(den<1e-10f)den=1e-10f;
        for(int d=0;d<dd;d++)out[d]=(num1*a[d]+num2*tv[d])/den;
        for(int d=dd;d<D;d++)out[d]=0;
        /* project */
        float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
        if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<D;d++)out[d]*=s;}
    }
}

int wubu_hm_cluster(const float* pts,int n,int D,float c,
                    WubuHMNode* tree /* [2n-1] */){
    if(!pts||!tree||n<1)return -1;
    int total=2*n-1;

    /* init leaves */
    for(int i=0;i<n;i++){
        memcpy(tree[i].proto,pts+(size_t)i*D,sizeof(float)*D);
        tree[i].left=-1;tree[i].right=-1;
        tree[i].size=1;tree[i].alive=1;
    }
    int next=n;
    int alive_count=n;

    while(alive_count>1&&next<total){
        /* find closest alive pair */
        float best=1e30f;int bi=-1,bj=-1;
        for(int i=0;i<next;i++){
            if(!tree[i].alive)continue;
            for(int j=i+1;j<next;j++){
                if(!tree[j].alive)continue;
                float d=wubu_hm_distance(tree[i].proto,tree[j].proto,D,c);
                if(d<best){best=d;bi=i;bj=j;}
            }
        }
        if(bi<0)break;
        /* merge into node `next` */
        tree[next].left=bi;tree[next].right=bj;
        tree[next].size=tree[bi].size+tree[bj].size;
        tree[next].alive=1;
        hm_midpoint(tree[bi].proto,tree[bj].proto,D,c,tree[next].proto);
        tree[bi].alive=0;tree[bj].alive=0;
        next++;alive_count--;
    }
    return next-1;  /* index of root */
}
