/*
 * wubu_hdt.c -- GAP-D024: Hyperbolic decision tree (geodesic splits)
 *
 * Research source: Chlenski et al. 2023 "Fast Hyperboloid Decision Tree
 * Algorithms" (arXiv:2310.13841). Splits partition the ball by geodesic
 * distance to an anchor point: x goes LEFT iff d(x,anchor) < threshold.
 * Geodesic balls are convex and topologically continuous — unlike naive
 * coordinate thresholds which cut the ball into weird lens shapes.
 *
 * Greedy axis-free construction: at each node, search over candidate
 * anchors (training points) × thresholds; pick the split maximizing
 * information gain. Leaves carry majority labels.
 */
#include "wubu_hdt.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hdt_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* entropy of label counts */
static float hdt_entropy(const int* labels,const int* idx,int count,
                          int n_classes){
    if(count<=0)return 0;
    int counts[16]={0};
    for(int i=0;i<count&&i<4096;i++){
        int li=idx[i];
        if(labels[li]>=0&&labels[li]<n_classes)counts[labels[li]]++;
    }
    float H=0;
    for(int k=0;k<n_classes&&k<16;k++){
        if(counts[k]<=0)continue;
        float p=(float)counts[k]/count;
        H-=p*log2f(p);
    }
    return H;
}

int wubu_hdt_build(WubuHDT* tree,const float* pts,const int* labels,
                   int n,int D,float c,int max_depth){
    if(!pts||!labels||!tree||n<1)return -1;
    tree->n=n;tree->D=D;tree->c=c;
    tree->nodes=malloc(sizeof(WubuHDTNode)*(size_t)(2*n));
    if(!tree->nodes)return -2;
    memset(tree->nodes,0,sizeof(WubuHDTNode)*(size_t)(2*n));
    tree->used=0;

    /* index array: all points at root */
    int* idx=malloc(sizeof(int)*(size_t)n);
    for(int i=0;i<n;i++)idx[i]=i;

    /* recursive build via explicit stack */
    struct Frame{int node_idx;int* idx;int count;int depth;};
    struct Frame stack[64];
    int sp=0;
    int root=tree->used++;
    stack[sp++]=(struct Frame){root,idx,n,0};

    while(sp>0){
        struct Frame f=stack[--sp];
        WubuHDTNode* node=&tree->nodes[f.node_idx];

        /* leaf conditions */
        int n_classes_present=0,last=-1,same=1;
        for(int i=0;i<f.count;i++){
            int l=labels[f.idx[i]];
            if(l!=last){if(last!=-1)same=0;n_classes_present++;last=l;}
        }
        if(f.depth>=max_depth||same||f.count<4){
            node->is_leaf=1;
            node->label=last;
            free(f.idx);
            continue;
        }

        /* find best anchor/threshold split */
        float parent_H=hdt_entropy(labels,f.idx,f.count,16);
        float best_gain=1e-4f;   /* min gain to split */
        float best_anchor[64],best_thresh=0;
        int found=0;

        for(int cand=0;cand<f.count;cand++){
            const float* anc=pts+(size_t)f.idx[cand]*D;
            /* distances from this anchor to all points in node */
            float dists[256];
            int cnt=f.count<256?f.count:256;
            for(int i=0;i<cnt;i++)
                dists[i]=hdt_dist(anc,pts+(size_t)f.idx[i]*D,D,c);
            /* candidate thresholds: midpoints between consecutive sorted */
            float sorted[256];
            memcpy(sorted,dists,sizeof(float)*cnt);
            for(int a=0;a<cnt;a++)
                for(int b=a+1;b<cnt;b++)
                    if(sorted[b]<sorted[a]){float t=sorted[a];sorted[a]=sorted[b];sorted[b]=t;}
            for(int t_i=0;t_i<cnt-1;t_i++){
                float thr=(sorted[t_i]+sorted[t_i+1])*0.5f;
                /* partition by thr, compute info gain */
                int left_n=0;
                for(int i=0;i<cnt;i++)if(dists[i]<thr)left_n++;
                if(left_n==0||left_n==cnt)continue;
                /* build left/right index lists */
                int* li=malloc(sizeof(int)*(size_t)left_n);
                int* ri=malloc(sizeof(int)*(size_t)(cnt-left_n));
                int lc=0,rc=0;
                for(int i=0;i<cnt;i++){
                    if(dists[i]<thr)li[lc++]=f.idx[i];
                    else ri[rc++]=f.idx[i];
                }
                float Hl=hdt_entropy(labels,li,lc,16);
                float Hr=hdt_entropy(labels,ri,rc,16);
                float gain=parent_H-((float)lc/cnt)*Hl-((float)rc/cnt)*Hr;
                free(li);free(ri);
                if(gain>best_gain){
                    best_gain=gain;
                    int lim=D<64?D:64;
                    for(int d2=0;d2<lim;d2++)best_anchor[d2]=anc[d2];
                    best_thresh=thr;
                    found=1;
                }
            }
        }

        if(!found){
            node->is_leaf=1;
            node->label=last;
            free(f.idx);
            continue;
        }

        /* apply best split */
        node->is_leaf=0;
        memcpy(node->anchor,best_anchor,sizeof(float)*D);
        node->threshold=best_thresh;

        int* li=malloc(sizeof(int)*(size_t)f.count);
        int* ri=malloc(sizeof(int)*(size_t)f.count);
        int lc=0,rc=0;
        for(int i=0;i<f.count;i++){
            float d=hdt_dist(pts+(size_t)f.idx[i]*D,best_anchor,D,c);
            if(d<best_thresh)li[lc++]=f.idx[i];
            else ri[rc++]=f.idx[i];
        }
        int li_node=tree->used++;
        int ri_node=tree->used++;
        node->left=li_node;node->right=ri_node;
        /* NOTE: node pointer may be invalidated by tree->used++ only if
         * realloc'd — we preallocated 2n so it's stable. */
        stack[sp++]=(struct Frame){li_node,li,lc,f.depth+1};
        stack[sp++]=(struct Frame){ri_node,ri,rc,f.depth+1};
        free(f.idx);
    }
    /* NOTE: root frame's idx was freed when the root frame was processed
     * (either as leaf or after split). Do NOT free again here. */
    return 0;
}

int wubu_hdt_predict(const WubuHDT* tree,const float* x){
    int cur=0;
    while(!tree->nodes[cur].is_leaf){
        const WubuHDTNode* nd=&tree->nodes[cur];
        float d=hdt_dist(x,nd->anchor,tree->D,tree->c);
        cur=(d<nd->threshold)?nd->left:nd->right;
    }
    return tree->nodes[cur].label;
}

void wubu_hdt_free(WubuHDT* tree){
    free(tree->nodes);tree->nodes=NULL;tree->used=0;
}
