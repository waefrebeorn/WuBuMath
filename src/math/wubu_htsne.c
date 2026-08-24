/*
 * wubu_htsne.c -- GAP-D026: Hyperbolic t-SNE (CO-SNE simplified)
 *
 * Research source: CO-SNE (arXiv:2104.03505) + hyperbolic t-SNE
 * (TUDelft 2024 acceleration paper). The joint probability P_ij is
 * computed in HIGH-DIMENSIONAL EUCLIDEAN space (input similarities),
 * while Q_ij uses HYPERBOLIC Student-t probabilities on the ball:
 *
 *   q_ij ∝ [1 + d_c(y_i,y_j)/√c]^(-1/τ)
 *
 * Minimize KL(P||Q) by gradient descent with boundary-guarded steps.
 * τ=1 gives the heavy-tailed hyperbolic distribution that respects the
 * exponential volume growth — clusters don't collapse at the boundary.
 */
#include "wubu_htsne.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* high-dim P: Gaussian affinities with fixed perplexity-scaled sigma */
int wubu_htsne_p_high(const float* xs,int n,int D_in,float* P){
    if(!xs||!P||n<2)return -1;
    float* dists=malloc(sizeof(float)*(size_t)n*n);
    if(!dists)return -2;
    /* pairwise euclidean distances in input space */
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float d2=0;
            for(int d=0;d<D_in;d++){
                float df=xs[(size_t)i*D_in+d]-xs[(size_t)j*D_in+d];
                d2+=df*df;
            }
            float d=sqrtf(d2);
            dists[(size_t)i*n+j]=dists[(size_t)j*n+i]=d;
        }

    /* GAP-D026 fix: per-point LOCAL bandwidth = distance to k-th nearest
     * neighbor. Global median sigma blurs blob structure; local scaling
     * preserves it — what real t-SNE does with perplexity. */
    int kk=5<n-1?5:n-1;
    float* sigmas=malloc(sizeof(float)*(size_t)n);
    if(!sigmas){free(dists);return -3;}
    for(int i=0;i<n;i++){
        float row[256];int cnt=n-1<256?n-1:256;
        int cc=0;
        for(int j=0;j<n&&cc<cnt;j++){
            if(j==i)continue;
            row[cc++]=dists[(size_t)i*n+j];
        }
        for(int a=0;a<kk&&a<cnt;a++)
            for(int b=a+1;b<cnt;b++)
                if(row[b]<row[a]){float t=row[a];row[a]=row[b];row[b]=t;}
        sigmas[i]=(kk<cnt)?row[kk]:row[cnt-1];
        sigmas[i]+=1e-6f;
    }

    /* asymmetric p_j|i with per-point bandwidth, then symmetrize */
    for(int i=0;i<n;i++){
        float z=0,sig=sigmas[i];
        for(int j=0;j<n;j++){
            if(i==j){P[(size_t)i*n+j]=0;continue;}
            float v=expf(-dists[(size_t)i*n+j]*dists[(size_t)i*n+j]/(2*sig*sig));
            P[(size_t)i*n+j]=v;z+=v;
        }
        if(z>0)for(int j=0;j<n;j++)P[(size_t)i*n+j]/=z;
    }
    free(sigmas);
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float p=(P[(size_t)i*n+j]+P[(size_t)j*n+i])/(2*n);
            P[(size_t)i*n+j]=P[(size_t)j*n+i]=p;
        }
    free(dists);
    return 0;
}

/* hyperbolic student-t q_ij on the ball */
void wubu_htsne_q_low(const float* ys,int n,int D,float c,float* Q){
    float z=0;
    for(int i=0;i<n;i++)Q[(size_t)i*n+i]=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float ab2=0,a2=0,b2=0;
            for(int d=0;d<D;d++){
                float df=ys[(size_t)i*D+d]-ys[(size_t)j*D+d];ab2+=df*df;
                a2+=ys[(size_t)i*D+d]*ys[(size_t)i*D+d];
                b2+=ys[(size_t)j*D+d]*ys[(size_t)j*D+d];
            }
            float den=(1-c*a2)*(1-c*b2);
            if(den<1e-9f)den=1e-9f;
            float dc=acoshf(1+2*c*ab2/den)/sqrtf(c);
            /* tau=1: q ∝ 1/(1+dc) — heavy-tailed in hyperbolic distance */
            float q=1.0f/(1.0f+dc);
            Q[(size_t)i*n+j]=Q[(size_t)j*n+i]=q;
            z+=2*q;
        }
    if(z>0)for(int i=0;i<n*n;i++)Q[i]/=z;
}

float wubu_htsne_kl(const float* P,const float* Q,int n){
    double kl=0;
    for(int i=0;i<n*n;i++)
        if(P[i]>1e-12f&&Q[i]>1e-12f)
            kl+=(double)P[i]*log((double)P[i]/(double)Q[i]);
    return (float)kl;
}
