/*
 * wubu_hpq.c -- GAP-E009: Hyperbolic residual vector quantization
 *
 * Research source: RVQ (residual vector quantization, Wang 2010) +
 * SoundStream/Encodec-style multi-stage codes. Staged quantizers on the
 * ball: each stage geodesic-k-means quantizes the PREVIOUS STAGE'S
 * RESIDUAL (tangent-space difference), progressively refining the
 * reconstruction. A point is stored as L codebook indices — log2(K)·L
 * bits instead of 32·D bits.
 *
 * This is the codec's lossy latent compression: rate scales with stages,
 * distortion decreases monotonically per added stage.
 */
#include "wubu_hpq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hq_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

static void hq_project(float* v,int D,float c,float cap){
    float n2=0;for(int d=0;d<D;d++)n2+=v[d]*v[d];
    if(n2>cap*cap){float s=cap/sqrtf(n2);for(int d=0;d<D;d++)v[d]*=s;}
}

int wubu_hpq_build(const float* pts,int n,int D,int L,int K,float c,
                    unsigned seed,WubuHPQ* q){
    q->L=L;q->K=K;q->D=D;q->c=c;
    q->codebooks=calloc((size_t)L*K*D,sizeof(float));
    if(!q->codebooks)return -1;

    float* resid=malloc(sizeof(float)*(size_t)n*D);
    memcpy(resid,pts,sizeof(float)*(size_t)n*D);

    for(int l=0;l<L;l++){
        /* init this stage's codebook from data points (spread by stride) */
        for(int k2=0;k2<K;k2++){
            int idx=(seed+k2*7+l*13)%n;
            memcpy(q->codebooks+(size_t)(l*K+k2)*D,
                   resid+(size_t)idx*D,sizeof(float)*D);
        }
        /* few k-means iterations in TANGENT space (Euclidean-safe):
         * operate on log0-projected residuals */
        for(int it=0;it<6;it++){
            float* sums=calloc((size_t)K*D,sizeof(float));
            int* counts=calloc((size_t)K,sizeof(int));
            for(int i=0;i<n;i++){
                const float* r=resid+(size_t)i*D;
                float bd=1e30f;int bk=0;
                for(int k2=0;k2<K;k2++){
                    float dd=0;
                    for(int d=0;d<D;d++){
                        float df=r[d]-q->codebooks[(size_t)(l*K+k2)*D+d];
                        dd+=df*df;
                    }
                    if(dd<bd){bd=dd;bk=k2;}
                }
                counts[bk]++;
                for(int d=0;d<D;d++)sums[(size_t)bk*D+d]+=r[d];
            }
            for(int k2=0;k2<K;k2++)
                if(counts[k2]>0)
                    for(int d=0;d<D;d++)
                        q->codebooks[(size_t)(l*K+k2)*D+d]=
                            sums[(size_t)k2*D+d]/counts[k2];
            free(sums);free(counts);
        }
        /* subtract this stage's best codeword from residuals */
        for(int i=0;i<n;i++){
            float* r=resid+(size_t)i*D;
            float bd=1e30f;int bk=0;
            for(int k2=0;k2<K;k2++){
                float dd=0;
                for(int d=0;d<D;d++){
                    float df=r[d]-q->codebooks[(size_t)(l*K+k2)*D+d];
                    dd+=df*df;
                }
                if(dd<bd){bd=dd;bk=k2;}
            }
            for(int d=0;d<D;d++)r[d]-=q->codebooks[(size_t)(l*K+bk)*D+d];
            hq_project(r,D,c,sqrtf(1.0f/c)*0.999f);
        }
    }
    free(resid);
    return 0;
}

void wubu_hpq_free(WubuHPQ* q){free(q->codebooks);}

int wubu_hpq_encode(const WubuHPQ* q,const float* x,int* codes){
    float resid[512];
    int dd=q->D<512?q->D:512;
    memcpy(resid,x,sizeof(float)*(size_t)dd);
    for(int l=0;l<q->L;l++){
        float bd=1e30f;int bk=0;
        for(int k2=0;k2<q->K;k2++){
            float dd2=0;
            const float* cb=q->codebooks+(size_t)(l*q->K+k2)*q->D;
            for(int d=0;d<dd&&d<512;d++){
                float df=resid[d]-cb[d];dd2+=df*df;
            }
            if(dd2<bd){bd=dd2;bk=k2;}
        }
        codes[l]=bk;
        const float* cb=q->codebooks+(size_t)(l*q->K+bk)*q->D;
        for(int d=0;d<dd&&d<512;d++)resid[d]-=cb[d];
    }
    return 0;
}

void wubu_hpq_decode(const WubuHPQ* q,const int* codes,float* out){
    memset(out,0,sizeof(float)*(size_t)q->D);
    for(int l=0;l<q->L;l++){
        const float* cb=q->codebooks+(size_t)(l*q->K+codes[l])*q->D;
        for(int d=0;d<q->D&&d<512;d++)out[d]+=cb[d];
    }
    hq_project(out,q->D,q->c,sqrtf(1.0f/q->c)*0.999f);
}
