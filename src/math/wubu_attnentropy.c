/*
 * wubu_attnentropy.c -- GAP-B017: Attention entropy monitor
 * (collapse detection for the hyperbolic transformer)
 *
 * Research source: Zhai et al. ICML 2023 (sigmaReparam, attention
 * entropy collapse) — low attention entropy predicts training
 * instability; entropy lower bound decays exponentially with logit norm.
 *
 * For each attention row (a probability distribution over positions),
 * compute Shannon entropy H = -Σ p log p. The monitor reports:
 *   - mean row entropy across heads/layers
 *   - collapse flag: mean H < threshold (e.g. 0.1*log(N))
 */
#include "wubu_attnentropy.h"
#include <math.h>
#include <stddef.h>

float wubu_ae_row_entropy(const float* row,int N){
    float h=0;
    for(int j=0;j<N;j++){
        float p=row[j];
        if(p>1e-12f)h-=p*logf(p);
    }
    return h;
}

float wubu_ae_mean_entropy(const float* attn,int N,int n_heads){
    /* attn: [n_heads, N, N] block-row probabilities.
     * Mean of per-row entropies across all heads and queries. */
    double total=0;
    long count=0;
    int rows=n_heads*N;
    for(int r=0;r<rows;r++)
        total+=wubu_ae_row_entropy(attn+(size_t)r*N,N),count++;
    return count?(float)(total/count):0.0f;
}

int wubu_ae_collapsed(const float* attn,int N,int n_heads,float thresh){
    float h=wubu_ae_mean_entropy(attn,N,n_heads);
    return h<thresh?1:0;
}
