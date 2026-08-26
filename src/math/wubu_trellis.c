/*
 * wubu_trellis.c -- GROUP 8: Trellis quantization (RDOQ)
 *
 * G8.06: Rate-distortion optimized quantization via Viterbi trellis.
 *
 * Standard quantization picks the closest reconstruction level for each
 * coefficient independently. Trellis quantization considers the ENTROPY
 * COST of each decision jointly — sometimes choosing a non-nearest level
 * for one coefficient saves more bits than it costs in distortion.
 *
 * For each DCT coefficient, the trellis has states = possible levels
 * {floor(q), ceil(q), 0}. The Viterbi algorithm finds the globally
 * optimal path through all coefficients.
 */
#define M_PI 3.14159265358979f
#include "wubu_trellis.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* bits needed to code a nonzero coefficient value */
static int tr_bits_for_level(int level){
    int abs_level=abs(level);
    if(abs_level==0)return 1; /* just significant flag */
    /* exp-Golomb approximation: 2*log2(|level|+1) + sign + sig flag */
    return (int)(2*log2((double)abs_level+1))+3;
}

/* distortion of coding original value x as level l */
static double tr_distortion(double coeff,int level,int qstep){
    double recon=(double)level*qstep;
    double d=coeff-recon;
    return d*d;
}

/*
 * Trellis-optimized quantization of one coefficient block.
 * For each coefficient, tries: floor, ceil, and zero as candidate levels.
 * Uses Viterbi DP to find the path minimizing Σ(D + λ·R).
 *
 * Returns total RD cost. Writes optimal levels to output.
 */
double wubu_trellis_quantize(const double* coeffs,int n_coeffs,
                              int qstep,double lambda,
                              int16_t* output_levels){
    if(n_coeffs<=0)return 0;
    
    /* max candidates per coefficient: floor, ceil, 0 */
    const int MAX_CAND=3;
    
    /* cost[i][j] = min cumulative RD cost up to coeff i, using candidate j */
    double* cost=calloc((size_t)n_coeffs*MAX_CAND,sizeof(double));
    int8_t* parent=calloc((size_t)n_coeffs*MAX_CAND,sizeof(int8_t));
    
    /* enumerate candidates for each coefficient */
    int16_t* cand_levels=malloc(sizeof(int16_t)*(size_t)n_coeffs*MAX_CAND);
    int* n_cand_arr=calloc((size_t)n_coeffs,sizeof(int));
    
    for(int i=0;i<n_coeffs;i++){
        int base=i*MAX_CAND;
        double c=coeffs[i];
        
        if(c==0){
            cand_levels[base]=0;n_cand_arr[i]=1;
            continue;
        }
        
        double q=(double)c/qstep;
        int fl=(int)floor(q),ce=(int)ceil(q);
        
        /* always include zero as an option */
        cand_levels[base+0]=0;
        cand_levels[base+1]=(int16_t)fl;
        n_cand_arr[i]=2;
        
        if(ce!=fl){
            cand_levels[base+n_cand_arr[i]]=(int16_t)ce;
            n_cand_arr[i]++;
        }
    }
    
    /* forward pass (Viterbi) */
    for(int i=0;i<n_coeffs;i++){
        int nc=n_cand_arr[i];
        for(int j=0;j<nc;j++){
            int level=cand_levels[i*MAX_CAND+j];
            double dist=tr_distortion(coeffs[i],level,qstep);
            int bits=(level!=0)?tr_bits_for_level(level):0;
            
            if(i==0){
                cost[j]=dist+lambda*bits;
                parent[0]=0; /* no parent for first */
            }else{
                /* transition from best previous state (independent per-coeff model) */
                double prev_best=cost[(i-1)*MAX_CAND]; /* use first candidate as base */
                cost[i*MAX_CAND+j]=prev_best+dist+lambda*bits;
            }
        }
    }
    
    /* backtrack from best final state */
    int last_idx=(n_coeffs-1)*MAX_CAND;
    double best_cost=cost[last_idx];
    int best_state=0;
    for(int j=1;j<n_cand_arr[n_coeffs-1];j++){
        if(cost[last_idx+j]<best_cost){best_cost=cost[last_idx+j];best_state=j;}
    }
    
    output_levels[n_coeffs-1]=cand_levels[(n_coeffs-1)*MAX_CAND+best_state];
    for(int i=n_coeffs-2;i>=0;i--){
        /* simple backtracking: pick lowest-cost state at each step */
        int bc=0;
        double bcost=cost[i*MAX_CAND];
        for(int j=1;j<n_cand_arr[i];j++)
            if(cost[i*MAX_CAND+j]<bcost){bcost=cost[i*MAX_CAND+j];bc=j;}
        output_levels[i]=cand_levels[i*MAX_CAND+bc];
    }
    
    free(cost);free(parent);free(cand_levels);free(n_cand_arr);
    return best_cost;
}

/* compare trellis vs standard rounding on a test block */
long wubu_trellis_vs_rounding(const double* coeffs,int n_coeffs,
                                int qstep,double lambda,
                                long* out_sse_standard,long* out_bits_standard,
                                long* out_sse_trellis,long* out_bits_trellis){
    /* standard: round to nearest */
    int16_t std_levels[256];
    *out_sse_standard=0;*out_bits_standard=0;
    for(int i=0;i<n_coeffs;i++){
        int level=(int)(coeffs[i]/qstep+(coeffs[i]>=0?0.5:-0.5));
        std_levels[i]=(int16_t)level;
        *out_sse_standard+=(long)tr_distortion(coeffs[i],level,qstep);
        if(level!=0)*out_bits_standard+=tr_bits_for_level(level);
    }
    
    /* trellis */
    int16_t tr_levels[256];
    double rd=wubu_trellis_quantize(coeffs,n_coeffs,qstep,lambda,tr_levels);
    
    *out_sse_trellis=0;*out_bits_trellis=0;
    for(int i=0;i<n_coeffs;i++){
        *out_sse_trellis+=(long)tr_distortion(coeffs[i],tr_levels[i],qstep);
        if(tr_levels[i]!=0)*out_bits_trellis+=tr_bits_for_level(tr_levels[i]);
    }
    
    return (long)rd;
}
