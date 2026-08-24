/*
 * wubu_ldirect.c -- GAP-C022: Tangent-space-free Lorentz linear layer
 *
 * Research source: SRBGCN (BMVC 2023) + H2H-GCN (arXiv:2104.06942).
 * The HGCN approach (C017) does log₀→matmul→exp₀ — a round-trip through
 * the tangent space at origin. The fully-hyperbolic approach instead:
 *
 *   1. split x into (time, space) = (x₀, x₁..xₙ)
 *   2. apply W to the SPACE part only: s' = W · x_space
 *   3. recompute time from the hyperboloid constraint:
 *        x'_time = sqrt(1/c + |s'|²)
 *   4. output = (x'_time, s')
 *
 * No exp/log maps are used — the result stays on the hyperboloid because
 * the time component is derived from the defining equation. This is
 * faster (no trig), more stable (no boundary singularity), and the
 * SRBGCN paper shows it outperforms tangent-space approaches.
 */

#include "wubu_ldirect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_ldirect_forward(const float* W,       /* [D_out, D_in] */
                         const float* b_space, /* [D_out] or NULL */
                         const float* x,       /* [N, D_in+1] hyperboloid pts */
                         int N,int D_in,int D_out,
                         float c,
                         float* out            /* [N, D_out+1] */
){
    if(!W||!x||!out||D_in<1||D_out<1)return -1;

    for(int i=0;i<N;i++){
        const float* xi=x+(size_t)i*(D_in+1);
        float* oi=out+(size_t)i*(D_out+1);

        /* Step 1: extract space components */
        const float* x_space=xi+1;

        /* Step 2: transform space with W: s'[j] = sum_k W[j,k]*x_space[k] */
        float sp[256];int dout=D_out<256?D_out:256;
        for(int j=0;j<dout;j++){
            float acc=0;
            for(int k=0;k<D_in;k++)acc+=W[(size_t)j*D_in+k]*x_space[k];
            sp[j]=acc;
        }
        /* add bias if present */
        if(b_space)
            for(int j=0;j<dout;j++)sp[j]+=b_space[j];

        /* Step 3: recompute time from hyperboloid constraint:
         * x_time = sqrt(1/c + |s'|²)  (Eq. 3 in MERU/HGCN) */
        float n2=0;
        for(int j=0;j<dout;j++)n2+=sp[j]*sp[j];
        float time_val=sqrtf(1.0f/c+n2);

        /* Step 4: write output */
        oi[0]=time_val;
        for(int j=0;j<D_out;j++)oi[j+1]=sp[j];
    }
    return 0;
}
