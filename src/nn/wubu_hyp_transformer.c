#include "wubu_hyp_transformer.h"
#include "wubu_gated_hyp_attn.h"
#include "wubu_hyp_layernorm.h"
#include "wubu_hyp_residual.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void wubu_hyp_transformer_block(const double *x, const double *Wg, const double **keys, const double **vals,
    int nkeys, size_t dim, const double *Wffn, double *out) {
    /* gated attention */
    double *attn = malloc(dim*sizeof(double));
    wubu_gated_hyp_attn(x, Wg, keys, vals, nkeys, dim, attn);
    /* residual */
    double *res1 = malloc(dim*sizeof(double));
    wubu_hyp_residual(x, attn, res1, dim);
    /* layernorm */
    double *normed = malloc(dim*sizeof(double));
    wubu_hyp_layernorm(res1, normed, dim, 1e-5);
    /* FFN: simple linear + tanh */
    for (size_t d = 0; d < dim; d++) {
        double s = 0;
        for (size_t i = 0; i < dim; i++) s += Wffn[d*dim+i]*normed[i];
        out[d] = tanh(s);
    }
    /* residual 2 */
    double *res2 = malloc(dim*sizeof(double));
    wubu_hyp_residual(normed, out, res2, dim);
    memcpy(out, res2, dim*sizeof(double));
    free(attn); free(res1); free(normed); free(res2);
}