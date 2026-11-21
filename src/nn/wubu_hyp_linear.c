#include "wubu_hyp_linear.h"
void wubu_hyp_linear(const double *x, const double *W, const double *b, size_t in_dim, size_t out_dim, double *out) {
    for (size_t o = 0; o < out_dim; o++) {
        double s = b ? b[o] : 0;
        for (size_t i = 0; i < in_dim; i++) s += W[o*in_dim+i]*x[i];
        out[o] = s;
    }
}