#include "wubu_hyp_rnn.h"
#include <math.h>
#include <string.h>
void wubu_hyp_rnn_cell(const double *x, const double *h_prev, const double *Wxh, const double *Whh, const double *b, size_t dim, double *h_out) {
    for (size_t i = 0; i < dim; i++) {
        double s = b ? b[i] : 0;
        for (size_t j = 0; j < dim; j++) s += Wxh[i*dim+j]*x[j] + Whh[i*dim+j]*h_prev[j];
        h_out[i] = tanh(s);
    }
}