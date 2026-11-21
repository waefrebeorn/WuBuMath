/* wubu_hyp_rnn.h */
#ifndef WUBU_HYP_RNN_H
#define WUBU_HYP_RNN_H
#include <stddef.h>
/* Simple RNN cell: h_t = tanh(W_hh*h_{t-1} + W_xh*x_t + b) */
void wubu_hyp_rnn_cell(const double *x, const double *h_prev, const double *Wxh, const double *Whh, const double *b, size_t dim, double *h_out);
#endif