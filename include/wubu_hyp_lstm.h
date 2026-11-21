/* wubu_hyp_lstm.h */
#ifndef WUBU_HYP_LSTM_H
#define WUBU_HYP_LSTM_H
#include <stddef.h>
/* Hyperbolic LSTM cell (simplified). h,c updated, projected to ball. */
void wubu_hyp_lstm_step(const double *x, const double *h, const double *c, double *hout, double *cout, size_t dim);
#endif