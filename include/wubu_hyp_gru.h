/* wubu_hyp_gru.h */
#ifndef WUBU_HYP_GRU_H
#define WUBU_HYP_GRU_H
#include <stddef.h>
/* Simplified hyperbolic GRU: h' = (1-z)*h + z*tanh(Wx+b) with projection to ball */
void wubu_hyp_gru_step(const double *x, const double *h, double *hout, size_t dim,
                       const double *Wz, const double *Wr, const double *Wh, const double *bz);
#endif