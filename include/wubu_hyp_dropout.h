/* wubu_hyp_dropout.h */
#ifndef WUBU_HYP_DROPOUT_H
#define WUBU_HYP_DROPOUT_H
#include <stddef.h>
/* Apply dropout in tangent space at origin: zero out with prob p, scale */
void wubu_hyp_dropout(double *x, size_t dim, double p, unsigned int *rng);
#endif