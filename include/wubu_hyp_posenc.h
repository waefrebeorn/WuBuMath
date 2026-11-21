/* wubu_hyp_posenc.h */
#ifndef WUBU_HYP_POSENC_H
#define WUBU_HYP_POSENC_H
#include <stddef.h>
/* Add positional encoding (sinusoidal) to point x in ball, project */
void wubu_hyp_posenc(double *x, size_t dim, int pos);
#endif