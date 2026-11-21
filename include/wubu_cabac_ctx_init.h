/* wubu_cabac_ctx_init.h */
#ifndef WUBU_CABAC_CTX_INIT_H
#define WUBU_CABAC_CTX_INIT_H
#include <stddef.h>
/* QP-dependent context initialization value (Fraunhofer prototype).
   Returns initial probability (0..1) for a context given QP and slice type. */
double wubu_cabac_init_prob(int qp, int slice_type, int ctx_idx);
#endif