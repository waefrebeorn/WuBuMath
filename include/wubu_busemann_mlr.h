/* wubu_busemann_mlr.h */
#ifndef WUBU_BUSEMANN_MLR_H
#define WUBU_BUSEMANN_MLR_H
#include <stddef.h>
/* Busemann MLR: hyperplane via Busemann function (point-to-horosphere distance).
   logits_k = <a_k, x>_L + b_k (simplified). */
void wubu_busemann_mlr(const double *x, const double *a, const double *b, int ncls, size_t dim, double *probs);
#endif