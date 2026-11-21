/* wubu_product_space.h */
#ifndef WUBU_PRODUCT_SPACE_H
#define WUBU_PRODUCT_SPACE_H
#include <stddef.h>
/* Mixed-curvature product: concatenate Euclidean + hyperbolic + spherical components.
   out_dim = e_dim + h_dim + s_dim. Just copies. */
void wubu_product_pack(const double *e, size_t ed, const double *h, size_t hd, const double *s, size_t sd, double *out);
#endif