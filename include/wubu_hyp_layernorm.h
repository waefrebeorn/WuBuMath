/* wubu_hyp_layernorm.h */
#ifndef WUBU_HYP_LAYERNORM_H
#define WUBU_HYP_LAYERNORM_H
#include <stddef.h>
/* Layer norm in tangent space at origin (Euclidean) then map back via exp */
void wubu_hyp_layernorm(const double *x, double *out, size_t dim, double eps);
#endif