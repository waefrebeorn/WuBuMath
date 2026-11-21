/* wubu_nested_transformer.h */
#ifndef WUBU_NESTED_TRANSFORMER_H
#define WUBU_NESTED_TRANSFORMER_H
#include <stddef.h>
/* Nested hyperbolic transformer: apply SO(4) rotation between levels.
   levels: number of nesting levels. */
void wubu_nested_transformer(const double *x, int levels, const double *qL, const double *qR, double *out, size_t dim);
#endif