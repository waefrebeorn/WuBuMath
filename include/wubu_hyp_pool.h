/* wubu_hyp_pool.h */
#ifndef WUBU_HYP_POOL_H
#define WUBU_HYP_POOL_H
#include <stddef.h>
/* Einstein (gyro) midpoint pooling over n points in Poincare ball (dim) */
void wubu_hyp_pool(const double *pts, int n, size_t dim, double *out);
#endif