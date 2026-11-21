#include "wubu_nested_ball.h"
#include "wubu_mobius_scalar.h"
#include <math.h>

void wubu_nested_scale(const double *x, int level, double *out, size_t dim) {
    double phi = (1.0+sqrt(5.0))/2.0;
    double r = pow(phi, (double)level);
    wubu_mobius_scalar(r, x, out, dim, 1.0);
}