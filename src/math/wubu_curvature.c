#include "wubu_curvature.h"

void wubu_curvature_step(double *K, double grad, double lr, double Kmin, double Kmax) {
    *K -= lr * grad;
    if (*K < Kmin) *K = Kmin;
    if (*K > Kmax) *K = Kmax;
}