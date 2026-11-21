#include "wubu_curvature_schedule.h"
#include <math.h>
double wubu_curvature_schedule(int step, int warmup, double K_init, double K_min, double K_max) {
    if (step < warmup) return K_init;
    double t = (double)(step - warmup) / 1000.0;
    double K = K_init * exp(-0.1*t);
    if (K < K_min) K = K_min;
    if (K > K_max) K = K_max;
    return K;
}